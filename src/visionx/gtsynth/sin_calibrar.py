"""E0 de PLUG-AND-PLAY: enchufar una cámara distinta suponiendo que es la ELP.

Para cada recorrido de `data/gt-video` (una cámara, una escena) compara tres versiones de la
misma «otra unidad» (hermana ×1, como en los videos), con CREStereo continuo:

    calibrada            el sistema conoce la cámara: clásico + distancia conocida (lo de hoy)
    sin calibrar         supone la geometría de la ELP (116°, 5,98 cm) y no corrige nada
    sin calibrar + clás. supone la ELP, corre el clásico desde ahí y usa la distancia conocida
    … + escala           lo mismo, pero la distancia conocida corrige la escala (f·B) en vez de
                         restar un corrimiento (PLUG-AND-PLAY, sección 5)

E3 (con `red`, la estimación de la red de entrada de `camara_net evaluar`): en vez de suponer la
ELP, el sistema arranca de la focal y el centro que da la red en los primeros cuadros del
recorrido:

    red + clásico + escala       separación desconocida (la de la ELP); un objeto corrige f·B
    red + clásico + separación   separación conocida (la de la hoja de datos), sin objeto: la
                                 escala queda en manos de la focal de la red
    red + escala                 sin clásico, para ver cuánto aporta alinear

Es la línea de base del plug and play: cuánto se pierde cuando no se sabe qué cámara es.

    python -m visionx.gtsynth sin-calibrar data/gt-video --salida outputs/gtsynth/e0.json
"""

from __future__ import annotations

import dataclasses
import json
import math
import zlib
from pathlib import Path

import cv2
import numpy as np

from visionx.gtsynth.dataset import vistas
from visionx.gtsynth.video import (
    CALENTAR,
    _camara_nominal,
    _delta_conocida,
    _delta_de_cuadros,
    _error,
    _hermana,
    _z,
)

ELP_HFOV_DEG = 116.0
ELP_BASE_M = 0.0598


def geometria_elp(g):
    """La geometría que supone el sistema si cree que la cámara es una ELP (misma resolución)."""
    fx = (g.width / 2.0) / math.tan(math.radians(ELP_HFOV_DEG) / 2.0)
    return dataclasses.replace(g, fx=fx, fy=fx, cx=g.width / 2.0, cy=g.height / 2.0,
                               baseline_m=ELP_BASE_M, doffs=0.0)  # fmt: skip


VERSIONES = ("calibrada", "sin calibrar", "sin calibrar + clásico", "sin calibrar + escala",
             "sin calibrar + clásico + escala", "red + clásico + escala",
             "red + clásico + separación", "red + escala")  # fmt: skip


def leer_red(carpeta: str | Path) -> dict[str, dict]:
    """Por vista (carpeta resuelta): media y b de la red, de los CSV de `camara_net evaluar`."""
    import csv

    out: dict[str, dict] = {}
    for p in sorted(Path(carpeta).glob("video-*.csv")):
        with open(p, encoding="utf-8", newline="") as fh:
            for f in csv.DictReader(fh):
                out[str(Path(f["carpeta"]).resolve())] = {
                    "ancho": int(f["ancho"]), "alto": int(f["alto"]),
                    "mu": np.array([float(f[f"mu_{n}"]) for n in ("log_f", "cx", "cy")]),
                    "b": np.array([float(f[f"b_{n}"]) for n in ("log_f", "cx", "cy")]),
                }  # fmt: skip
    return out


def camara_de_la_red(vs, red: dict[str, dict], cuadros: int = 8) -> dict | None:
    """Lo que el sistema creería de la cámara mirando los primeros `cuadros` del recorrido.

    Combina los cuadros pesando por 1/b² (como `camara_net.entrenar.combinar`) y devuelve fx, cx
    y cy en px de la imagen nominal, más la b de log f. La distorsión de la red no se usa: los
    renders de `gt-video` son pinhole y la red casi no aprende k1 (BITACORA 22/11).
    """
    filas = [red[k] for k in (str(Path(v.carpeta).resolve()) for v in vs[:cuadros]) if k in red]
    if not filas:
        return None
    mu, b = np.stack([f["mu"] for f in filas]), np.stack([f["b"] for f in filas])
    w = 1.0 / np.maximum(b, 1e-6) ** 2
    m, bm = (w * mu).sum(0) / w.sum(0), 1.0 / np.sqrt(w.sum(0))
    ancho, alto = filas[0]["ancho"], filas[0]["alto"]
    return {"fx": float(ancho * math.exp(m[0])), "cx": float((m[1] + 0.5) * ancho),
            "cy": float((m[2] + 0.5) * alto), "b_log_f": float(bm[0]), "cuadros": len(filas),
            "ancho": ancho}  # fmt: skip


def geometria_red(g, cam: dict, baseline_m: float):
    """La geometría que supone el sistema con la estimación de la red (`cam`) y esa separación."""
    q = g.width / cam["ancho"]  # la red ve la imagen nominal; `g` puede venir recortada a ×8
    return dataclasses.replace(g, fx=cam["fx"] * q, fy=cam["fx"] * q, cx=cam["cx"] * q,
                               cy=cam["cy"] * q, baseline_m=baseline_m, doffs=0.0)  # fmt: skip


def _escala_conocida(depth, g, z_gt: np.ndarray, mascara: np.ndarray) -> float | None:
    """Z real / Z medida del objeto apuntado (mediana sobre sus píxeles): la escala de f·B."""
    h, w = depth.disparity_px.shape
    m = cv2.resize(mascara.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    zg = cv2.resize(np.nan_to_num(z_gt), (w, h), interpolation=cv2.INTER_NEAREST)
    d = depth.disparity_px + depth.doffs
    ok = m & depth.valid_mask & (zg > 0) & (d > 0)
    if ok.sum() < 30:
        return None
    return float(np.median(zg[ok] * d[ok] / (g.fx * g.baseline_m)))


def evaluar(
    vs,
    cfg: dict,
    cada: int = 6,
    versiones: list[str] | None = None,
    red: dict[str, dict] | None = None,
    cuadros_red: int = 8,
    focal_mov: dict | None = None,
) -> dict:
    """Error mediano (%) de las tres versiones en los cuadros `vs[::cada]` de un recorrido."""
    from visionx.depth.crestereo_adapt.align_apply import Alineador
    from visionx.depth.crestereo_adapt.perturb import perturb_right_camera
    from visionx.gtsynth.evaluar import Medidor, _frame

    v0 = vs[0]
    g_lienzo = v0.geometria
    g = v0.geometria_nominal
    w8, h8 = g.width // 8 * 8, g.height // 8 * 8
    g = dataclasses.replace(g, width=w8, height=h8)
    g_elp = geometria_elp(g)
    nominal = _camara_nominal(g_lienzo)
    real = _hermana(zlib.crc32(f"{v0.escena}/{v0.variante}".encode()), nominal)

    def recorte(img):
        return v0.recortar(img)[:h8, :w8]

    idx = list(range(0, len(vs), cada))
    paso = max(1, len(vs) // 8)
    izq = {i: recorte(vs[i].izq()) for i in sorted(set(idx) | set(range(0, len(vs), paso)))}
    her = {i: recorte(perturb_right_camera(vs[i].der(), real, nominal, g_lienzo).right)
           for i in izq}  # fmt: skip
    zs = {i: recorte(vs[i].z()) for i in idx}
    apuntado = int(v0.meta["apuntado"]["id"])
    mascaras = {i: recorte(vs[i].instancias()) == apuntado for i in idx}

    def correr(geo, clasico: bool, referencia: str | None) -> tuple[float | None, float]:
        est = Medidor.armar("crestereo", cfg, geo).estimador
        dg = getattr(est, "geometry", geo)
        mapas = None
        if clasico:
            mapas, _ = Alineador("clasico", geo, np.zeros(5))._clasico(
                [(izq[i], her[i]) for i in range(0, len(vs), paso)]
            )
        deps = {}
        for k, i in enumerate(idx):
            der = her[i] if mapas is None else cv2.remap(her[i], mapas[0], mapas[1],
                                                         cv2.INTER_LINEAR)  # fmt: skip
            deps[i] = est.predict(_frame(vs[i], izq[i], der, geo, k))
        delta, escala = 0.0, 1.0
        if referencia == "corrimiento":
            ds = [_delta_conocida(deps[i], dg, zs[i], mascaras[i]) for i in idx[min(CALENTAR, 1) :]]
            delta = _delta_de_cuadros(ds, dg)
        elif referencia == "escala":
            es = [_escala_conocida(deps[i], dg, zs[i], mascaras[i])
                  for i in idx[min(CALENTAR, 1) :]]  # fmt: skip
            es = [e for e in es if e is not None and e > 0]
            escala = float(np.median(es)) if es else 1.0
        errs = [e for e in (_error(escala * _z(deps[i], dg, w8, h8, delta), zs[i])[2] for i in idx)
                if e is not None]  # fmt: skip
        ajuste = round(escala, 4) if referencia == "escala" else round(delta, 3)
        return (round(float(np.median(errs)), 2) if errs else None), ajuste

    out = {"escena": v0.escena, "variante": v0.variante,
           "hfov_deg": round(math.degrees(2 * math.atan(g.width / (2 * g.fx))), 1),
           "base_cm": round(100 * g.baseline_m, 1), "cuadros": len(idx)}  # fmt: skip
    corridas = [("calibrada", g, True, "corrimiento"), ("sin calibrar", g_elp, False, None),
                ("sin calibrar + clásico", g_elp, True, "corrimiento"),
                ("sin calibrar + escala", g_elp, False, "escala"),
                ("sin calibrar + clásico + escala", g_elp, True, "escala")]  # fmt: skip
    cam = camara_de_la_red(vs, red, cuadros_red) if red else None
    if cam is not None and focal_mov:  # la focal medida con el movimiento (`gtsynth autocalib`)
        fm = focal_mov.get((v0.variante, v0.escena))
        if fm is not None:
            cam = {**cam, "fx": fm * cam["ancho"] / g.width}
    if cam is not None:
        g_red, g_sep = geometria_red(g, cam, ELP_BASE_M), geometria_red(g, cam, g.baseline_m)
        out["red_focal_pct"] = round(100 * abs(g_red.fx / g.fx - 1), 2)
        out["red_b_log_f"] = round(cam["b_log_f"], 5)
        out["red_cx_px"] = round(abs(g_red.cx - g.cx), 1)
        out["red_cuadros"] = cam["cuadros"]
        corridas += [("red + clásico + escala", g_red, True, "escala"),
                     ("red + clásico + separación", g_sep, True, None),
                     ("red + escala", g_red, False, "escala")]  # fmt: skip
    elif red:
        out["red_falta"] = True
    for nombre, geo, clas, ref in corridas:
        if versiones and nombre not in versiones:
            continue
        clave = "escala" if ref == "escala" else "delta"
        sufijo = "" if ref == "escala" else "_px"
        out[f"error_{nombre}_pct"], out[f"{clave}_{nombre}{sufijo}"] = correr(geo, clas, ref)
    return out


def correr_todo(raiz: str | Path, salida: str | Path, variantes: list[str] | None = None,
                cfg: dict | None = None, cada: int = 6,
                escenas: list[str] | None = None,
                versiones: list[str] | None = None, red: str | Path | None = None,
                cuadros_red: int = 8, focal_mov: str | Path | None = None,
                ) -> list[dict]:  # fmt: skip
    from visionx.shared.config import load_config

    cfg = cfg or load_config()
    raiz = Path(raiz)
    est_red = leer_red(red) if red else None
    fm = None
    if focal_mov:  # fx combinada (red + movimiento) por recorrido, en px de la imagen nominal
        filas_fm = json.loads(Path(focal_mov).read_text(encoding="utf-8"))["filas"]
        fm = {(f["variante"], f["escena"]): f["fx_real"] * (1 + f["comb_err_pct"] / 100)
              for f in filas_fm}  # fmt: skip
    variantes = variantes or sorted(p.name for p in raiz.iterdir()
                                    if p.is_dir() and p.name != "video-elp")  # fmt: skip
    filas = []
    for var in variantes:
        por_escena: dict[str, list] = {}
        for v in vistas(raiz, var):
            if escenas is None or v.escena in escenas:
                por_escena.setdefault(v.escena, []).append(v)
        for escena in sorted(por_escena):
            vs = sorted(por_escena[escena], key=lambda v: v.meta["indice"])
            fila = evaluar(vs, cfg, cada, versiones, est_red, cuadros_red, fm)
            filas.append(fila)
            print(json.dumps(fila, ensure_ascii=False), flush=True)
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    Path(salida).write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
    return filas
