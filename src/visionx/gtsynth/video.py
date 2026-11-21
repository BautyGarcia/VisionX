"""Videos de la demo «Cámaras»: un recorrido por una casa con cuatro cámaras distintas.

Cada cámara recorre **el mismo camino** (`render_tanda.py --tipo-pose recorrido`, 90 cuadros) y
cada cuadro pasa por CREStereo continuo (el modo de la app en vivo) en cuatro versiones:

    ideal          el par perfecto con su calibración: el techo
    sin corregir   el ojo derecho de «otra unidad» del mismo modelo (giro, aumento y lente un
                   poco distintos; generador de hermanas de A), con la calibración nominal
    clásico        lo que usa hoy VisionX: alineación clásica con lente (los cuadros juntos) y
                   distancia conocida con el objeto apuntado (mediana de los cuadros ya en caliente)
    red            la red de entrada (v3) encima del clásico: lugar reservado hasta que exista
                   su checkpoint (`--red`)

Por cuadro se dibuja una lámina de 1920×1080 (imagen, distancia real, la de cada versión y su
error) y se guarda un MP4 por escena y cámara. La pestaña del stand solo reproduce los videos.

    python -m visionx.gtsynth video data/gt-video --salida data/stand-video [--red ckpt.pt]
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionx.gtsynth.camaras import Z_MAX, Z_MIN, _color_error, _color_z
from visionx.gtsynth.dataset import Vista, vistas

CAMARAS_VIDEO: tuple[tuple[str, str], ...] = (
    # La ELP a resolución completa, como corre la app: a media resolución la regla del clásico
    # (ajustada a 1920×1080) dejaba pasar términos de lente que la unidad no tenía (20/11).
    ("video-elp-full", "ELP de VisionX"),
    # Geometría publicada de cámaras estéreo conocidas (PLUG-AND-PLAY.md, sección 9): se simula
    # el ángulo, la separación y la resolución, no el sensor.
    ("video-realsense-d435", "Geometría de Intel RealSense D435"),
    ("video-realsense-d455", "Geometría de Intel RealSense D455"),
    ("video-zed-2", "Geometría de Stereolabs ZED 2"),
    ("video-zed-mini", "Geometría de Stereolabs ZED Mini"),
    ("video-oak-d", "Geometría de Luxonis OAK-D"),
)
# Las cámaras genéricas del primer video (60°, 130°, base de 20 cm) quedan para evaluar
# (`gtsynth video --solo video-angosta-60` sigue funcionando con estas mismas variantes).
CAMARAS_GENERICAS: tuple[tuple[str, str], ...] = (
    ("video-angosta-60", "Angular normal"),
    ("video-gran-angular-130", "Gran angular"),
    ("video-base-larga-20", "Ojos muy separados"),
)
VERSIONES = ("ideal", "sin corregir", "clásico", "red")
FPS = 10
HERMANA = 1.0  # cuánto se aleja «la otra unidad» (rangos de CameraRanges × HERMANA)


def _camara_nominal(g):
    """Modelo pinhole sin distorsión de la cámara simulada (para el generador de hermanas)."""
    from visionx.depth.crestereo_adapt.perturb import CameraModel

    k = np.array([[g.fx, 0, g.cx], [0, g.fy, g.cy], [0, 0, 1]], dtype=np.float64)
    return CameraModel(k, np.zeros(5), np.eye(3))


def _hermana(semilla: int, nominal, f: float = HERMANA):
    from visionx.depth.crestereo_adapt.perturb import CameraRanges, sample_sibling

    base = CameraRanges()
    rangos = dataclasses.replace(
        base, **{k: getattr(base, k) * f for k in ("roll_deg", "pitch_deg", "yaw_deg", "focal",
                                                    "center_px", "distortion")},
    )  # fmt: skip
    return sample_sibling(np.random.default_rng(semilla), nominal, rangos)


def _z(depth, g, w8: int, h8: int, delta_px: float = 0.0) -> np.ndarray:
    """Z (m) en la resolución del recorte, desde la disparidad de la red (menos un corrimiento)."""
    disp = depth.disparity_px.astype(np.float32) - np.float32(delta_px) + depth.doffs
    z = np.where(depth.valid_mask & (disp > 0), g.fx * g.baseline_m / np.maximum(disp, 1e-6),
                 np.nan).astype(np.float32)  # fmt: skip
    if z.shape != (h8, w8):
        z = cv2.resize(z, (w8, h8), interpolation=cv2.INTER_NEAREST)
    return z


def _limite_delta(g) -> float:
    """Corrimiento máximo creíble: el que produce ~1,2° de yaw con esta focal, y al menos 2 px.

    El yaw corre la imagen fx·tan(yaw) píxeles, así que el tope crece con la focal. Antes era
    2 px × fx / 200 y quedaba corto: la D455 en hotel_0 pedía −3,7 px en todos los cuadros (tope
    3,4) y el clásico quedaba en 30 % de error (BITACORA 20/11).
    """
    return max(2.0, g.fx * math.tan(math.radians(1.2)))


def _delta_de_cuadros(ds: list[float | None], g) -> float:
    """El corrimiento de un grupo de cuadros: la mediana, si es creíble y los cuadros coinciden.

    Se controla el conjunto y no cada cuadro: un valor grande pero parejo en todos los cuadros es
    un yaw real; uno que salta de cuadro a cuadro es un objeto que no sirve (se usa 0).
    """
    ds = [d for d in ds if d is not None]
    if not ds:
        return 0.0
    med = float(np.median(ds))
    disperso = len(ds) >= 3 and float(np.percentile(ds, 90) - np.percentile(ds, 10)) > 1.5
    return med if abs(med) <= _limite_delta(g) and not disperso else 0.0


def _delta_conocida(depth, g, z_gt: np.ndarray, mascara: np.ndarray) -> float | None:
    """Corrimiento de disparidad que hace coincidir el objeto apuntado con su distancia real.

    Mediana, sobre los píxeles del objeto, de la diferencia entre la disparidad medida y la que
    corresponde a su Z real. (El percentil 80 de la disparidad contra el 20 de Z, como el
    ubicador, daba corrimientos de ~1 px de más: los extremos de las dos distribuciones no son
    los mismos píxeles.) Sin tope: lo controla `_delta_de_cuadros` sobre todos los cuadros.
    """
    h, w = depth.disparity_px.shape
    m = cv2.resize(mascara.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    zg = cv2.resize(np.nan_to_num(z_gt), (w, h), interpolation=cv2.INTER_NEAREST)
    ok = m & depth.valid_mask & (zg > 0)
    if ok.sum() < 30:
        return None
    d = depth.disparity_px[ok] + depth.doffs
    d_gt = g.fx * g.baseline_m / np.maximum(zg[ok], 1e-6)
    return float(np.median(d - d_gt))


CALENTAR = 3  # cuadros del modo continuo que se descartan para la distancia conocida


def _error(z_est: np.ndarray, z_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    ok_gt = np.isfinite(z_gt) & (z_gt >= Z_MIN) & (z_gt <= Z_MAX)
    ok = ok_gt & np.isfinite(z_est)
    rel = np.where(ok, (z_est - z_gt) / np.where(ok_gt, z_gt, 1), 0.0)
    med = float(100 * np.median(np.abs(rel[ok]))) if ok.sum() > 100 else None
    return rel, ok, med


def preparar_camara(vs: list[Vista], nombre: str, red: str | None, cfg: dict) -> dict[str, Any]:
    """Profundidad de las cuatro versiones para los cuadros de una cámara en una escena."""
    from visionx.depth.crestereo_adapt.align_apply import Alineador
    from visionx.depth.crestereo_adapt.perturb import perturb_right_camera
    from visionx.gtsynth.evaluar import Medidor, _frame

    v0 = vs[0]
    g_lienzo = v0.geometria
    g = v0.geometria_nominal
    w8, h8 = g.width // 8 * 8, g.height // 8 * 8
    g = dataclasses.replace(g, width=w8, height=h8)
    nominal = _camara_nominal(g_lienzo)
    real = _hermana(zlib.crc32(f"{v0.escena}/{v0.variante}".encode()), nominal)

    def recorte(img):
        return v0.recortar(img)[:h8, :w8]

    izqs = [recorte(v.izq()) for v in vs]
    ders = [recorte(v.der()) for v in vs]
    ders_her = [recorte(perturb_right_camera(v.der(), real, nominal, g_lienzo).right) for v in vs]
    zs = [recorte(v.z()) for v in vs]
    inst0 = recorte(v0.instancias())
    est = Medidor.armar("crestereo", cfg, g).estimador
    dg = getattr(est, "geometry", g)

    # El clásico junta puntos de unos pocos cuadros separados, como la app (8 pares): con los 90
    # cuadros seguidos, casi la misma vista repetida, la regla dejaba pasar parámetros de más.
    cl = Alineador("clasico", g, np.zeros(5))
    paso = max(1, len(vs) // 8)
    muestra = list(zip(izqs[::paso], ders_her[::paso], strict=True))
    mapas, aplica = cl._clasico(muestra)
    ders_cl = ders_her if mapas is None else [
        cv2.remap(d, mapas[0], mapas[1], cv2.INTER_LINEAR) for d in ders_her]  # fmt: skip
    info_cl = type("Info", (), {"aplica": aplica})
    ders_red = None
    if red:
        al_red = Alineador("clasico+red", g, np.zeros(5), red=red)
        ders_red, _ = al_red.corregir(list(zip(izqs, ders_her, strict=True)))

    def correr(derechas):
        return [est.predict(_frame(v, i, d, g, k)) for k, (v, i, d) in
                enumerate(zip(vs, izqs, derechas, strict=True))]  # fmt: skip

    t0 = time.perf_counter()
    prof = {"ideal": correr(ders), "sin corregir": correr(ders_her), "clásico": correr(ders_cl)}
    if ders_red is not None:
        prof["red"] = correr(ders_red)
    ms = (time.perf_counter() - t0) * 1000 / (len(vs) * len(prof))
    # Distancia conocida (decisión (a)): el objeto apuntado fija el corrimiento constante de
    # columnas. Para el clásico y la red; el ideal y sin corregir van tal cual. Se usa la mediana
    # sobre los cuadros donde se ve el objeto, salteando los primeros: el modo continuo arranca en
    # frío y su primer cuadro sale corrido (con un solo cuadro el corrimiento variaba de −0,1 a
    # +1,6 px según la muestra y el error del clásico pasaba de ~1,5 % a 15–30 %).
    apuntado = int(v0.meta["apuntado"]["id"])
    insts = [inst0] + [recorte(v.instancias()) for v in vs[1:]]
    deltas = {k: 0.0 for k in prof}
    for k in ("clásico", "red"):
        if k not in prof:
            continue
        cuadros = range(min(CALENTAR, len(vs) - 1), len(vs))
        ds = [_delta_conocida(prof[k][i], dg, zs[i], insts[i] == apuntado) for i in cuadros]
        deltas[k] = _delta_de_cuadros(ds, dg)
    out: dict[str, Any] = {
        "nombre": nombre, "izq": izqs, "z": zs, "g": g,
        "hfov_deg": math.degrees(2 * math.atan(g.width / (2 * g.fx))),
        "base_cm": 100 * g.baseline_m, "ms_por_cuadro": ms,
        "clasico_aplica": list(info_cl.aplica), "delta_px": deltas, "versiones": {},
    }  # fmt: skip
    for k, ds in prof.items():
        out["versiones"][k] = [_z(d, dg, w8, h8, deltas[k]) for d in ds]
    return out


# --- lámina de cada cuadro -------------------------------------------------------------------


def lamina(c: dict[str, Any], i: int, escena: str) -> np.ndarray:
    from visionx.app.stand import (
        ACCENT,
        BG,
        FAINT,
        INK,
        LEVEL_COLOR,
        MUTED,
        H,
        Texts,
        W,
        _panel,
        _place,
        text_width,
        wrap,
    )

    canvas = np.full((H, W, 3), BG, np.uint8)
    texts = Texts()
    texts.add("VisionX", (32, 18), 30, INK, True)
    titulo = "¿Funciona con otra cámara?"
    texts.add(titulo, (32 + text_width("VisionX", 30, True) + 16, 26), 20, ACCENT, True)
    sub = (f"{c['nombre']}  ·  campo {c['hfov_deg']:.0f}°  ·  ojos a {c['base_cm']:.0f} cm  ·  "
           f"cuadro {i + 1}/{len(c['izq'])}")  # fmt: skip
    texts.add(sub, (W - 32 - text_width(sub, 17), 30), 17, MUTED)

    z_gt = c["z"][i]
    ok_gt = np.isfinite(z_gt) & (z_gt >= Z_MIN) & (z_gt <= Z_MAX)
    lo, hi = (float(x) for x in np.percentile(z_gt[ok_gt], (2, 98))) if ok_gt.any() else (1, 5)
    lo, hi = max(lo, Z_MIN), max(hi, max(lo, Z_MIN) * 1.5)

    # Arriba: lo que ve y la distancia real.
    y0, ph = 72, 392
    pw = 700
    _panel(canvas, 24, y0, pw, ph, "Lo que ve el ojo izquierdo", texts)
    _place(canvas, c["izq"][i], 36, y0 + 36, pw - 24, ph - 48)
    _panel(canvas, 24 + pw + 14, y0, pw, ph, "Distancia real", texts, "render: la verdad")
    _place(canvas, _color_z(np.nan_to_num(z_gt), ok_gt, lo, hi), 36 + pw + 14, y0 + 36,
           pw - 24, ph - 48)  # fmt: skip
    ix = 24 + 2 * (pw + 14)
    _panel(canvas, ix, y0, W - ix - 24, ph, "Qué se compara", texts)
    lineas = [
        ("Ideal", "el par perfecto: el techo"),
        ("Sin corregir", "otra unidad del mismo modelo: ojo derecho un poco girado, otra lente"),
        ("Clásico", "lo que usa hoy VisionX (alineación + distancia conocida)"),
        ("Red de entrada", "corrige encima del clásico (en entrenamiento)"),
    ]
    yy = y0 + 46
    for a, b in lineas:
        texts.add(a, (ix + 18, yy), 17, INK, True)
        for linea in wrap(b, 14, W - ix - 24 - 36):
            yy += 22
            texts.add(linea, (ix + 18, yy), 14, MUTED)
        yy += 36
    texts.add("Error: verde < 3 %  ·  ámbar < 10 %  ·  coral más", (ix + 18, y0 + ph - 40), 13,
              FAINT)  # fmt: skip

    # Abajo: por versión, la distancia que calcula y su error.
    vw, vh = (W - 48 - 3 * 14) // 4, 280
    for j, nombre in enumerate(VERSIONES):
        x = 24 + j * (vw + 14)
        y = y0 + ph + 14
        zs = c["versiones"].get(nombre)
        if zs is None:
            _panel(canvas, x, y, vw, 2 * vh + 14, "Red de entrada (v3)", texts)
            pendiente = ("Pendiente: la red se está", "por reentrenar. Cuando esté,",
                         "este lugar muestra su", "distancia y su error.")  # fmt: skip
            for k, linea in enumerate(pendiente):
                texts.add(linea, (x + 24, y + 120 + k * 26), 16, FAINT)
            continue
        z = zs[i]
        rel, ok, med = _error(z, z_gt)
        nota = "—" if med is None else f"error {med:.1f} %".replace(".", ",")
        color = LEVEL_COLOR["sin-medida"] if med is None else LEVEL_COLOR[
            "precisa" if med < 3 else "aproximada" if med < 10 else "no-segura"]  # fmt: skip
        _panel(canvas, x, y, vw, vh, nombre.capitalize(), texts)
        texts.add(nota, (x + vw - 18 - text_width(nota, 18, True), y + 10), 18, color, True)
        _place(canvas, _color_z(np.nan_to_num(z), np.isfinite(z), lo, hi), x + 10, y + 36,
               vw - 20, vh - 46)  # fmt: skip
        _panel(canvas, x, y + vh + 14, vw, vh, f"Error · {nombre}", texts)
        _place(canvas, _color_error(rel, ok), x + 10, y + vh + 50, vw - 20, vh - 46)
    pie = (f"Escena {escena} · CREStereo continuo (el modo de la app) · render con la verdad "
           "exacta: la cámara real agrega ruido y desenfoque.")  # fmt: skip
    texts.add(pie, (24, H - 30), 14, FAINT)
    return texts.draw(canvas)


def preparar(
    raiz: str | Path,
    salida: str | Path,
    red: str | None = None,
    cfg: dict | None = None,
    solo: tuple[str, ...] | None = None,
) -> list[dict]:
    """Todos los videos, o solo los de las variantes `solo` (se suman al resumen que había)."""
    from visionx.shared.config import load_config

    cfg = cfg or load_config()
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    previo = salida / "resumen.json"
    resumen = json.loads(previo.read_text(encoding="utf-8")) if solo and previo.is_file() else []
    resumen = [f for f in resumen if f["variante"] not in (solo or ())]
    todas = CAMARAS_VIDEO + CAMARAS_GENERICAS
    escenas = sorted({v.escena for var, _ in todas for v in vistas(raiz, var)})
    for escena in escenas:
        for var, nombre in CAMARAS_VIDEO + CAMARAS_GENERICAS:
            if solo and var not in solo:
                continue
            vs = sorted((v for v in vistas(raiz, var) if v.escena == escena),
                        key=lambda v: v.meta["indice"])  # fmt: skip
            if not vs:
                continue
            c = preparar_camara(vs, nombre, red, cfg)
            path = salida / f"{escena}-{var}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
            for i in range(len(vs)):
                writer.write(lamina(c, i, escena))
            writer.release()
            fila = {"escena": escena, "variante": var, "nombre": nombre, "video": path.name,
                    "hfov_deg": round(c["hfov_deg"], 1), "base_cm": round(c["base_cm"], 1),
                    "ms_por_cuadro": round(c["ms_por_cuadro"]),
                    "clasico_aplica": c["clasico_aplica"],
                    "delta_px": {k: round(v, 3) for k, v in c["delta_px"].items()}}  # fmt: skip
            for k, zs in c["versiones"].items():
                errs = [e for e in (_error(z, zg)[2] for z, zg in zip(zs, c["z"], strict=True))
                        if e is not None]  # fmt: skip
                fila[f"error_{k}_pct"] = round(float(np.median(errs)), 2) if errs else None
            resumen.append(fila)
            print(json.dumps(fila, ensure_ascii=False), flush=True)
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1),
                                         encoding="utf-8")  # fmt: skip
    return resumen
