"""Entrenar y evaluar la red que estima la cámara (E2 de PLUG-AND-PLAY).

Pérdida = NLL de Laplace de las cinco etiquetas (en unidades de `lente.ESCALA`) + error de
reproyección: los mismos rayos proyectados con la cámara estimada y con la real, en anchos de
imagen. La segunda hace que un error de focal que casi no cambia la imagen no se castigue de más.

Métricas de E2 (revisión #85): error de la **focal** en % (meta: mediana ≤ 2 %), del centro en
px de la imagen nominal, de `k1`, de reproyección en px, y la cobertura del intervalo del 90 %
de cada b (|error| ≤ b·ln 10). Siempre al lado de la referencia «constante» (predecir la media
de entrenamiento), para ver cuánto aprendió la red.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from visionx.depth.camara_net.datos import (
    Aumento,
    Conjunto,
    Item,
    LotesPorAlto,
    item_de,
    juntar,
    lotes_por_alto,
    submuestra_recorridos,
)
from visionx.depth.camara_net.lente import ESCALA, NOMBRES, distorsionar

LN10 = math.log(10.0)  # Laplace: P(|e| ≤ b·ln 10) = 90 %
ESCALA_REPROY = 0.005  # 0,5 % del ancho ≈ 3 px a 640


@dataclass
class Params:
    raiz: str = "data/gt-sintetico-elp"
    variantes: tuple[str, ...] = ("camara-sorteada", "hssd-camara-sorteada")
    recorridos: tuple[str, ...] = ("hssd-recorridos-sorteada",)
    cuadros_por_recorrido: int = 6
    epocas: int = 20
    lote: int = 12
    lr: float = 3e-4
    lr_tronco: float = 1e-4
    peso_reproy: float = 0.5
    calentamiento: int = 300  # pasos
    trabajadores: int = 8
    semilla: int = 0
    max_items: int = 0  # 0 = todos (para pruebas rápidas)


# ---------------------------------------------------------------- pérdida


def proyectar(mu, rayos, aspecto):
    """Píxeles (en anchos de imagen) donde caen `rayos` (B, N, 2) con la cámara `mu` (B, 5)."""
    import torch

    f = torch.exp(mu[:, 0:1])
    xd, yd = distorsionar(rayos[..., 0], rayos[..., 1], mu[:, 3:4], mu[:, 4:5])
    u = f * xd + (mu[:, 1:2] + 0.5)
    v = f * yd + (mu[:, 2:3] + 0.5) * aspecto[:, None]
    return torch.stack([u, v], -1)


def perdida(out: dict, lote: dict, peso_reproy: float) -> tuple[Any, dict[str, float]]:
    import torch

    escala = torch.as_tensor(ESCALA, device=out["mu"].device)
    r = (lote["y"] - out["mu"]) / escala
    nll = (r.abs() * torch.exp(-out["log_b"]) + out["log_b"]).mean()
    # `pix` ya viene en anchos de imagen (u y v divididos por el ancho).
    reproy = (
        (proyectar(out["mu"], lote["rayos"], lote["aspecto"]) - lote["pix"]).norm(dim=-1).mean()
    )
    total = nll + peso_reproy * reproy / ESCALA_REPROY
    return total, {"nll": nll.item(), "reproy_pct": reproy.item() * 100}


# ---------------------------------------------------------------- datos


def items_de(raiz: str | Path, variantes: Sequence[str], particion: str | None,
             ) -> list[Item]:  # fmt: skip
    from visionx.gtsynth.dataset import vistas

    out: list[Item] = []
    for v in variantes:
        out += [item_de(x) for x in vistas(raiz, v, particion)]
    return out


def items_entrenamiento(p: Params) -> list[Item]:
    items = items_de(p.raiz, p.variantes, "train")
    rec = items_de(p.raiz, p.recorridos, "train")
    items += submuestra_recorridos(rec, p.cuadros_por_recorrido, p.semilla)
    return items


def _cargador(conj: Conjunto, lotes, trabajadores: int, persistente: bool = False):
    from torch.utils.data import DataLoader

    return DataLoader(conj, batch_sampler=lotes, num_workers=trabajadores, collate_fn=juntar,
                      persistent_workers=persistente and trabajadores > 0,
                      prefetch_factor=4 if trabajadores else None)  # fmt: skip


# ---------------------------------------------------------------- inferencia y métricas


def predecir(modelo, conj: Conjunto, device: str, lote: int = 16, trabajadores: int = 4,
             ) -> dict[str, np.ndarray]:  # fmt: skip
    """mu, b (unidades de etiqueta) y y de cada ítem de `conj`, en su orden."""
    import torch

    from visionx.depth.camara_net.modelo import b_en_unidades

    n = len(conj)
    mu, b, y = np.zeros((n, 5), np.float32), np.zeros((n, 5), np.float32), np.zeros((n, 5))
    aspecto = np.zeros(n, np.float32)
    modelo.eval()
    with torch.no_grad():
        for bt in _cargador(conj, lotes_por_alto(conj.items, lote, None), trabajadores):
            out = modelo(bt["ojos"].to(device), bt["aspecto"].to(device))
            i = bt["indice"].numpy()
            mu[i] = out["mu"].cpu().numpy()
            b[i] = b_en_unidades(out["log_b"], modelo.escala).cpu().numpy()
            y[i] = bt["y"].numpy()
            aspecto[i] = bt["aspecto"].numpy()
    return {"mu": mu, "b": b, "y": y.astype(np.float32), "aspecto": aspecto}


def errores(mu: np.ndarray, y: np.ndarray, nominal: np.ndarray) -> dict[str, np.ndarray]:
    """Errores por ítem. `nominal` (N, 2) = ancho y alto de la imagen nominal en px."""
    from visionx.depth.camara_net.lente import Camara, grilla_rayos

    f_pct = np.abs(np.exp(mu[:, 0] - y[:, 0]) - 1) * 100
    hfov = lambda a: np.degrees(2 * np.arctan(0.5 / np.exp(a)))  # noqa: E731
    reproy = np.zeros(len(mu), np.float32)
    for k in range(len(mu)):
        w, h = int(nominal[k, 0]), int(nominal[k, 1])
        real = Camara.de_etiqueta(y[k], w, h)
        est = Camara.de_etiqueta(mu[k], w, h)
        rayos, pix = grilla_rayos(real)
        xd, yd = distorsionar(rayos[:, 0], rayos[:, 1], est.k1, est.k2)
        p = np.stack([est.fx * xd + est.cx, est.fx * yd + est.cy], -1)
        reproy[k] = float(np.linalg.norm(p - pix, axis=1).mean())
    return {
        "focal_pct": f_pct,
        "hfov_deg": np.abs(hfov(mu[:, 0]) - hfov(y[:, 0])),
        "cx_px": np.abs(mu[:, 1] - y[:, 1]) * nominal[:, 0],
        "cy_px": np.abs(mu[:, 2] - y[:, 2]) * nominal[:, 1],
        "k1": np.abs(mu[:, 3] - y[:, 3]),
        "k2": np.abs(mu[:, 4] - y[:, 4]),
        "reproy_px": reproy,
    }


def resumen(e: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for k, v in e.items():
        out[f"{k}_med"] = round(float(np.median(v)), 4)
        out[f"{k}_p90"] = round(float(np.percentile(v, 90)), 4)
    return out


def cobertura(mu: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict[str, float]:
    dentro = np.abs(y - mu) <= b * LN10
    return {f"cob90_{n}": round(float(dentro[:, j].mean()), 3) for j, n in enumerate(NOMBRES)}


def combinar(mu: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Varios cuadros de la misma cámara → una estimación: media pesada por 1/b² y su b."""
    w = 1.0 / np.maximum(b, 1e-6) ** 2
    m = (w * mu).sum(0) / w.sum(0)
    return m.astype(np.float32), (1.0 / np.sqrt(w.sum(0))).astype(np.float32)


def evaluar_conjunto(modelo, items: list[Item], aum: Aumento, device: str, centro: np.ndarray,
                     trabajadores: int = 4, csv_path: str | Path | None = None,
                     ) -> dict[str, Any]:  # fmt: skip
    """Métricas de un conjunto (semilla fija por vista) y la referencia constante."""
    conj = Conjunto(items, aum, semilla_fija=True)
    p = predecir(modelo, conj, device, trabajadores=trabajadores)
    nominal = np.array([it.nominal for it in items], np.float32)
    e = errores(p["mu"], p["y"], nominal)
    const = errores(np.repeat(centro[None].astype(np.float32), len(items), 0), p["y"], nominal)
    out: dict[str, Any] = {"n": len(items), **resumen(e), **cobertura(p["mu"], p["b"], p["y"])}
    out["constante"] = resumen(const)
    # Por cámara (recorridos): todos los cuadros juntos y de a 8.
    grupos: dict[str, list[int]] = {}
    for k, it in enumerate(items):
        grupos.setdefault(it.grupo, []).append(k)
    if any(len(v) > 1 for v in grupos.values()):
        por = {}
        for g, idx in sorted(grupos.items()):
            idx = np.array(idx)
            m_all, _ = combinar(p["mu"][idx], p["b"][idx])
            e_all = errores(m_all[None], p["y"][idx[:1]], nominal[idx[:1]])
            ocho = [combinar(p["mu"][idx[j : j + 8]], p["b"][idx[j : j + 8]])[0]
                    for j in range(0, len(idx) - 7, 8)]  # fmt: skip
            e8 = errores(np.array(ocho), np.repeat(p["y"][idx[:1]], len(ocho), 0),
                         np.repeat(nominal[idx[:1]], len(ocho), 0)) if ocho else None  # fmt: skip
            hfov_real = float(np.degrees(2 * np.arctan(0.5 / np.exp(p["y"][idx[0], 0]))))
            de_a_8 = None if e8 is None else round(float(np.median(e8["focal_pct"])), 3)
            por[g] = {
                "cuadros": len(idx),
                "hfov_real": round(hfov_real, 1),
                "un_cuadro": resumen(errores(p["mu"][idx], p["y"][idx], nominal[idx])),
                "de_a_8_focal_pct_med": de_a_8,
                "todos_focal_pct": round(float(e_all["focal_pct"][0]), 3),
                "todos_cx_px": round(float(e_all["cx_px"][0]), 2),
                "todos_k1": round(float(e_all["k1"][0]), 4),
            }  # fmt: skip
        out["por_camara"] = por
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["carpeta", "grupo", "ancho", "alto"] + [f"y_{n}" for n in NOMBRES]
                        + [f"mu_{n}" for n in NOMBRES] + [f"b_{n}" for n in NOMBRES]
                        + list(e))  # fmt: skip
            for k, it in enumerate(items):
                errs = [round(float(v[k]), 4) for v in e.values()]
                wr.writerow([it.carpeta, it.grupo, *it.nominal, *np.round(p["y"][k], 5),
                             *np.round(p["mu"][k], 5), *np.round(p["b"][k], 5), *errs])  # fmt: skip
    return out


# ---------------------------------------------------------------- entrenamiento


def _lr(paso: int, total: int, calentamiento: int) -> float:
    if paso < calentamiento:
        return (paso + 1) / calentamiento
    t = (paso - calentamiento) / max(1, total - calentamiento)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t))


def entrenar(p: Params, salida: str | Path, device: str = "auto", continuar: bool = False,
             ) -> dict[str, Any]:  # fmt: skip
    import torch

    from visionx.depth.camara_net.modelo import RedCamara
    from visionx.depth.mono import pick_device

    device = pick_device(device)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(p.semilla)
    rng = np.random.default_rng(p.semilla)

    tren = items_entrenamiento(p)
    val = items_de(p.raiz, p.variantes, "val")
    if p.max_items:
        tren = [tren[i] for i in rng.choice(len(tren), min(p.max_items, len(tren)), replace=False)]
        val = val[: max(8, p.max_items // 4)]
    print(f"entrenamiento: {len(tren)} vistas; val: {len(val)}; device {device}", flush=True)

    # Media de las etiquetas con el aumento (la red predice desvíos respecto de esto, y es la
    # referencia «constante» de las métricas). Se estima sorteando cámaras, sin abrir imágenes.
    from visionx.depth.camara_net.datos import sortear_camara

    muestra_idx = rng.choice(len(tren), min(2000, len(tren)), replace=False)
    centro = np.mean([sortear_camara(tren[i], Aumento(), rng).etiqueta() for i in muestra_idx], 0)
    modelo = RedCamara(preentrenada=not continuar).to(device)
    modelo.fijar_centro(torch.tensor(centro, dtype=torch.float32, device=device))
    tronco = list(modelo.tronco.parameters())
    resto = [q for n, q in modelo.named_parameters() if not n.startswith("tronco.")]
    opt = torch.optim.AdamW([{"params": tronco, "lr": p.lr_tronco},
                             {"params": resto, "lr": p.lr}], weight_decay=1e-4)  # fmt: skip
    base_lr = [g["lr"] for g in opt.param_groups]
    epoca0, mejor = 0, math.inf
    if continuar and (salida / "ultimo.pt").exists():
        ck = torch.load(salida / "ultimo.pt", map_location=device, weights_only=False)
        modelo.load_state_dict(ck["modelo"])
        opt.load_state_dict(ck["opt"])
        epoca0, mejor = ck["epoca"] + 1, ck["mejor"]
        print(f"sigue desde la época {epoca0}", flush=True)

    meta = {"params": asdict(p), "device": device, "n_train": len(tren), "n_val": len(val),
            "centro": centro.tolist(), "escala": ESCALA.tolist()}  # fmt: skip
    (salida / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    curvas = salida / "curvas.csv"
    if not continuar or not curvas.exists():
        curvas.write_text("epoca,perdida,nll,reproy_pct,val_focal_med,val_focal_p90,"
                          "val_cx_med,val_k1_med,val_reproy_med,val_cob90_f,seg\n")  # fmt: skip

    conj = Conjunto(tren, Aumento())
    lotes = LotesPorAlto(tren, p.lote, p.semilla)
    lotes.epoca = epoca0
    cargador = _cargador(conj, lotes, p.trabajadores, persistente=True)
    pasos_epoca = len(lotes)
    total = p.epocas * pasos_epoca
    for ep in range(epoca0, p.epocas):
        t0 = time.time()
        modelo.train()
        acum, n = np.zeros(3), 0
        for j, bt in enumerate(cargador):
            paso = ep * pasos_epoca + j
            for g, b0 in zip(opt.param_groups, base_lr, strict=True):
                g["lr"] = b0 * _lr(paso, total, p.calentamiento)
            bt = {k: v.to(device) for k, v in bt.items()}
            out = modelo(bt["ojos"], bt["aspecto"])
            loss, partes = perdida(out, bt, p.peso_reproy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            acum += [loss.item(), partes["nll"], partes["reproy_pct"]]
            n += 1
            if j % 50 == 0:
                print(f"  ép {ep} paso {j}/{pasos_epoca}: pérdida {float(loss):.3f} "
                      f"(reproy {partes['reproy_pct']:.2f} %)", flush=True)  # fmt: skip
        ev = evaluar_conjunto(modelo, val, Aumento(), device, centro, p.trabajadores // 2)
        seg = time.time() - t0
        m = acum / max(n, 1)
        fila = [ep, *np.round(m, 4), ev["focal_pct_med"], ev["focal_pct_p90"], ev["cx_px_med"],
                ev["k1_med"], ev["reproy_px_med"], ev["cob90_log_f"], round(seg)]  # fmt: skip
        with open(curvas, "a", encoding="utf-8") as fh:
            fh.write(",".join(str(x) for x in fila) + "\n")
        print(
            f"época {ep}: pérdida {m[0]:.3f}; val focal {ev['focal_pct_med']:.2f} % "
            f"(p90 {ev['focal_pct_p90']:.2f}), centro {ev['cx_px_med']:.1f} px, "
            f"k1 {ev['k1_med']:.3f}, cob90 f {ev['cob90_log_f']:.2f}; {seg:.0f} s",
            flush=True,
        )
        ck = {"modelo": modelo.state_dict(), "centro": centro.tolist(), "params": asdict(p),
              "epoca": ep, "val": ev}  # fmt: skip
        if ev["focal_pct_med"] < mejor:
            mejor = ev["focal_pct_med"]
            torch.save(ck, salida / "mejor.pt")
        torch.save({**ck, "opt": opt.state_dict(), "mejor": mejor}, salida / "ultimo.pt")
    return {"mejor_focal_pct_med": mejor}


def cargar(checkpoint: str | Path, device: str = "auto"):
    import torch

    from visionx.depth.camara_net.modelo import RedCamara
    from visionx.depth.mono import pick_device

    device = pick_device(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    modelo = RedCamara(preentrenada=False).to(device)
    modelo.load_state_dict(ck["modelo"])
    modelo.eval()
    return modelo, np.array(ck["centro"], np.float32), device
