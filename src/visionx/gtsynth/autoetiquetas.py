"""Etiquetas de la geometría para entrenar la red de entrada sin calibrar nada (auto-supervisión).

Por cada recorrido de una variante con secuencias (p. ej. `hssd-recorridos-sorteada-b`, que la
red v2 nunca vio), el sistema hace lo mismo que en la pantalla «Cámara nueva»:

1. la red de entrada mira los primeros cuadros y da focal, centro y su duda;
2. el movimiento mide la focal partiendo de ahí (`app.autocalibracion`);
3. las dos se combinan.

La focal combinada es la **etiqueta** de todos los cuadros del recorrido, con su duda: sirve
para reentrenar la red con video que nadie calibró. La cámara real (que en la simulación se
conoce) **no se usa para etiquetar**: solo se guarda al lado para medir qué tan buenas son las
etiquetas.

    python -m visionx.gtsynth autoetiquetas data/gt-sintetico-elp \\
        --variante hssd-recorridos-sorteada-b --salida outputs/autoetiquetas/b.csv
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

CAMPOS = ("variante", "escena", "recorrido", "cuadros", "ancho", "fx_real", "fx_red", "sigma_red",
          "fx_mov", "sigma_mov", "pares", "motivo", "fx_comb", "sigma_comb",
          "carpetas")  # fmt: skip


def recorridos(raiz: str | Path, variante: str, particion: str | None = "train") -> dict:
    from visionx.gtsynth.dataset import vistas

    por: dict[tuple, list] = {}
    for v in vistas(raiz, variante, particion):
        por.setdefault((v.escena, v.meta["secuencia"]["recorrido"]), []).append(v)
    return {k: sorted(vs, key=lambda v: v.meta["secuencia"]["cuadro"]) for k, vs in por.items()}


def etiquetar(vs, estimador, saltos=(2, 4, 8)) -> dict:
    """Etiqueta de un recorrido: red + movimiento, sin mirar la cámara real."""
    from visionx.app.autocalibracion import combinar, focal_por_movimiento
    from visionx.gtsynth.dataset import VistaNominal

    vn = [VistaNominal(v.carpeta) for v in vs]
    g = vn[0].geometria  # solo ancho/alto para el sistema; fx_real queda para medir
    izq = [v.izq() for v in vn]
    est = estimador.estimar([(izq[i], vn[i].der()) for i in range(min(8, len(vn)))])
    f_red = (g.width / 2) / math.tan(math.radians(est.campo_horizontal_deg) / 2)
    cx = (0.5 + est.centro_rel[0]) * g.width - 0.5
    cy = (0.5 + est.centro_rel[1]) * g.height - 0.5
    m = focal_por_movimiento(izq, cx, cy, f_red, saltos=saltos)
    lf, s = combinar(math.log(f_red), est.sigma_log_f, m)
    return {
        "variante": vs[0].variante, "escena": vs[0].escena,
        "recorrido": vs[0].meta["secuencia"]["recorrido"], "cuadros": len(vs), "ancho": g.width,
        "fx_real": round(g.fx, 3), "fx_red": round(f_red, 3),
        "sigma_red": round(est.sigma_log_f, 5), "fx_mov": m.fx and round(m.fx, 3),
        "sigma_mov": m.sigma_log and round(m.sigma_log, 5), "pares": m.pares,
        "motivo": m.motivo or "", "fx_comb": round(math.exp(lf), 3), "sigma_comb": round(s, 5),
        "carpetas": ";".join(str(v.carpeta) for v in vs),
    }  # fmt: skip


def calidad(filas: list[dict], sigma_max: float = 0.03) -> dict:
    """Qué tan buenas son las etiquetas frente a la cámara real (solo para el informe)."""
    out = {"recorridos": len(filas)}
    for k in ("red", "comb"):
        e = np.array([abs(float(f[f"fx_{k}"]) / float(f["fx_real"]) - 1) * 100 for f in filas])
        out[k] = {"mediana_pct": round(float(np.median(e)), 2),
                  "p90_pct": round(float(np.percentile(e, 90)), 2)}  # fmt: skip
    ok = [f for f in filas if float(f["sigma_comb"]) <= sigma_max]
    if ok:
        e = np.array([abs(float(f["fx_comb"]) / float(f["fx_real"]) - 1) * 100 for f in ok])
        out["aceptadas"] = {"n": len(ok), "sigma_max": sigma_max,
                            "mediana_pct": round(float(np.median(e)), 2),
                            "p90_pct": round(float(np.percentile(e, 90)), 2)}  # fmt: skip
    return out


VISTAS = ("vista", "recorrido", "ancho", "log_f", "sigma_log_f", "fx_etiqueta", "fx_real")


def exportar(csv_recorridos, raiz, salida, sigma_max: float = 0.03) -> int:
    """Una fila por cuadro, con la ruta relativa a `raiz`, para entrenar en otra máquina.

    `log_f` = log(fx / ancho) de la imagen nominal, la misma etiqueta que `camara_net`. Solo
    los recorridos con duda ≤ `sigma_max`. `fx_real` va para el control (no para entrenar).
    """
    raiz = Path(raiz)
    with open(csv_recorridos, encoding="utf-8", newline="") as fh:
        filas = [f for f in csv.DictReader(fh) if float(f["sigma_comb"]) <= sigma_max]
    n = 0
    with open(salida, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=VISTAS)
        w.writeheader()
        for f in filas:
            ancho, fx = int(f["ancho"]), float(f["fx_comb"])
            clave = f"{f['variante']}/{f['escena']}/r{f['recorrido']}"
            for c in f["carpetas"].split(";"):
                w.writerow({"vista": Path(c).relative_to(raiz).as_posix(), "recorrido": clave,
                            "ancho": ancho, "log_f": round(math.log(fx / ancho), 6),
                            "sigma_log_f": f["sigma_comb"], "fx_etiqueta": f["fx_comb"],
                            "fx_real": f["fx_real"]})  # fmt: skip
                n += 1
    return n


def correr(raiz, variante: str, red: str, salida, particion: str | None = "train",
           limite: int = 0) -> dict:  # fmt: skip
    from visionx.app.estimador_red import EstimadorRed

    est = EstimadorRed(red)
    recs = recorridos(raiz, variante, particion)
    claves = sorted(recs)[: limite or None]
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    hechas = set()
    filas: list[dict] = []
    if salida.exists():  # retomar
        with open(salida, encoding="utf-8", newline="") as fh:
            filas = list(csv.DictReader(fh))
        hechas = {(f["escena"], int(f["recorrido"])) for f in filas}
    nuevo = not salida.exists()
    with open(salida, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        t0 = time.time()
        for k, clave in enumerate(claves):
            if clave in hechas:
                continue
            fila = etiquetar(recs[clave], est)
            w.writerow(fila)
            fh.flush()
            filas.append(fila)
            if k % 10 == 0:
                print(f"{k + 1}/{len(claves)} ({time.time() - t0:.0f} s)", flush=True)
    q = calidad(filas)
    q["vistas_exportadas"] = exportar(salida, raiz, salida.with_name(salida.stem + "-vistas.csv"))
    salida.with_suffix(".json").write_text(json.dumps(q, indent=1, ensure_ascii=False), "utf-8")
    return q
