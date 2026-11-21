"""Disparidades de CREStereo precalculadas sobre el GT sintético (para entrenar en otra máquina).

La U-Net de incertidumbre y la red de entrada se entrenan en la Mac; CREStereo corre más rápido
en la PC de escritorio (Radeon, ONNX + DirectML). Este comando corre CREStereo **una vez** por
vista y guarda lo que la U-Net necesita, así la Mac solo entrena:

    python -m visionx.gtsynth disparidades data/gt-sintetico-elp \\
        --variantes camara-sorteada hssd-camara-sorteada hssd-recorridos-sorteada \\
        --salida data/cache-crestereo

Por vista, `<salida>/<variante>/<escena>/<indice>.npz` (comprimido, float16):

    disparidad   H × W   disparidad de CREStereo en píxeles del recorte nominal (ojo izquierdo)
    vertical     H × W   la componente vertical que devuelve la red (desalineación de filas)
    mitad        H × W   la disparidad de la primera pasada a media resolución, ya escalada
    meta         JSON    variante, escena, índice, fx, separación, tamaño, modo, pesos

La disparidad real sale de la vista: `d_gt = fx · separación / Z`, con Z del `z.npz` recortado
igual (`Vista.recortar`). Se usa el par **limpio** (sin hermana); las variantes desalineadas se
pueden agregar después con otra semilla. Si el archivo ya existe y coincide, se saltea.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _matcher(cfg: dict):
    from visionx.depth.crestereo_adapt.matcher import CreStereoMatcher, CreStereoParams
    from visionx.depth.crestereo_adapt.port import PortConfig, load_model

    sec = cfg["depth"]["crestereo"]
    port = PortConfig.from_config(sec)
    model, device = load_model(port)
    params = CreStereoParams.from_config(sec["preciso"])
    return CreStereoMatcher(model, device, params), params, port.weights_sha256[:8]


def calcular(raiz: str | Path, variantes: list[str], salida: str | Path,
             cfg: dict | None = None, limite: int = 0, log=print) -> dict[str, int]:  # fmt: skip
    """Corre CREStereo sobre cada vista que falte; devuelve cuántas calculó y cuántas salteó."""
    from visionx.gtsynth.dataset import vistas
    from visionx.shared.config import load_config

    cfg = cfg or load_config()
    matcher, params, pesos = _matcher(cfg)
    salida = Path(salida)
    hechas = salteadas = 0
    t0 = time.perf_counter()
    for var in variantes:
        todas = sorted(vistas(raiz, var), key=lambda v: (v.escena, v.meta["indice"]))
        if limite:
            todas = todas[:limite]
        log(f"{var}: {len(todas)} vistas")
        for n, v in enumerate(todas, start=1):
            path = salida / var / v.escena / f"{int(v.meta['indice']):04d}.npz"
            g = v.geometria_nominal
            meta = {"variante": var, "escena": v.escena, "indice": int(v.meta["indice"]),
                    "fx": float(g.fx), "separacion_m": float(g.baseline_m),
                    "ancho": int(g.width), "alto": int(g.height), "modo": params.tag,
                    "pesos": pesos}  # fmt: skip
            if path.is_file():
                try:
                    with np.load(path, allow_pickle=False) as d:
                        if json.loads(str(d["meta"])) == meta:
                            salteadas += 1
                            continue
                except (OSError, ValueError, KeyError):
                    pass
            izq, der = v.recortar(v.izq()), v.recortar(v.der())
            matcher.reset()  # cada vista es otra escena: sin arranque en caliente
            disp = matcher.compute(izq, der)
            extras = dict(matcher.last_extras)
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {"disparidad": disp.astype(np.float16), "meta": np.array(json.dumps(meta))}
            if extras.get("vertical") is not None:
                arrays["vertical"] = extras["vertical"].astype(np.float16)
            if extras.get("half") is not None:
                arrays["mitad"] = extras["half"].astype(np.float16)
            np.savez_compressed(path, **arrays)
            hechas += 1
            if n % 200 == 0 or n == len(todas):
                minutos = (time.perf_counter() - t0) / 60
                log(f"  {var}: {n}/{len(todas)} ({hechas} calculadas, {minutos:.1f} min)")
    return {"calculadas": hechas, "salteadas": salteadas}
