"""Mosaico de la expo: las 6 cámaras de `CAMARAS_VIDEO` a la vez, en una grilla de 3 × 2.

Se arma **recortando los videos de `gtsynth video`** (sin volver a correr CREStereo): de cada
lámina toma el panel «Clásico» (la distancia que mide VisionX con esa cámara, con su error) y
le agrega el nombre, el campo, la separación y, si está `e0.json`, el error «sin calibrar»
(la misma cámara si el sistema creyera que es la ELP).

    python -m visionx.gtsynth mosaico data/stand-video [--e0 outputs/gtsynth/e0.json]

Escribe `<escena>-video-mosaico.mp4` en la misma carpeta; la pestaña «Cámaras» lo muestra
primero.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from visionx.gtsynth.video import CAMARAS_VIDEO, FPS, VERSIONES

# Panel «Clásico» dentro de la lámina de `video.lamina` (1920×1080): fila de abajo, tercera
# columna. Si cambia el dibujo de la lámina, cambia esto (lo cubre un test).
_VW = (1920 - 48 - 3 * 14) // 4
CLASICO = (24 + VERSIONES.index("clásico") * (_VW + 14), 72 + 392 + 14, _VW, 280)


def _e0(path: str | Path | None) -> dict[tuple[str, str], float]:
    if not path or not Path(path).is_file():
        return {}
    filas = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(f["escena"], f["variante"]): f["error_sin calibrar_pct"] for f in filas}


def _resumen(carpeta: Path) -> dict[tuple[str, str], dict]:
    path = carpeta / "resumen.json"
    if not path.is_file():
        return {}
    return {(f["escena"], f["variante"]): f for f in json.loads(path.read_text(encoding="utf-8"))}


def celda(lamina: np.ndarray | None, nombre: str, detalle: str, sin_calibrar: float | None,
          w: int, h: int, es_elp: bool = False) -> np.ndarray:  # fmt: skip
    """Una celda del mosaico: nombre arriba, el panel «Clásico» y el dato de E0 abajo."""
    from visionx.app.stand import BG, FAINT, INK, LEVEL_COLOR, MUTED, Texts, _panel, _place

    img = np.full((h, w, 3), BG, np.uint8)
    t = Texts()
    _panel(img, 6, 6, w - 12, h - 12, "", t)
    t.add(nombre.removeprefix("Geometría de "), (24, 18), 22, INK, True)
    t.add(detalle, (24, 50), 15, MUTED)
    if lamina is None:
        t.add("Video pendiente", (24, h // 2), 18, FAINT)
        return t.draw(img)
    x, y, pw, ph = CLASICO
    _place(img, lamina[y : y + ph, x : x + pw], 18, 80, w - 36, h - 80 - 44)
    if sin_calibrar is not None:
        cual = (
            "Sin corregir la otra unidad" if es_elp else "Sin calibrar (si creyera que es la ELP)"
        )
        texto = f"{cual}: {sin_calibrar:.0f} % de error"
        t.add(texto, (24, h - 40), 15, LEVEL_COLOR["no-segura"])
    return t.draw(img)


def armar(carpeta: str | Path, escena: str, e0: str | Path | None = None) -> Path | None:
    """El MP4 del mosaico de una escena, o None si no hay ningún video de esa escena."""
    from visionx.app.stand import ACCENT, BG, INK, MUTED, H, Texts, W, text_width

    carpeta = Path(carpeta)
    caps = {var: cv2.VideoCapture(str(carpeta / f"{escena}-{var}.mp4"))
            for var, _ in CAMARAS_VIDEO if (carpeta / f"{escena}-{var}.mp4").is_file()}  # fmt: skip
    if not caps:
        return None
    n = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
    errores_e0 = _e0(e0)
    resumen = _resumen(carpeta)
    salida = carpeta / f"{escena}-video-mosaico.mp4"
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    cw, ch = W // 3, (H - 72) // 2
    for i in range(n):
        canvas = np.full((H, W, 3), BG, np.uint8)
        t = Texts()
        t.add("VisionX", (32, 18), 30, INK, True)
        t.add("La misma app con 6 cámaras estéreo distintas",
              (32 + text_width("VisionX", 30, True) + 16, 26), 20, ACCENT, True)  # fmt: skip
        nota = f"Distancia que mide VisionX (clásico + distancia conocida) · cuadro {i + 1}/{n}"
        t.add(nota, (W - 32 - text_width(nota, 16), 30), 16, MUTED)
        canvas = t.draw(canvas)
        for k, (var, nombre) in enumerate(CAMARAS_VIDEO):
            lam = None
            if var in caps:
                ok, lam = caps[var].read()
                lam = lam if ok else None
            r = resumen.get((escena, var), {})
            detalle = ""
            if r:
                base = f"{r['base_cm']:.1f}".replace(".", ",")
                detalle = f"campo {r['hfov_deg']:.0f}° · lentes a {base} cm"
            x, y = (k % 3) * cw, 72 + (k // 3) * ch
            canvas[y : y + ch, x : x + cw] = celda(
                lam, nombre, detalle, errores_e0.get((escena, var)), cw, ch, var == "video-elp-full"
            )
        writer.write(canvas)
    writer.release()
    for c in caps.values():
        c.release()
    return salida


def armar_todos(carpeta: str | Path, e0: str | Path | None = None) -> list[Path]:
    carpeta = Path(carpeta)
    escenas = sorted({p.name.split("-video-")[0] for p in carpeta.glob("*-video-*.mp4")})
    hechos = [armar(carpeta, e, e0) for e in escenas]
    return [p for p in hechos if p is not None]
