"""Mosaico de la expo: las 6 cámaras de `CAMARAS_VIDEO` a la vez, en una grilla de 3 × 2.

Se arma **recortando los videos de `gtsynth video`** (sin volver a correr CREStereo): de cada
lámina toma el panel «Clásico» (la distancia que mide VisionX con esa cámara, con su error del
cuadro). Cada celda es una tarjeta con:

- el modelo y sus datos (campo, separación, resolución, de dónde sale la calibración);
- el error de distancia en barras: calibrada (VisionX), otra unidad sin corregir y sin calibrar
  (E0, si está `e0.json`: la misma cámara si el sistema creyera que es la ELP);
- qué corrigió el clásico y cuánto tarda CREStereo por cuadro, y en qué placa.

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
ENCABEZADO = 92  # alto de la franja de arriba del mosaico
ESCALA_ERROR = 60.0  # % que llena una barra
PARAMETROS = {"pitch": "inclinación", "roll": "giro", "yaw": "yaw", "log_focal": "focal",
              "radial2": "lente", "radial4": "lente"}  # fmt: skip
# Variante del video -> id del perfil (configs/camaras) para la resolución y la calibración.
PERFIL_DE = {"video-elp-full": "elp", "video-realsense-d435": "realsense-d435",
             "video-realsense-d455": "realsense-d455", "video-zed-2": "zed-2",
             "video-zed-mini": "zed-mini", "video-oak-d": "oak-d"}  # fmt: skip


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


def _perfiles() -> dict[str, object]:
    try:
        from visionx.camera.perfiles import cargar_perfiles

        return {p.id: p for p in cargar_perfiles()}
    except Exception:  # noqa: BLE001 (el mosaico se arma igual sin los perfiles)
        return {}


def _coma(x: float, dec: int = 1) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def _color_error(pct: float) -> tuple[int, int, int]:
    from visionx.app.stand import LEVEL_COLOR

    return LEVEL_COLOR["precisa" if pct < 3 else "aproximada" if pct < 10 else "no-segura"]


def _pastilla(img, t, x: int, y: int, texto: str) -> int:
    """Una pastilla de dato («87°»); devuelve el x donde termina."""
    from visionx.app.stand import INK, _rounded, text_width

    w = text_width(texto, 13) + 20
    _rounded(img, x, y, x + w, y + 24, (52, 44, 37), 12)
    t.add(texto, (x + 10, y + 4), 13, INK)
    return x + w + 6


def _barra(img, t, x: int, y: int, w: int, etiqueta: str, pct: float | None, color) -> None:
    """Etiqueta, valor y una barra fina (0 a ESCALA_ERROR %)."""
    from visionx.app.stand import FAINT, MUTED, _rounded, text_width

    t.add(etiqueta, (x, y), 13, MUTED)
    valor = "—" if pct is None else f"{_coma(pct)} %"
    t.add(valor, (x + w - text_width(valor, 17, True), y - 3), 17, color if pct else FAINT, True)
    _rounded(img, x, y + 22, x + w, y + 30, (48, 40, 34), 4)
    if pct is not None:
        largo = int(w * min(pct, ESCALA_ERROR) / ESCALA_ERROR)
        if largo >= 2:
            _rounded(img, x, y + 22, x + max(largo, 8), y + 30, color, 4)


def celda(lamina: np.ndarray | None, nombre: str, fila: dict, sin_calibrar: float | None,
          perfil, w: int, h: int, gpu: str | None = None) -> np.ndarray:  # fmt: skip
    """La tarjeta de una cámara (ver el docstring del módulo)."""
    from visionx.app.stand import (
        ACCENT,
        BG,
        FAINT,
        INK,
        LEVEL_COLOR,
        MUTED,
        Texts,
        _fondo_panel,
        _place,
    )  # fmt: skip

    img = np.full((h, w, 3), BG, np.uint8)
    t = Texts()
    _fondo_panel(img, 8, 8, w - 16, h - 16)
    t.add(nombre.removeprefix("Geometría de "), (26, 22), 23, INK, True)
    px = 26
    if fila:
        px = _pastilla(img, t, px, 58, f"{fila['hfov_deg']:.0f}°")
        px = _pastilla(img, t, px, 58, f"{_coma(fila['base_cm'])} cm entre lentes")
    if perfil is not None:
        ancho, alto = perfil.resolucion
        px = _pastilla(img, t, px, 58, f"{ancho}×{alto}")
        origen = "ChArUco" if perfil.calibracion else (
            "de fábrica" if perfil.formato == "realsense" else "hoja de datos")  # fmt: skip
        _pastilla(img, t, px, 58, f"calibración: {origen}")
    if lamina is None:
        t.add("Video pendiente", (26, h // 2), 18, FAINT)
        return t.draw(img)

    # La distancia que mide VisionX en este cuadro (el panel «Clásico» de la lámina).
    x, y, pw, ph = CLASICO
    iw, ih = 380, int(380 * ph / pw)
    _place(img, lamina[y : y + ph, x : x + pw], 18, 96, iw, ih)

    # Error de distancia (mediana de la escena).
    cx, cw = 18 + iw + 18, w - (18 + iw + 18) - 26
    t.add("ERROR DE DISTANCIA", (cx, 100), 12, MUTED, True)
    t.add("mediana de la escena", (cx, 116), 12, FAINT)
    cal = fila.get("error_clásico_pct") if fila else None
    _barra(img, t, cx, 144, cw, "VisionX", cal,
           _color_error(cal) if cal is not None else FAINT)  # fmt: skip
    sin_corregir = fila.get("error_sin corregir_pct") if fila else None
    _barra(img, t, cx, 196, cw, "Otra unidad", sin_corregir,
           LEVEL_COLOR["aproximada"])  # fmt: skip
    _barra(img, t, cx, 248, cw, "Sin calibrar", sin_calibrar,
           LEVEL_COLOR["no-segura"])  # fmt: skip

    # Alineación y rendimiento.
    yb = 96 + ih + 18
    cv2.line(img, (26, yb), (w - 26, yb), (52, 44, 37), 1, cv2.LINE_AA)
    if fila:
        corr = []
        for p in fila.get("clasico_aplica", []):
            nombre_p = PARAMETROS.get(p, p)
            if nombre_p not in corr:
                corr.append(nombre_p)
        t.add("ALINEACIÓN", (26, yb + 12), 12, MUTED, True)
        t.add("corrige " + (" · ".join(corr) if corr else "nada"), (150, yb + 10), 15, INK)
        ms = float(fila.get("ms_por_cuadro", 0))
        placa = fila.get("placa") or ((gpu or "GPU") if 0 < ms < 400 else "CPU")
        t.add("RENDIMIENTO", (26, yb + 42), 12, MUTED, True)
        rend = f"CREStereo {ms:.0f} ms por cuadro · {_coma(1000 / ms)} fps · {placa}" if ms else "—"
        t.add(rend, (150, yb + 40), 15, INK)
        ideal = fila.get("error_ideal_pct")
        if ideal is not None:
            t.add(f"con la calibración perfecta: {_coma(ideal)} %", (150, yb + 68), 13, ACCENT)
    return t.draw(img)


def _leyenda(canvas, t, x: int, y: int) -> None:
    """Barra de colores de la profundidad: rojo cerca, azul lejos."""
    from visionx.app.stand import MUTED

    grad = np.linspace(255, 0, 180).astype(np.uint8)[None, :].repeat(10, 0)
    canvas[y : y + 10, x : x + 180] = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    t.add("cerca", (x - 44, y - 4), 13, MUTED)
    t.add("lejos", (x + 188, y - 4), 13, MUTED)


def armar(carpeta: str | Path, escena: str, e0: str | Path | None = None) -> Path | None:
    """El MP4 del mosaico de una escena, o None si no hay ningún video de esa escena."""
    from visionx.app.aceleracion import nombre_gpu
    from visionx.app.stand import ACCENT, BG, INK, MUTED, H, Texts, W, text_width

    carpeta = Path(carpeta)
    caps = {var: cv2.VideoCapture(str(carpeta / f"{escena}-{var}.mp4"))
            for var, _ in CAMARAS_VIDEO if (carpeta / f"{escena}-{var}.mp4").is_file()}  # fmt: skip
    if not caps:
        return None
    n = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
    errores_e0 = _e0(e0)
    resumen = _resumen(carpeta)
    perfiles = _perfiles()
    gpu = nombre_gpu()
    salida = carpeta / f"{escena}-video-mosaico.mp4"
    writer = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    cw, ch = W // 3, (H - ENCABEZADO) // 2
    for i in range(n):
        canvas = np.full((H, W, 3), BG, np.uint8)
        t = Texts()
        t.add("VisionX", (32, 20), 32, INK, True)
        sub = "Una misma app, seis cámaras estéreo distintas"
        t.add(sub, (32 + text_width("VisionX", 32, True) + 18, 30), 21, ACCENT, True)
        leyenda = ("Error de distancia · VisionX: calibrada (alineación clásica + distancia "
                   "conocida) · otra unidad: el mismo modelo sin corregir · sin calibrar: si "
                   "creyera que es la ELP")  # fmt: skip
        t.add(leyenda, (32, 60), 15, MUTED)
        nota = f"escena {escena} · cuadro {i + 1}/{n}"
        t.add(nota, (W - 32 - text_width(nota, 15), 60), 15, MUTED)
        _leyenda(canvas, t, W - 32 - 180 - 52, 30)
        for k, (var, nombre) in enumerate(CAMARAS_VIDEO):
            lam = None
            if var in caps:
                ok, lam = caps[var].read()
                lam = lam if ok else None
            x, y = (k % 3) * cw, ENCABEZADO + (k // 3) * ch
            canvas[y : y + ch, x : x + cw] = celda(
                lam, nombre, resumen.get((escena, var), {}),
                None if var == "video-elp-full" else errores_e0.get((escena, var)),
                perfiles.get(PERFIL_DE.get(var, "")), cw, ch, gpu,
            )  # fmt: skip
        writer.write(t.draw(canvas))
    writer.release()
    for c in caps.values():
        c.release()
    return salida


def armar_todos(carpeta: str | Path, e0: str | Path | None = None) -> list[Path]:
    carpeta = Path(carpeta)
    escenas = sorted({p.name.split("-video-")[0] for p in carpeta.glob("*-video-*.mp4")})
    hechos = [armar(carpeta, e, e0) for e in escenas]
    return [p for p in hechos if p is not None]
