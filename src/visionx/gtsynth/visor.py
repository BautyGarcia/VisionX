"""Visor del GT sintético para el stand y el póster.

Muestra, vista por vista: la imagen que «ve» la ELP, la profundidad exacta del render, los
objetos con su distancia verdadera y, abajo, qué es el conjunto y qué se midió con él. Pasa solo
cada `--cada` segundos (n / p para avanzar / volver, f pantalla completa, q salir), o exporta las
láminas como JPG (`--exportar carpeta`) para el póster o una presentación.

    uv run python -m visionx.gtsynth visor data/gt-sintetico-elp
    uv run python -m visionx.gtsynth visor data/gt-sintetico-elp \\
        --exportar outputs/gtsynth/laminas --n 12
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from visionx.app.dashboard import depth_colormap
from visionx.app.stand import ACCENT, BG, INK, MUTED, PANEL, Texts, _overlaps, text_width
from visionx.detection.catalog import COCO_ES
from visionx.gtsynth.dataset import objetos_de, vistas

W, H = 1920, 1080
COLORS = [(66, 135, 245), (80, 200, 120), (240, 180, 60), (200, 90, 220), (60, 200, 220),
          (230, 110, 90), (150, 150, 240), (120, 220, 200)]  # fmt: skip


# Clases de Replica sin equivalente en el detector, en castellano (para mostrar).
REPLICA_ES = {
    "lamp": "lámpara", "cushion": "almohadón", "pillow": "almohada", "box": "caja",
    "clothing": "ropa", "rack": "estante", "cabinet": "mueble", "bin": "tacho",
    "small-appliance": "electrodoméstico", "plate": "plato", "countertop": "mesada",
    "blanket": "manta", "camera": "cámara", "shoe": "zapato", "mat": "alfombrita",
    "desk-organizer": "organizador", "tissue-paper": "pañuelos", "tablet": "tablet",
    "basket": "canasta", "picture": "cuadro", "candle": "vela", "sculpture": "escultura",
    "pot": "maceta", "plant-stand": "soporte de planta", "stool": "banqueta", "bench": "banco",
    "desk": "escritorio", "nightstand": "mesa de luz", "towel": "toalla", "pan": "sartén",
    "bike": "bicicleta", "wardrobe": "placard", "beanbag": "puf", "tv-stand": "mueble de tele",
    "kitchen-utensil": "utensilio", "utensil-holder": "portautensilios",
    "knife-block": "taco de cuchillos", "paper-towel": "rollo de cocina", "coaster": "posavasos",
    "table-runner": "camino de mesa", "cooktop": "anafe", "faucet": "canilla",
    "exercise-ball": "pelota", "comforter": "acolchado", "scarf": "bufanda",
    "shower-stall": "ducha", "bathtub": "bañera", "handrail": "baranda", "cloth": "tela",
    "chopping-board": "tabla de cortar", "set-of-clothing": "ropa", "monitor": "monitor",
}  # fmt: skip


def _nombre(o) -> str:
    if o.coco and o.coco in COCO_ES:
        return COCO_ES[o.coco][0]
    return REPLICA_ES.get(o.clase, o.clase.replace("-", " "))


def _robustez(raiz: Path) -> list[str]:
    """Una línea por cámara con los resúmenes de `gtsynth evaluar` (si están)."""
    from visionx.shared.config import ROOT

    out = []
    nombres = {"elp": "ELP calibrada", "hermana-0.5": "otra unidad (leve)",
               "hermana-1": "otra unidad", "distinta": "cámara distinta"}  # fmt: skip
    for var, nombre in nombres.items():
        for base in (ROOT / "datasets" / "gtsynth", ROOT / "outputs" / "gtsynth"):
            p = base / f"val-{var}.json"
            if p.is_file():
                r = json.loads(p.read_text(encoding="utf-8"))["por_rango"]
                vals = [round(r[k]["mae_pct"]) for k in ("1–2", "2–3", "3–4") if k in r]
                rango = f"{min(vals)} %" if min(vals) == max(vals) else f"{min(vals)}–{max(vals)} %"
                out.append(f"{nombre}: {rango}")
                break
    return out


def lamina(v, robustez: list[str], total: dict) -> np.ndarray:
    canvas = np.full((H, W, 3), BG, np.uint8)
    t = Texts()
    t.add("VisionX · ground truth sintético", (28, 14), 40, ACCENT, True)
    t.add(f"{v.escena.replace('_', ' ')} · vista {v.meta['indice'] + 1}", (720, 28), 22, MUTED)
    izq = v.izq()
    z = v.z()
    objs = [o for o in objetos_de(v) if o.z_frente_m <= 4.5][:10]
    # Panel 1: lo que ve la cámara, con los objetos y su distancia verdadera.
    x0, y0, w1, h1 = 20, 84, 1180, 664
    cv2.rectangle(canvas, (x0, y0), (x0 + w1, y0 + h1), PANEL, -1)
    s = min(w1 / izq.shape[1], h1 / izq.shape[0])
    nw, nh = int(izq.shape[1] * s), int(izq.shape[0] * s)
    ox, oy = x0 + (w1 - nw) // 2, y0 + (h1 - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = cv2.resize(izq, (nw, nh), interpolation=cv2.INTER_AREA)
    inst = cv2.resize(v.instancias().astype(np.uint16), (nw, nh), interpolation=cv2.INTER_NEAREST)
    region = canvas[oy : oy + nh, ox : ox + nw]
    placed: list[tuple[int, int, int, int]] = []
    for k, o in enumerate(objs):
        color = COLORS[k % len(COLORS)]
        mask = inst == o.id
        region[mask] = (0.55 * region[mask] + 0.45 * np.array(color)).astype(np.uint8)
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(region, cnts, -1, color, 2, cv2.LINE_AA)
        x1, y1 = int(o.caja[0] * s), int(o.caja[1] * s)
        label = f"{_nombre(o)} · {o.z_frente_m:.2f} m".replace(".", ",")
        tw = text_width(label, 21, True) + 14
        lx = min(ox + x1, ox + nw - tw)  # dentro de la imagen
        ly = max(oy + 2, oy + y1 - 32)
        rect = (lx, ly, lx + tw, ly + 30)
        if _overlaps(rect, placed):
            continue  # el objeto queda pintado y en la lista de abajo
        placed.append(rect)
        cv2.rectangle(canvas, (lx, ly), (lx + tw, ly + 30), color, -1)
        t.add(label, (lx + 7, ly + 3), 21, (20, 20, 20), True)
    t.add("Lo que ve la cámara (geometría de la ELP: 116°, 5,98 cm) y la distancia exacta de cada "
          "objeto", (x0 + 14, y0 + h1 + 8), 20, MUTED)  # fmt: skip
    # Panel 2: profundidad exacta.
    x2, w2 = 1220, 680
    cv2.rectangle(canvas, (x2, y0), (x2 + w2, y0 + 330), PANEL, -1)
    t.add("Profundidad exacta del render", (x2 + 14, y0 + 8), 20, MUTED, True)
    dm = depth_colormap(z, np.isfinite(z) & (z > 0), (0.5, 5.0))
    s2 = min((w2 - 20) / dm.shape[1], 240 / dm.shape[0])
    dw, dh = int(dm.shape[1] * s2), int(dm.shape[0] * s2)
    canvas[y0 + 50 : y0 + 50 + dh, x2 + (w2 - dw) // 2 : x2 + (w2 - dw) // 2 + dw] = cv2.resize(
        dm, (dw, dh)
    )
    t.add("0,5 m (cerca) → 5 m (lejos), cada píxel", (x2 + 14, y0 + 298), 18, MUTED)
    # Panel 3: el conjunto y lo medido.
    y3 = y0 + 346
    cv2.rectangle(canvas, (x2, y3), (x2 + w2, y0 + h1), PANEL, -1)
    t.add("El conjunto", (x2 + 14, y3 + 8), 20, MUTED, True)
    lines = [
        f"{total['vistas']} vistas de {total['escenas']} interiores escaneados (Replica)",
        "renderizadas con la cámara de VisionX, a la altura del pecho",
        f"{total['objetos']:,} objetos con distancia y nombre exactos".replace(",", "."),
        "Error de CREStereo por objeto (1–4 m):",
    ] + [f"   {r}" for r in robustez]
    for i, line in enumerate(lines):
        t.add(line, (x2 + 18, y3 + 42 + 30 * i), 21, INK if i < 3 else ACCENT, i == 3)
    # Abajo: la lista de objetos de esta vista.
    y4 = y0 + h1 + 44
    cv2.rectangle(canvas, (20, y4), (W - 20, H - 20), PANEL, -1)
    t.add("Objetos de esta vista, del más cercano al más lejano", (34, y4 + 8), 20, MUTED, True)
    cx = 34
    for k, o in enumerate(objs):
        label = f"{_nombre(o)} {o.z_frente_m:.2f} m".replace(".", ",")
        tw = text_width(label, 22, True) + 30
        if cx + tw > W - 40:
            break
        cv2.circle(canvas, (cx + 8, y4 + 74), 8, COLORS[k % len(COLORS)], -1, cv2.LINE_AA)
        t.add(label, (cx + 22, y4 + 59), 22, INK, True)
        cx += tw + 10
    return t.draw(canvas)


def run(
    raiz: str, particion: str | None, cada_s: float, exportar: str | None, n: int, semilla: int
):
    lista = list(vistas(raiz, "elp", particion))
    if not lista:
        raise SystemExit(f"no hay vistas en {raiz}")
    random.Random(semilla).shuffle(lista)

    # Para mostrar: vistas sin huecos de escaneo (ventanas abiertas, techo faltante) y con
    # varios objetos.
    def linda(v) -> bool:
        z = v.z()
        return float((~(np.isfinite(z) & (z > 0))).mean()) < 0.02 and len(v.meta["instancias"]) >= 8

    lista = [v for v in lista if linda(v)] or lista
    escenas = {v.escena for v in vistas(raiz, "elp")}
    todas = list(vistas(raiz, "elp"))
    total = {"vistas": len(todas) * 2, "escenas": len(escenas), "objetos": _contar(raiz, todas)}
    robustez = _robustez(Path(raiz))
    if exportar:
        out = Path(exportar)
        out.mkdir(parents=True, exist_ok=True)
        for k, v in enumerate(lista[:n]):
            path = out / f"gt-{k:02d}-{v.escena}-{v.meta['indice']:04d}.jpg"
            cv2.imwrite(str(path), lamina(v, robustez, total), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(path)
        return
    cv2.namedWindow("VisionX - GT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VisionX - GT", 1600, 900)
    k = 0
    while True:
        cv2.imshow("VisionX - GT", lamina(lista[k % len(lista)], robustez, total))
        key = cv2.waitKey(int(cada_s * 1000)) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            k -= 1
        elif key == ord("f"):
            full = cv2.getWindowProperty("VisionX - GT", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("VisionX - GT", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL if full == cv2.WINDOW_FULLSCREEN
                                  else cv2.WINDOW_FULLSCREEN)  # fmt: skip
        else:
            k += 1
    cv2.destroyAllWindows()


def _contar(raiz: str, todas) -> int:
    from visionx.shared.config import ROOT

    for p in (Path(raiz) / "objetos-elp.csv", ROOT / "outputs" / "gtsynth" / "objetos-elp.csv"):
        if p.is_file():
            return sum(1 for _ in p.open(encoding="utf-8")) - 1
    return 15088  # tabla de 19/11 (BITACORA), si no está el CSV
