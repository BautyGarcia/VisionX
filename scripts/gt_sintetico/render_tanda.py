"""GT sintético para adaptar la red de entrada a cualquier cámara (issue #53).

Corre en el entorno de Habitat-Sim (WSL / conda), como `render_habitat.py`, cuyas funciones usa.
Todo se renderiza como **pinhole ideal con margen**: el mismo fx y el centro en el medio, en un
lienzo `--margen` (1,25) veces más grande que la imagen nominal, para que A pueda simular otra
lente o un giro sin bordes negros. `vista.json` trae `camara` (la del lienzo: con ella las cinco
salidas son coherentes y `gtsynth verificar` funciona igual), `lienzo` y `nominal` (el recorte
centrado: ancho, alto, x0, y0, cx, cy).

Perfiles (`--perfil`): la ELP (`elp`, `elp-media` a media resolución), los seis del issue
(`angosta-60`, `base-corta-3`, `media-90`, `cuatro-tercios-90`, `base-larga-20`,
`gran-angular-130`) y `sorteada` (por vista: campo 55–130°, separación 3–20 cm, 16:9 o 4:3,
lado largo 1280; se cambian en caliente sin recargar la escena).

Poses: `--poses-desde <carpeta de una variante>` reusa exactamente las de esas vistas (val:
mismas poses que `elp`); si no, se sortean (`--vistas`, `--tipo-pose normal|altura|
poca-textura|cerca`, `--semilla`). `--ojo-derecho dx dy dz` (m) corre el ojo derecho (+y arriba,
−z adelante). `--ojo-de-pez` renderiza con el modelo de doble esfera de Habitat (solo para
evaluar: el sistema tiene que darse cuenta de que no la puede adaptar).

    ~/bin/micromamba run -n habitat python scripts/gt_sintetico/render_tanda.py \\
        --replica /mnt/d/datasets/replica --escenas frl_apartment_3 hotel_0 office_2 room_1 \\
        --perfil elp --variante elp-margen --poses-desde /mnt/d/VisionX/data/gt-sintetico-elp/elp \\
        --salida /mnt/d/VisionX/data/gt-sintetico-elp
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import habitat_sim
import magnum as mn
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from render_habitat import fijar_pose, objetos  # noqa: E402

ELP_FX = 601.0069353622214
ELP_B = 0.05980835164566523


def fx_de(hfov_deg: float, ancho: int) -> float:
    return (ancho / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


# nombre -> (ancho nominal, alto nominal, fx, separación m)
MODELO_PEZ = "ojo-de-pez-doble-esfera"  # `camara.modelo` en vista.json
PERFILES = {
    "elp": (1920, 1080, ELP_FX, ELP_B),
    "elp-media": (960, 540, ELP_FX / 2.0, ELP_B),
    "angosta-60": (1280, 720, fx_de(60, 1280), 0.06),
    "base-corta-3": (1280, 720, fx_de(80, 1280), 0.03),
    "media-90": (1920, 1080, fx_de(90, 1920), 0.12),
    "cuatro-tercios-90": (1280, 960, fx_de(90, 1280), 0.06),
    "base-larga-20": (1920, 1080, fx_de(100, 1920), 0.20),
    "gran-angular-130": (1920, 1080, fx_de(130, 1920), 0.06),
    # Geometría publicada de cámaras estéreo conocidas (campo horizontal y separación de la hoja
    # de datos del fabricante; resolución de un ojo). Solo la geometría: no el sensor.
    "zed-2": (1920, 1080, fx_de(110, 1920), 0.12),  # Stereolabs ZED 2: 110°, 120 mm
    "zed-mini": (1920, 1080, fx_de(102, 1920), 0.063),  # Stereolabs ZED Mini: 102°, 63 mm
    "realsense-d435": (1280, 720, fx_de(87, 1280), 0.05),  # Intel RealSense D435(i): 87°, 50 mm
    "realsense-d455": (1280, 720, fx_de(86, 1280), 0.095),  # Intel RealSense D455: 86°, 95 mm
    "oak-d": (1280, 800, fx_de(80, 1280), 0.075),  # Luxonis OAK-D Pro / S2: 80°, 75 mm
    # Solo con --ojo-de-pez y --margen 1: lente de doble esfera de ~180°, para evaluar.
    "ojo-de-pez": (1280, 960, 384.0, 0.06),
}
ESTRUCTURA = {"wall", "floor", "ceiling"}


def lienzo(ancho: int, alto: int, margen: float) -> tuple[int, int, int, int]:
    """(ancho, alto) del lienzo y (x0, y0) del recorte nominal; márgenes enteros y simétricos."""
    mx = round(ancho * (margen - 1) / 2)
    my = round(alto * (margen - 1) / 2)
    return ancho + 2 * mx, alto + 2 * my, mx, my


# Objetos de la escena cargada cuando no vienen del `semantic_scene` de Replica (HSSD): los
# arma `preparar_hssd` y los usa `sortear_poses`.
_OBJETOS: list | None = None


def categorias_hssd(raiz: Path) -> dict[str, str]:
    """Plantilla de objeto de HSSD -> categoría (metadata/fpmodels-with-decomposed.csv)."""
    import csv

    out = {}
    with open(raiz / "metadata" / "fpmodels-with-decomposed.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cat = r.get("main_category") or (r.get("wnsynsetkey") or "").split(".")[0]
            out[r["id"]] = cat or "otro"
    return out


def preparar_hssd(sim, categorias: dict[str, str]) -> dict[int, str]:
    """Navmesh, un id semántico por objeto (el sensor `inst` los dibuja) y la tabla id -> clase.

    HSSD no trae navmesh: se calcula con los objetos estáticos, para una persona de 1,7 m. La
    sala (paredes, piso, techo) queda con id 0.
    """
    global _OBJETOS
    # Luz de casa iluminada: dos luces de arriba y una que acompaña a la cámara (con la luz por
    # defecto de Habitat el brillo medio daba 30–50 sobre 255 y se perdía la textura).
    gfx = habitat_sim.gfx
    luces = [
        gfx.LightInfo(vector=[0.3, 1.0, 0.4, 0.0], color=[1.6, 1.6, 1.5],
                      model=gfx.LightPositionModel.Global),
        gfx.LightInfo(vector=[-0.4, 1.0, -0.3, 0.0], color=[1.2, 1.2, 1.2],
                      model=gfx.LightPositionModel.Global),
        gfx.LightInfo(vector=[0.0, 0.0, 1.0, 0.0], color=[1.0, 1.0, 1.0],
                      model=gfx.LightPositionModel.Camera),
    ]  # fmt: skip
    sim.set_light_setup(luces, gfx.DEFAULT_LIGHTING_KEY)
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_height, ns.agent_radius = 1.7, 0.2
    ns.include_static_objects = True
    if not sim.recompute_navmesh(sim.pathfinder, ns):
        raise RuntimeError("no se pudo calcular la navmesh")
    rom = sim.get_rigid_object_manager()
    clases, objs = {}, []
    for k, handle in enumerate(sorted(rom.get_object_handles()), start=1):
        obj = rom.get_object_by_handle(handle)
        obj.semantic_id = k
        # «<hash>_:0000» o, si el modelo viene en partes, «<hash>_part_15_:0000».
        plantilla = handle.split("/")[-1].split(":")[0].rstrip("_").split("_part_")[0]
        clase = categorias.get(plantilla, "otro")
        clases[k] = clase
        bb = obj.root_scene_node.cumulative_bb
        centro = obj.transformation.transform_point(bb.center())
        if clase not in NO_OBJETO_HSSD and bb.size().length() > 0.05:
            objs.append((k, clase, np.asarray(centro, dtype=np.float64)))
    _OBJETOS = objs
    return clases


# Categorías de HSSD que no se apuntan (estructura o cosas a las que nadie pregunta la distancia).
NO_OBJETO_HSSD = frozenset({"otro", "window", "door", "curtain", "ceiling_lamp", "wall_lamp",
                            "carpet", "picture", "counter", "stairs", "fireplace"})  # fmt: skip


def armar(
    malla: str,
    ancho_l: int,
    alto_l: int,
    fx: float,
    base: float,
    ojo_der: tuple[float, float, float],
    ojo_de_pez: bool,
) -> habitat_sim.Simulator:
    b = habitat_sim.SimulatorConfiguration()
    if "::" in malla:  # «config_del_dataset::escena» (HSSD)
        b.scene_dataset_config_file, b.scene_id = malla.split("::")
        b.load_semantic_mesh = False
        # HSSD trae materiales con luz física y ninguna luz: sin esto las vistas salen oscuras.
        b.override_scene_light_defaults = True
        b.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
        if hasattr(b, "pbr_image_based_lighting"):  # luz de entorno para los materiales PBR
            b.pbr_image_based_lighting = True
    else:
        b.scene_id = malla
    b.gpu_device_id = -1
    b.enable_physics = False
    specs = []
    for uuid, tipo, pos in [
        ("izq", habitat_sim.SensorType.COLOR, (0.0, 0.0, 0.0)),
        ("der", habitat_sim.SensorType.COLOR, (base + ojo_der[0], ojo_der[1], ojo_der[2])),
        ("z", habitat_sim.SensorType.DEPTH, (0.0, 0.0, 0.0)),
        ("inst", habitat_sim.SensorType.SEMANTIC, (0.0, 0.0, 0.0)),
    ]:
        if ojo_de_pez:
            s = habitat_sim.FisheyeSensorDoubleSphereSpec()
            s.focal_length = [fx, fx]
            s.principal_point_offset = [ancho_l / 2.0, alto_l / 2.0]
            s.xi, s.alpha = -0.27, 0.56  # una lente de ojo de pez típica (~180°)
            s.sensor_model_type = habitat_sim.FisheyeSensorModelType.DOUBLE_SPHERE
        else:
            s = habitat_sim.CameraSensorSpec()
            s.hfov = math.degrees(2.0 * math.atan(ancho_l / (2.0 * fx)))
        s.uuid, s.sensor_type = uuid, tipo
        s.resolution = [alto_l, ancho_l]
        s.position = list(pos)
        specs.append(s)
    agente = habitat_sim.agent.AgentConfiguration(sensor_specifications=specs)
    return habitat_sim.Simulator(habitat_sim.Configuration(b, [agente]))


def poses_desde(carpeta: Path) -> list[dict]:
    out = []
    for js in sorted(carpeta.glob("*/vista.json")):
        v = json.loads(js.read_text(encoding="utf-8"))
        out.append({"ojo": np.array(v["pose"]["ojo_izq"]), "yaw": v["pose"]["yaw_rad"],
                    "pitch": v["pose"]["pitch_rad"], "altura": v["pose"]["altura_m"],
                    "apuntado": v["apuntado"], "tipo": v.get("tipo_pose", "normal"),
                    "secuencia": v.get("secuencia")})  # fmt: skip
    return out


def _recorte(arr: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    return arr[y0 : y0 + h, x0 : x0 + w]


def secuencias(sim, rng, n, rec, d_min, d_max, min_px, ids, largo: int = 8) -> list[dict]:
    """n // largo recorridos: arranca apuntando a un objeto y avanza ~10 cm por cuadro
    (hacia donde mira, con un giro suave de ±3°), como una persona que camina con la cámara.

    Camina sobre la navmesh (`try_step`) y gira 30–60° si adelante, en el centro de la imagen,
    queda algo a menos de 0,6 m: sin eso, algunos recorridos terminaban metidos en una pared
    (19/11: cuadros con todo a centímetros, que no se pueden verificar).
    """
    out = []
    for r, inicio in enumerate(sortear_poses(sim, rng, n // largo, "normal", rec, max(d_min, 1.2),
                                              d_max, min_px, ids)):  # fmt: skip
        ojo, yaw = inicio["ojo"].copy(), inicio["yaw"]
        altura = float(inicio["altura"])
        for f in range(largo):
            for _ in range(6):  # mirar adelante; si está muy cerca, girar y volver a mirar
                fijar_pose(sim, ojo, yaw, inicio["pitch"])
                z = _recorte(np.asarray(ver(sim, "z")["z"]), *rec)
                h, w = z.shape
                centro = z[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
                centro = centro[np.isfinite(centro) & (centro > 0)]
                if centro.size and float(np.percentile(centro, 10)) >= 0.6:
                    break
                yaw += math.radians(float(rng.choice([-1, 1])) * rng.uniform(30, 60))
            out.append({**inicio, "ojo": ojo.copy(), "yaw": yaw, "tipo": "secuencia",
                        "secuencia": {"recorrido": r, "cuadro": f}})  # fmt: skip
            pie = ojo - np.array([0.0, altura, 0.0])
            destino = pie + 0.10 * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
            pie = np.asarray(sim.pathfinder.try_step(pie, destino), dtype=np.float64)
            ojo = pie + np.array([0.0, altura, 0.0])
            yaw += math.radians(rng.uniform(-3, 3))
    return out


def recorrido(sim, rng, n, rec, d_min, d_max, min_px, ids) -> list[dict]:
    """Un solo recorrido largo de n cuadros (video): pasos de 5 cm y giro suave.

    Arranca mirando a un objeto a 1,5–4 m. El giro cambia de a poco (velocidad angular que
    deriva), y si adelante queda algo a menos de 0,8 m gira hacia el lado más libre. Camina
    sobre la navmesh (`try_step`), así que nunca atraviesa paredes ni muebles.
    """
    for _ in range(20):  # arrancar con al menos 1,5 m libres adelante
        inicio = sortear_poses(sim, rng, 1, "normal", rec, max(d_min, 1.5), d_max, min_px, ids)[0]
        fijar_pose(sim, inicio["ojo"], inicio["yaw"], inicio["pitch"])
        z0 = _recorte(np.asarray(ver(sim, "z")["z"]), *rec)
        h0, w0 = z0.shape
        c0 = z0[h0 // 3 : 2 * h0 // 3, w0 // 3 : 2 * w0 // 3]
        c0 = c0[np.isfinite(c0) & (c0 > 0)]
        if c0.size and float(np.percentile(c0, 10)) >= 1.5:
            break
    ojo, yaw = inicio["ojo"].copy(), inicio["yaw"]
    trabado = 0
    altura, pitch = float(inicio["altura"]), inicio["pitch"]
    giro = 0.0  # grados por cuadro
    out = []
    for f in range(n):
        fijar_pose(sim, ojo, yaw, pitch)
        z = _recorte(np.asarray(ver(sim, "z")["z"]), *rec)
        h, w = z.shape
        franja = z[h // 3 : 2 * h // 3]
        izq_l = franja[:, : w // 3]
        cen = franja[:, w // 3 : 2 * w // 3]
        der_l = franja[:, 2 * w // 3 :]

        def cerca(a):
            a = a[np.isfinite(a) & (a > 0)]
            return float(np.percentile(a, 10)) if a.size else 0.0

        if cerca(cen) < 0.8 or trabado >= 2:  # algo adelante o no avanza: girar al lado libre
            if abs(giro) < 5.0:  # elegir el lado una vez y mantenerlo hasta salir
                giro = 6.0 if cerca(izq_l) > cerca(der_l) else -6.0
            giro = math.copysign(9.0 if trabado >= 4 else 6.0, giro)
        else:
            giro = float(np.clip(0.85 * giro + rng.normal(0, 0.4), -2.5, 2.5))
        out.append({**inicio, "ojo": ojo.copy(), "yaw": yaw, "tipo": "recorrido",
                    "secuencia": {"recorrido": 0, "cuadro": f}})  # fmt: skip
        yaw += math.radians(giro)
        if cerca(cen) >= 0.8:
            pie = ojo - np.array([0.0, altura, 0.0])
            destino = pie + 0.05 * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
            nuevo = np.asarray(sim.pathfinder.try_step(pie, destino), dtype=np.float64)
            trabado = trabado + 1 if np.linalg.norm(nuevo - pie) < 0.02 else 0
            ojo = nuevo + np.array([0.0, altura, 0.0])
    return out


def ver(sim, *nombres: str) -> dict:
    """Dibuja solo esos sensores (sortear con `inst` y `z` cuesta la mitad que con los cuatro)."""
    out = {}
    for nombre in nombres:
        s = sim._sensors[nombre]
        s.draw_observation()
        out[nombre] = s.get_observation()
    return out


def sortear_poses(sim, rng, n, tipo, rec, d_min, d_max, min_px, ids_estructura) -> list[dict]:
    """Poses nuevas. `rec` = (x0, y0, w, h) del recorte nominal: lo visible se mide ahí.

    `min_px` está pensado para 1920×1080: se escala con el área del recorte nominal (a media
    resolución, el mismo objeto ocupa la cuarta parte de los píxeles).
    """
    min_px = max(60, round(min_px * rec[2] * rec[3] / (1920 * 1080)))
    if tipo == "secuencia":
        return secuencias(sim, rng, n, rec, d_min, d_max, min_px, ids_estructura)
    if tipo == "recorrido":
        return recorrido(sim, rng, n, rec, d_min, d_max, min_px, ids_estructura)
    objs = _OBJETOS if _OBJETOS is not None else objetos(sim)
    out, intentos = [], 0
    while len(out) < n and intentos < n * 60:
        intentos += 1
        if tipo == "poca-textura":
            p = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float64)
            if not np.all(np.isfinite(p)):
                continue
            altura = float(rng.uniform(1.2, 1.4))
            yaw = float(rng.uniform(-math.pi, math.pi))
            pitch = math.radians(float(rng.choice([-55.0, 0.0, 50.0])) + rng.uniform(-8, 8))
            ojo = p + np.array([0.0, altura, 0.0])
            fijar_pose(sim, ojo, yaw, pitch)
            inst = _recorte(np.asarray(ver(sim, "inst")["inst"]), *rec)
            if np.isin(inst, ids_estructura).mean() < 0.75:  # que domine pared, piso o techo
                continue
            out.append({"ojo": ojo, "yaw": yaw, "pitch": pitch, "altura": altura,
                        "apuntado": {"id": 0, "clase": "estructura"}, "tipo": tipo})  # fmt: skip
            continue
        idx, clase, centro = objs[rng.integers(len(objs))]
        if tipo == "altura":
            altura = float(rng.choice([1.0, 1.6]))
        else:
            altura = float(rng.uniform(1.2, 1.4))
        lo, hi = (0.3, 0.5) if tipo == "cerca" else (d_min, d_max)
        p = np.asarray(
            sim.pathfinder.get_random_navigable_point_near(centro, max(hi, 1.0), max_tries=50),
            np.float64,
        )
        if not np.all(np.isfinite(p)):
            continue
        ojo = p + np.array([0.0, altura, 0.0])
        delta = centro - ojo
        horiz = math.hypot(delta[0], delta[2])
        if tipo != "cerca" and not (lo <= horiz <= hi):
            continue
        yaw = math.atan2(-delta[0], -delta[2]) + math.radians(rng.uniform(-20, 20))
        if tipo == "altura":
            pitch = math.radians(rng.uniform(-30, 10))
        else:
            pitch = math.atan2(delta[1], max(horiz, 0.05)) + math.radians(rng.uniform(-8, 8))
            pitch = float(np.clip(pitch, math.radians(-35), math.radians(10)))
        fijar_pose(sim, ojo, yaw, pitch)
        obs = ver(sim, "inst", "z")
        inst = _recorte(np.asarray(obs["inst"]), *rec)
        z = _recorte(np.asarray(obs["z"]), *rec)
        vis = inst == idx
        if vis.sum() < min_px:
            continue
        zf = float(np.percentile(z[vis], 20))
        if not (lo <= zf <= hi):
            continue
        out.append({"ojo": ojo, "yaw": yaw, "pitch": pitch, "altura": altura,
                    "apuntado": {"id": idx, "clase": clase}, "tipo": tipo})  # fmt: skip
    return out


def exponer(obs: dict, objetivo: float = 110.0) -> float | None:
    """Si la vista quedó oscura, sube la exposición con la misma curva en los dos ojos.

    Es una transformación punto a punto idéntica en izquierda y derecha: no cambia qué píxel
    corresponde a cuál (la disparidad y la Z siguen exactas). Devuelve el exponente usado.
    """
    izq = np.asarray(obs["izq"])[..., :3].astype(np.float32)
    media = float(izq.mean())
    if media >= 0.7 * objetivo or media <= 1.0:
        return None
    g = float(np.clip(np.log(objetivo / 255.0) / np.log(media / 255.0), 0.35, 1.0))
    for ojo in ("izq", "der"):
        img = np.asarray(obs[ojo]).copy()
        rgb = img[..., :3].astype(np.float32) / 255.0
        img[..., :3] = np.clip(255.0 * rgb**g + 0.5, 0, 255).astype(np.uint8)
        obs[ojo] = img
    return g


def guardar(carpeta: Path, obs, meta: dict) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(obs["izq"])[:, :, :3]).save(carpeta / "izq.png")
    Image.fromarray(np.asarray(obs["der"])[:, :, :3]).save(carpeta / "der.png")
    np.savez_compressed(carpeta / "z.npz", z=np.asarray(obs["z"], dtype=np.float32))
    Image.fromarray(np.asarray(obs["inst"]).astype(np.uint16)).save(carpeta / "inst.png")
    (carpeta / "vista.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                        encoding="utf-8")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replica", help="carpeta de Replica (una escena por carpeta)")
    ap.add_argument("--hssd", help="carpeta de hssd-hab (en vez de --replica)")
    ap.add_argument(
        "--ojo-sorteado",
        type=float,
        default=0.0,
        metavar="M",
        help="por vista, corre el ojo derecho ±M metros en y y en z (paralaje real)",
    )
    ap.add_argument("--escenas", nargs="+", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--variante", required=True, help="nombre de la carpeta de salida")
    ap.add_argument("--perfil", required=True, choices=sorted(PERFILES) + ["sorteada"])
    ap.add_argument("--margen", type=float, default=1.25)
    ap.add_argument("--poses-desde", help="carpeta de una variante: reusa sus poses por escena")
    ap.add_argument("--vistas", type=int, default=40)
    ap.add_argument("--tipo-pose", default="normal",
                    choices=["normal", "altura", "poca-textura", "cerca", "secuencia",
                             "recorrido"])  # fmt: skip
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--ojo-derecho", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("DX", "DY", "DZ"))  # fmt: skip
    ap.add_argument("--ojo-de-pez", action="store_true")
    ap.add_argument("--d-min", type=float, default=0.5)
    ap.add_argument("--d-max", type=float, default=4.0)
    ap.add_argument("--min-px", type=int, default=400)
    ap.add_argument(
        "--proporciones",
        nargs="+",
        default=["16:9", "4:3"],
        choices=["16:9", "4:3"],
        help="con --perfil sorteada; cada proporción vuelve a cargar la escena",
    )
    ap.add_argument("--max-vistas", type=int, default=0, help="por escena (pruebas); 0 = todas")
    args = ap.parse_args()

    if bool(args.replica) == bool(args.hssd):
        ap.error("hace falta --replica o --hssd (uno solo)")
    hssd = Path(args.hssd) if args.hssd else None
    cats = categorias_hssd(hssd) if hssd else {}
    for escena in args.escenas:
        if hssd:
            malla = f"{hssd / 'hssd-hab.scene_dataset_config.json'}::{escena}"
        else:
            malla = str(Path(args.replica) / escena / "habitat" / "mesh_semantic.ply")
        rng = np.random.default_rng(
            [args.semilla, sum(map(ord, escena)), sum(map(ord, args.variante))]
        )
        if args.perfil == "sorteada":
            grupos = {k: v for k, v in {"16:9": (1280, 720), "4:3": (1280, 960)}.items()
                      if k in args.proporciones}  # fmt: skip
            elegido = [str(rng.choice(list(grupos))) for _ in range(args.vistas)]
        else:
            grupos = {"fijo": PERFILES[args.perfil][:2]}
            elegido = ["fijo"] * args.vistas
        hechas = 0
        poses_fijas = poses_desde(Path(args.poses_desde) / escena) if args.poses_desde else None
        if poses_fijas is not None and args.max_vistas:
            poses_fijas = poses_fijas[: args.max_vistas]
        if args.max_vistas:
            elegido = elegido[: args.max_vistas]
        for grupo, (wn, hn) in grupos.items():
            idxs = [i for i, g in enumerate(elegido) if g == grupo]
            if poses_fijas is not None:
                idxs = list(range(len(poses_fijas))) if grupo == list(grupos)[0] else []
            if not idxs:
                continue
            wl, hl, x0, y0 = lienzo(wn, hn, args.margen)
            fx = PERFILES[args.perfil][2] if args.perfil != "sorteada" else fx_de(90, wn)
            base = PERFILES[args.perfil][3] if args.perfil != "sorteada" else 0.06
            sim = armar(malla, wl, hl, fx, base, tuple(args.ojo_derecho), args.ojo_de_pez)
            if hssd:
                clases = preparar_hssd(sim, cats)
            ids_estructura = [] if hssd else [
                int(o.id.split("_")[-1]) for o in sim.semantic_scene.objects
                if o is not None and o.category is not None and o.category.name() in ESTRUCTURA
            ]  # fmt: skip
            if not hssd:
                clases = {i: c for i, c, _ in objetos(sim)}
            if poses_fijas is not None:
                poses = poses_fijas
            else:
                poses = sortear_poses(
                    sim,
                    rng,
                    len(idxs),
                    args.tipo_pose,
                    (x0, y0, wn, hn),
                    args.d_min,
                    args.d_max,
                    args.min_px,
                    ids_estructura,
                )
            for k, pose in zip(idxs, poses):  # noqa: B905 (Habitat corre en Python 3.9: zip sin strict)
                cam_fx, cam_b = fx, base
                fijar_pose(sim, pose["ojo"], pose["yaw"], pose["pitch"])
                if args.perfil == "sorteada":  # campo y separación de esta vista, en caliente
                    # DESPUÉS de fijar la pose: set_state(reset_sensors=True) vuelve los sensores
                    # a la posición de su especificación.
                    hfov_n = float(rng.uniform(55, 130))
                    cam_fx = fx_de(hfov_n, wn)
                    cam_b = float(rng.uniform(0.03, 0.20))
                    hfov_l = math.degrees(2.0 * math.atan(wl / (2.0 * cam_fx)))
                    for name in ("izq", "der", "z", "inst"):
                        sim._sensors[name]._sensor_object.fov = mn.Deg(hfov_l)
                ojo_der = [float(v) for v in args.ojo_derecho]
                if args.ojo_sorteado:  # otra unidad: el ojo derecho corrido unos mm, con paralaje
                    ojo_der[1] += float(rng.uniform(-args.ojo_sorteado, args.ojo_sorteado))
                    ojo_der[2] += float(rng.uniform(-args.ojo_sorteado, args.ojo_sorteado))
                if args.perfil == "sorteada" or args.ojo_sorteado:
                    # DESPUÉS de fijar la pose (set_state reinicia los sensores).
                    sim._sensors["der"]._sensor_object.node.translation = mn.Vector3(
                        cam_b + ojo_der[0], ojo_der[1], ojo_der[2]
                    )
                obs = dict(sim.get_sensor_observations())
                gamma = exponer(obs) if hssd else None
                inst = np.asarray(obs["inst"])
                ids = [int(i) for i in np.unique(_recorte(inst, x0, y0, wn, hn)) if i != 0]
                meta = {
                    "escena": escena, "variante": args.variante, "indice": k,
                    "semilla": args.semilla, "perfil": args.perfil,
                    "camara": {"ancho": wl, "alto": hl, "fx": cam_fx, "fy": cam_fx,
                               "cx": wl / 2 - 0.5, "cy": hl / 2 - 0.5, "base_m": cam_b,
                               "hfov_deg": math.degrees(2 * math.atan(wl / (2 * cam_fx))),
                               "modelo": MODELO_PEZ if args.ojo_de_pez else "pinhole"},
                    "lienzo": {"ancho": wl, "alto": hl},
                    "nominal": {"ancho": wn, "alto": hn, "x0": x0, "y0": y0,
                                "cx": wn / 2 - 0.5, "cy": hn / 2 - 0.5,
                                "hfov_deg": math.degrees(2 * math.atan(wn / (2 * cam_fx)))},
                    "ojo_derecho_m": ojo_der,
                    "fuente": "hssd" if hssd else "replica",
                    "exposicion_gamma": gamma,
                    "pose": {"ojo_izq": [float(x) for x in pose["ojo"]], "yaw_rad": pose["yaw"],
                             "pitch_rad": pose["pitch"], "altura_m": pose["altura"]},
                    "tipo_pose": pose.get("tipo", args.tipo_pose),
                    "secuencia": pose.get("secuencia"),
                    "apuntado": pose["apuntado"],
                    "instancias": {str(i): clases.get(i, "otro") for i in ids},
                }  # fmt: skip
                guardar(Path(args.salida) / args.variante / escena / f"{k:04d}", obs, meta)
                hechas += 1
            sim.close()
        print(f"{args.variante}/{escena}: {hechas} vistas", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
