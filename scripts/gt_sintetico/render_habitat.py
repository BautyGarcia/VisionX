"""GT-sintético-ELP: renderiza pares estéreo de Replica con la geometría rectificada de la ELP.

Corre en el entorno de Habitat-Sim (no en el de VisionX): en Windows, dentro de WSL con
micromamba; en Linux o macOS, con conda. Ver docs/GT-SINTETICO.md.

Por cada vista escribe, en <salida>/<variante>/<escena>/<nnnn>/:
  izq.png, der.png   imágenes rectificadas (RGB, 8 bits)
  z.npz              Z (profundidad axial, metros, float32) del ojo izquierdo, clave "z"
  inst.png           id de instancia de Replica por píxel del ojo izquierdo (uint16; 0 = nada)
  vista.json         cámara, pose, objeto apuntado y tabla id -> clase de las instancias visibles

La cámara apunta a un objeto de la escena desde 0,5–4 m, a la altura del pecho, con un desvío
al azar para que el objeto no quede siempre centrado. Misma semilla = mismas vistas.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image

# Geometría rectificada de la ELP (configs/calib/elp-2026-11-11-corregida.yaml, a 1920x1080).
# Habitat usa cámara pinhole con el centro en el medio de la imagen y píxeles cuadrados: el
# centro real (958,6; 537,6) difiere en ~1-2 px, y la vista guarda la geometría exacta usada.
VARIANTES = {
    "elp": {"ancho": 1920, "alto": 1080, "fx": 601.0069353622214, "base_m": 0.05980835164566523},
    # Otra cámara estéreo: más separación y menos gran angular. Con la geometría de la ELP da
    # metros equivocados: es la que tiene que terminar pidiendo una distancia conocida.
    "distinta": {
        "ancho": 1920,
        "alto": 1080,
        "fx": 960.0 / math.tan(math.radians(45)),
        "base_m": 0.12,
    },
}

# Clases de Replica que no son objetos a los que una persona preguntaría la distancia.
NO_OBJETO = {
    "wall",
    "floor",
    "ceiling",
    "undefined",
    "wall-plug",
    "wall-cabinet",
    "beam",
    "pipe",
    "vent",
    "panel",
    "rug",
    "base-cabinet",
    "blinds",
    "curtain",
    "window",
    "door",
    "anonymize_text",
    "anonymize_picture",
    "pillar",
    "stair",
    "non-plane",
    "major-appliance",
    "switch",
    "cloth",
    "ceiling-lamp",
    "wall-plaque",
    "shelf",
}


def hfov_deg(ancho: int, fx: float) -> float:
    return math.degrees(2.0 * math.atan(ancho / (2.0 * fx)))


def armar_sim(
    escena: str, dataset_cfg: str | None, cam: dict, altura_m: float
) -> habitat_sim.Simulator:
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = escena
    if dataset_cfg:
        backend.scene_dataset_config_file = dataset_cfg
    backend.gpu_device_id = -1  # en WSL/AMD no hay CUDA: que Magnum elija el contexto EGL
    backend.enable_physics = False
    hfov = hfov_deg(cam["ancho"], cam["fx"])
    specs = []
    for uuid, tipo, x in [
        ("izq", habitat_sim.SensorType.COLOR, 0.0),
        ("der", habitat_sim.SensorType.COLOR, cam["base_m"]),
        ("z", habitat_sim.SensorType.DEPTH, 0.0),
        ("inst", habitat_sim.SensorType.SEMANTIC, 0.0),
    ]:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = tipo
        s.resolution = [cam["alto"], cam["ancho"]]
        s.hfov = hfov
        s.position = [x, altura_m, 0.0]  # +X = derecha de la cámara
        specs.append(s)
    agente = habitat_sim.agent.AgentConfiguration(sensor_specifications=specs)
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agente]))


def objetos(sim: habitat_sim.Simulator) -> list[tuple[int, str, np.ndarray]]:
    """(id, clase, centro) de las instancias de la escena que son objetos."""
    out = []
    for obj in sim.semantic_scene.objects:
        if obj is None or obj.category is None:
            continue
        clase = obj.category.name()
        if not clase or clase in NO_OBJETO:
            continue
        idx = int(obj.id.split("_")[-1])
        centro = obj.aabb.center
        centro = centro() if callable(centro) else centro  # método en habitat-sim 0.3
        out.append((idx, clase, np.asarray(centro, dtype=np.float64)))
    return out


def apuntar(sim, rng, centro, altura_m, d_min, d_max, intentos=40):
    """Busca un punto del piso a d_min–d_max del objeto desde el que se lo vea."""
    for _ in range(intentos):
        p = np.asarray(
            sim.pathfinder.get_random_navigable_point_near(centro, d_max, max_tries=50),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(p)):
            continue
        ojo = p + np.array([0.0, altura_m, 0.0])
        delta = centro - ojo
        horiz = math.hypot(delta[0], delta[2])
        if not (d_min <= horiz <= d_max):
            continue
        # Habitat: la cámara mira hacia -Z. yaw alrededor de +Y.
        yaw = math.atan2(-delta[0], -delta[2]) + math.radians(rng.uniform(-20, 20))
        pitch = math.atan2(delta[1], horiz) + math.radians(rng.uniform(-8, 8))
        pitch = float(np.clip(pitch, math.radians(-35), math.radians(10)))
        return p, yaw, pitch
    return None


def fijar_pose(sim, p, yaw, pitch):
    """El agente completo gira (yaw) y se inclina (pitch); los sensores van en su origen."""
    estado = habitat_sim.AgentState()
    estado.position = p.astype(np.float32)
    estado.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0])) * quat_from_angle_axis(
        pitch, np.array([1.0, 0.0, 0.0])
    )
    sim.get_agent(0).set_state(estado, reset_sensors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replica", required=True, help="carpeta de replica_v1")
    ap.add_argument("--escenas", nargs="+", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--vistas", type=int, default=40, help="por escena")
    ap.add_argument("--variantes", nargs="+", default=["elp"], choices=sorted(VARIANTES))
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--d-min", type=float, default=0.5)
    ap.add_argument("--d-max", type=float, default=4.0)
    ap.add_argument(
        "--min-px", type=int, default=400, help="píxeles visibles mínimos del objeto apuntado"
    )
    args = ap.parse_args()

    replica = Path(args.replica)
    for escena in args.escenas:
        malla = str(replica / escena / "habitat" / "mesh_semantic.ply")
        # Sin la config del dataset: con ella Habitat dibuja mesh.ply (texturas PTex) y la build
        # de conda no las soporta (sale todo negro). mesh_semantic.ply trae color por vértice,
        # la navmesh y las instancias, y se ve bien.
        cfg = None
        rng = np.random.default_rng([args.semilla, sum(map(ord, escena))])
        # Las poses se sortean con la primera variante y se reusan en las demás. Un simulador
        # por vez: Habitat no admite bien dos contextos abiertos en el mismo proceso.
        poses = []
        for k, var in enumerate(args.variantes):
            cam = VARIANTES[var]
            sim = armar_sim(malla, cfg, cam, 0.0)  # la altura va en la pose
            objs = objetos(sim)
            if not objs:
                print(f"{escena}: sin objetos semánticos; ¿falta info_semantic.json?")
                sim.close()
                break
            clases = {i: c for i, c, _ in objs}
            if k == 0:
                intentos = 0
                while len(poses) < args.vistas and intentos < args.vistas * 30:
                    intentos += 1
                    idx, clase, centro = objs[rng.integers(len(objs))]
                    altura = float(rng.uniform(1.2, 1.4))
                    r = apuntar(sim, rng, centro, altura, args.d_min, args.d_max)
                    if r is None:
                        continue
                    p, yaw, pitch = r
                    p_ojo = p + np.array([0.0, altura, 0.0])
                    fijar_pose(sim, p_ojo, yaw, pitch)
                    obs = sim.get_sensor_observations()
                    inst = np.asarray(obs["inst"])
                    visible = inst == idx
                    if visible.sum() < args.min_px:
                        continue
                    zf = float(np.percentile(np.asarray(obs["z"])[visible], 20))
                    if not (args.d_min <= zf <= args.d_max):
                        continue
                    poses.append((idx, clase, p_ojo, yaw, pitch, altura))
            for n, (idx, clase, p_ojo, yaw, pitch, altura) in enumerate(poses):
                fijar_pose(sim, p_ojo, yaw, pitch)
                obs = sim.get_sensor_observations()
                carpeta = Path(args.salida) / var / escena / f"{n:04d}"
                carpeta.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(obs["izq"])[:, :, :3]).save(carpeta / "izq.png")
                Image.fromarray(np.asarray(obs["der"])[:, :, :3]).save(carpeta / "der.png")
                np.savez_compressed(carpeta / "z.npz", z=np.asarray(obs["z"], dtype=np.float32))
                inst = np.asarray(obs["inst"]).astype(np.uint16)
                Image.fromarray(inst).save(carpeta / "inst.png")
                ids = [int(i) for i in np.unique(inst) if i != 0]
                vista = {
                    "escena": escena,
                    "variante": var,
                    "indice": n,
                    "semilla": args.semilla,
                    "camara": {
                        "ancho": cam["ancho"],
                        "alto": cam["alto"],
                        "fx": cam["fx"],
                        "fy": cam["fx"],
                        "cx": cam["ancho"] / 2 - 0.5,
                        "cy": cam["alto"] / 2 - 0.5,
                        "base_m": cam["base_m"],
                        "hfov_deg": hfov_deg(cam["ancho"], cam["fx"]),
                    },
                    "pose": {
                        "ojo_izq": [float(x) for x in p_ojo],
                        "yaw_rad": yaw,
                        "pitch_rad": pitch,
                        "altura_m": altura,
                    },
                    "apuntado": {"id": idx, "clase": clase},
                    "instancias": {str(i): clases.get(i, "otro") for i in ids},
                }
                (carpeta / "vista.json").write_text(
                    json.dumps(vista, ensure_ascii=False, indent=1), encoding="utf-8"
                )
            print(f"{var}/{escena}: {len(poses)} vistas")
            sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
