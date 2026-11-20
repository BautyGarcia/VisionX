"""Frases frecuentes de la app, para precalentar la caché de voz de Edge (T08).

Con Edge (Mateo) la primera vez que se dice una frase tarda ~1-2 s y necesita internet; desde la
caché en disco sale al instante y también sin red. `python -m visionx.voice precalentar` genera
estas frases de antemano (la carpeta `data/cache/voz` se puede copiar a la Mac de la demo).

Las frases salen de las mismas funciones que usan los modos (direcciones, metros, nombres del
catálogo), para que la caché coincida carácter por carácter con lo que se va a decir.
"""

from __future__ import annotations

from typing import Any

from visionx.modes.common import Measure, ModeSettings, measure_phrase

# Objetos de una casa que más se buscan (clases COCO); el resto se genera al primer uso.
COMMON = (
    "person", "cup", "bottle", "chair", "couch", "bed", "dining table", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock", "backpack", "handbag", "umbrella",
    "potted plant", "sink", "refrigerator", "oven", "microwave", "toilet", "bowl", "spoon",
    "fork", "knife", "scissors", "toothbrush", "vase", "suitcase", "wine glass", "dog", "cat",
)  # fmt: skip

H_WORDS = ("al frente", "un poco a la izquierda", "un poco a la derecha", "a la izquierda",
           "a la derecha")  # fmt: skip
V_WORDS = ("", ", arriba", ", abajo")

FIXED = (
    "No entendí. Podés decir: buscá la taza, a qué distancia está, leé, describí o cancelar.",
    "No escuché nada.",
    "Cancelado.",
    "Ya no está vigente. Pedilo de nuevo.",
    "No hay nada para elegir ahora.",
    "La del medio.",
    "La más cercana.",
    "La de la izquierda.",
    "La de la derecha.",
    "¿Qué querés que busque?",
    "Leo.",
    "Describo.",
    "Mido la distancia.",
    "No encuentro texto.",
    "No pude leer. Probá de nuevo.",
    "No pude leer.",
    "No terminé de leer a tiempo.",
    "No pude describir a tiempo.",
    "Terminé de buscar.",
    "No lo encontré.",
    "Fin de búsqueda.",
    "Ahora no puedo responder preguntas: no tengo conexión.",
    "No pude responder ahora. Probá de nuevo.",
    "No tengo una imagen reciente para responderte.",
    "No veo nada al frente que pueda medir.",
)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def frequent_phrases(cfg: dict[str, Any], spoken) -> list[str]:
    s = ModeSettings.from_config(cfg)
    out: list[str] = list(FIXED)
    tts = (cfg.get("audio") or {}).get("tts") or {}
    out += [str(p) for p in tts.get("preload") or ()]
    status = cfg.get("camera_status") or {}
    out += [str(p) for p in (status.get("frases") or {}).values()]
    if status.get("lista"):
        out.append(str(status["lista"]))
    wheres = [h + v for h in H_WORDS for v in V_WORDS]
    out += [_cap(w) + "." for w in wheres]
    near, far = s.range_m
    zs = [round(near + 0.1 * i, 1) for i in range(int(round((far - near) / 0.1)) + 1)]
    metres = sorted({measure_phrase(Measure(z, None), s) for z in zs})
    metres += [measure_phrase(Measure(near - 0.1, "muy-cerca"), s),
               measure_phrase(Measure(far + 0.5, "muy-lejos"), s)]  # fmt: skip
    out += [_cap(m) + "." for m in metres]
    for h in H_WORDS[:3]:  # metros con el objeto al frente o «un poco» a un lado
        out += [f"{_cap(h)}, {m}." for m in metres]
    # Guía «centrar» (19/11): instrucciones, centrado y distancia hacia adelante.
    for side in ("izquierda", "derecha"):
        out += [f"Girá a la {side}.", f"Más a la {side}.", f"Un poco a la {side}."]
    out += ["Abajo.", "Arriba.", "Un poco más abajo.", "Un poco más arriba.", "Centrado."]
    out += [f"Centrado. Está {m}, adelante." for m in metres if m.startswith("a ")]
    out += [f"Está {m}, adelante." for m in metres if m.startswith("a ")]
    for cls in COMMON:
        name = spoken(cls)
        out += [f"Veo {name}.", f"Busco {name}.", f"Perdí {name}.", f"No encontré {name}.",
                f"Terminé de buscar {name}.", f"No veo {name} ahora. Sigo buscando."]  # fmt: skip
        out += [f"{_cap(name)} {w}." for w in wheres]
        # La búsqueda la termina la persona (19/11).
        out += [f"Sigo buscando {name}.", f"Listo, terminé de buscar {name}."]
        out += [
            f"Llegaste: {name} adelante, {m}. Cuando lo tengas, decí: listo."
            for m in ("a 0,5 metros", "a 0,6 metros", "muy cerca, a menos de 0,5 metros")
        ]
        out += [
            f"Llegaste: {name} adelante, {m}."
            for m in ("a 0,5 metros", "a 0,6 metros", "muy cerca, a menos de 0,5 metros")
        ]
    out += ["No estoy buscando nada ahora.", "Listo."]
    # «Hola VisionX», preferencias y alerta de obstáculos (19/11).
    from types import SimpleNamespace

    from visionx.app.onboarding import intro_phrases

    out += intro_phrases(SimpleNamespace(height_known=False, height_m=1.7), {})
    out += ["VisionX está listo. Decí: hola VisionX, y te cuento cómo funciono."]
    out += [f"Listo, hablo {w}. ¿Así está bien?" for w in ("más rápido", "más despacio",
                                                           "a velocidad normal")]  # fmt: skip
    out += [f"Listo, desde ahora mis respuestas son {w}." for w in ("más cortas", "normales",
                                                                   "más detalladas")]  # fmt: skip
    out += ["Ya estoy hablando lo más rápido que puedo.",
            "Ya estoy hablando lo más despacio que puedo."]  # fmt: skip
    out += [f"Cuidado: algo adelante, a {cm} centímetros." for cm in range(30, 100, 10)]
    heights = ("1,5", "1,55", "1,6", "1,65", "1,7", "1,75", "1,8", "1,85", "1,9")
    out += [f"Listo, medís {h} metros. Lo uso para decirte a qué altura están las cosas."
            for h in heights]  # fmt: skip
    return list(dict.fromkeys(out))  # sin repetidas, en orden
