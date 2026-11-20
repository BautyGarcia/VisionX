"""«Hola VisionX»: la presentación interactiva y lo que la persona puede configurar por voz.

Al decir «Hola VisionX» (o «ayuda», «¿qué podés hacer?») VisionX cuenta en frases cortas qué
hace y cómo se le habla, qué se puede configurar, y pide la estatura si no la sabe (la respuesta
puede ser un número suelto: «1,75»). Lo configurable queda guardado para esa persona en
`~/.visionx/persona.json`:

- **Velocidad de la voz:** «hablá más rápido», «más despacio», «velocidad normal».
- **Largo de las respuestas:** «respuestas más cortas», «más detalladas», «respuestas normales».
- **Estatura:** «mido 1,80».
"""

from __future__ import annotations

from typing import Any

# Velocidades de la voz (Edge): 1,0 = normal. La de la config es el punto de partida.
SPEEDS = (0.9, 1.0, 1.15, 1.3, 1.45)
DETAILS = ("corto", "normal", "largo")


def intro_phrases(body: Any, prefs: dict) -> list[str]:
    """La presentación, en frases cortas (la voz las dice por tramos, sin esperar al final)."""
    out = [
        "Hola, soy VisionX. Te ayudo a encontrar cosas y a moverte con la cámara que llevás "
        "en el pecho.",
        "Para hablarme, mantené apretado el botón mientras hablás, y soltalo al terminar.",
        "Podés decirme: buscá la taza. ¿A qué distancia está la silla? Leé esto. O describí lo "
        "que hay adelante. También podés preguntarme lo que quieras de lo que veo.",
        "Mientras te guío hacia algo, te aviso si hay otra cosa en el camino.",
        "Si querés, hablo más rápido o más despacio, y doy respuestas más cortas o más "
        "detalladas. Pedímelo cuando quieras.",
    ]
    if body.height_known:
        h = f"{body.height_m:.2f}".rstrip("0").rstrip(".").replace(".", ",")
        out.append(f"Tengo guardado que medís {h} metros. Si no es así, decime cuánto medís.")
    else:
        out.append("Para decirte a qué altura están las cosas, ¿cuánto medís?")
    return out


def apply_preference(change: str, synth: Any, assistant: Any, base_speed: float) -> list[str]:
    """Aplica "voz:rapido|lento|normal" o "detalle:corto|largo|normal", lo guarda y lo confirma."""
    from visionx.modes import body
    from visionx.voice.engines import set_speech_rate

    kind, value = change.split(":", 1)
    prefs = body.preferences()
    if kind == "voz":
        current = float(prefs.get("velocidad", base_speed))
        i = min(range(len(SPEEDS)), key=lambda k: abs(SPEEDS[k] - current))
        if value == "normal":
            new = base_speed
        elif value == "rapido":
            if i == len(SPEEDS) - 1:
                return ["Ya estoy hablando lo más rápido que puedo."]
            new = SPEEDS[i + 1]
        else:
            if i == 0:
                return ["Ya estoy hablando lo más despacio que puedo."]
            new = SPEEDS[i - 1]
        if synth is None or not set_speech_rate(synth, new):
            return ["Con esta voz no puedo cambiar la velocidad."]
        body.save_preference(velocidad=new)
        word = {"rapido": "más rápido", "lento": "más despacio", "normal": "a velocidad normal"}
        return [f"Listo, hablo {word[value]}. ¿Así está bien?"]
    if kind == "detalle":
        if assistant is None:
            return ["El largo de las respuestas se cambia con el asistente, y ahora no está."]
        current = prefs.get("detalle", "normal")
        if value == "normal":
            new = "normal"
        else:
            i = DETAILS.index(current) if current in DETAILS else 1
            step = -1 if value == "corto" else 1
            new = DETAILS[max(0, min(len(DETAILS) - 1, i + step))]
            if new == current:
                edge = "lo más cortas" if value == "corto" else "lo más detalladas"
                return [f"Ya te respondo {edge} que puedo."]
        assistant.detail = new
        body.save_preference(detalle=new)
        word = {"corto": "más cortas", "normal": "normales", "largo": "más detalladas"}
        return [f"Listo, desde ahora mis respuestas son {word[new]}."]
    return ["No entendí qué querés cambiar."]


def restore(synth: Any, assistant: Any) -> dict:
    """Al arrancar: la velocidad y el largo que la persona eligió la última vez."""
    from visionx.modes import body
    from visionx.voice.engines import set_speech_rate

    prefs = body.preferences()
    if "velocidad" in prefs and synth is not None:
        set_speech_rate(synth, float(prefs["velocidad"]))
    if "detalle" in prefs and assistant is not None:
        assistant.detail = str(prefs["detalle"])
    return prefs
