"""Reglas en español: de lo que dijo el usuario a una orden (T08).

Primera versión con reglas y alias, como decide TECNICO T08: son predecibles, no necesitan red
y cada decisión dice qué regla la tomó. Si no se identifica el modo o el objeto, se pide una
aclaración en vez de adivinar. Un LLM de intenciones queda como alternativa si las frases
reales muestran que las reglas no alcanzan.

El texto llega del STT (Whisper), que pone mayúsculas, tildes y signos: «¿A qué distancia está
la taza?». Todo se normaliza antes de comparar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from visionx.detection.catalog import normalize
from visionx.shared.contracts import Command, FrameKey, Mode

CANCEL = "cancelar"
REPEAT = "repetir"
READ_ALL = "todo"  # «leé todo»: lectura completa por bloques, sin resumir
CALIBRATE = "calibrar"  # «calibrá, está a 2 metros» -> objetivo "calibrar:2"
_CALIBRATE = re.compile(r"^(calibra\w*|calibrar)\b")
# Preguntas (con asistente): empiezan con interrogativo; «¿qué dice?» sola sigue siendo leer.
_QUESTION_START = re.compile(
    r"^(y )?(que|cual|cuales|cuando|cuanto|cuanta|cuantos|cuantas|como|donde|quien|para que|"
    r"hay|es|esta|tiene|sirve|me sirve)( |$)"
)
_READ_ONLY = re.compile(
    r"^(y )?(que (mas )?dice|que dicen|que pone|que esta escrito)"
    r"( (aca|esto|eso|ahi|(la|el|esta|este|lo) \w+))?$"
)
# Opciones de elección entre varios objetos iguales (las usa T09).
CHOICES = ("izquierda", "derecha", "centro", "cerca")


class ObjectResolver(Protocol):
    """Lo que el intérprete necesita del catálogo de clases (detection.ClassCatalog)."""

    def resolve(self, text: str) -> str | None: ...

    def spoken(self, cls: str) -> str: ...


@dataclass(frozen=True, slots=True)
class Interpretation:
    """Resultado de interpretar una frase.

    mode None: no se entendió la orden. reply no None: hay que decirle algo al usuario antes de
    actuar (aclaración). rule: qué regla decidió, para auditar errores en las pruebas de frases.
    """

    utterance: str
    mode: Mode | None
    target: str | None
    rule: str
    reply: str | None = None

    @property
    def ok(self) -> bool:
        """True si se puede ejecutar sin preguntar nada."""
        return self.mode is not None and self.reply is None

    def to_command(
        self,
        request_id: int,
        issued_ns: int,
        deadline_ns: int | None = None,
        frame: FrameKey | None = None,
    ) -> Command:
        if not self.ok:
            raise ValueError(f"la frase {self.utterance!r} necesita aclaración: {self.reply}")
        return Command(
            request_id=request_id,
            mode=self.mode,  # type: ignore[arg-type]
            utterance=self.utterance,
            issued_ns=issued_ns,
            target=self.target,
            deadline_ns=deadline_ns,
            frame=frame,
        )


# Palabras que se dicen antes de la orden y no cambian su sentido.
_FILLERS = ("por favor", "visionx", "vision x", "vision", "che", "ey", "hola", "dale", "bueno")
_ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas", "mi", "mis", "lo", "al", "del"}

_CANCEL_WORDS = {
    "cancela", "cancelar", "cancelalo", "para", "parar", "basta", "stop", "silencio",
    "callate", "calla", "detene", "detenete", "frena", "frenar", "deja", "dejalo", "chau",
}  # fmt: skip
_REPEAT = re.compile(r"^(repeti\w*|repetir|otra vez|de nuevo|que dijiste|como dijiste)\b")
_DISTANCE = re.compile(
    r"\b(distancia|que tan lejos|cuan lejos|a cuanto (esta|estoy|queda)|cuantos metros|"
    r"que tan cerca|lejos (esta|queda|estoy))\b"
)
_READ = re.compile(r"^(lee\w*|leer|lei)\b|\b(que dice|que pone|que esta escrito)\b")
# "descri\w*" también cubre transcripciones con una sílaba comida ("descríme"); "cuentame" es la
# forma neutra que Whisper suele escribir por "contame" (bucle con Piper, BITACORA 2026-11-17).
_DESCRIBE = re.compile(r"^(descri\w*|contame|cuentame)\b|^que (hay|ves|tengo|veo)\b")
_SEARCH = re.compile(
    # Guiar empieza buscando (TECNICO T09): "llevame a la silla" es una búsqueda con guía.
    # "encontr\w*" y "encuentr\w*" aceptan "encontré" y "encuéntrame", que Whisper escribe a veces.
    r"^(busca\w*|buscar|encontr\w*|encuentr\w*|ubica\w*|ubicar|hay|guia\w*|llevame|acercame)\b|"
    r"^donde (esta|estan|hay|quedo|dejo|deje)\b"
)
_CHOOSE = {
    "izquierda": "izquierda",
    "derecha": "derecha",
    "medio": "centro",
    "centro": "centro",
    "cerca": "cerca",
    "cercana": "cerca",
    "cercano": "cerca",
}
# Palabras de orden que no forman parte del nombre del objeto pedido.
_COMMAND_WORDS = {
    "busca", "buscame", "buscar", "buscalo", "buscala", "encontra", "encontrame", "encontrar",
    "encontre", "encuentra", "encuentrame",
    "ubica", "ubicame", "ubicar", "hay", "donde", "esta", "estan", "quedo", "dejo", "deje",
    "a", "que", "distancia", "tan", "lejos", "cerca", "cuan", "cuanto", "cuantos", "metros",
    "queda", "estoy", "de", "hasta", "me", "se", "en", "algun", "alguna", "aca", "guiame",
    "guiar", "llevame", "acercame", "para", "con",
}  # fmt: skip


def clean(text: str) -> str:
    """Normaliza y quita signos y muletillas: «¿Che, buscá la taza?» -> "busca la taza"."""
    text = normalize(re.sub(r"[¿?¡!.,;:\"'«»()\-]", " ", text))
    changed = True
    while changed:
        changed = False
        for filler in _FILLERS:
            if text == filler:
                return ""
            if text.startswith(filler + " "):
                text, changed = text[len(filler) + 1 :], True
            if text.endswith(" " + filler):
                text, changed = text[: -len(filler) - 1], True
    return text


def _object_words(text: str) -> list[str]:
    return [w for w in text.split() if w not in _COMMAND_WORDS and w not in _ARTICLES]


def find_object(text: str, catalog: ObjectResolver) -> str | None:
    """Clase del objeto mencionado en la frase, probando primero los nombres más largos.

    "busca el control remoto" prueba "control remoto" antes que "control": así una palabra de
    un nombre compuesto no gana por llegar primero.
    """
    words = _object_words(text)
    for size in (3, 2, 1):
        for start in range(len(words) - size + 1):
            cls = catalog.resolve(" ".join(words[start : start + size]))
            if cls is not None:
                return cls
    return None


class CommandParser:
    """Interpreta frases en español con reglas fijas. No tiene estado: la misma frase da
    siempre la misma interpretación."""

    def __init__(self, catalog: ObjectResolver, *, questions: bool = False) -> None:
        self.catalog = catalog
        # Con el asistente (Claude) disponible, lo que no es una orden es una pregunta libre
        # sobre lo que la cámara tiene adelante: Mode.DESCRIBE con la frase como objetivo.
        self.questions = questions

    def parse(self, utterance: str) -> Interpretation:
        text = clean(utterance)
        words = text.split()
        if not words:
            return Interpretation(utterance, None, None, "vacio", "No escuché nada.")

        # Cancelar va primero: tiene que funcionar aunque la frase siga ("pará, pará").
        # «¿Para qué sirve?» es una pregunta, no «pará» (solo con el asistente).
        asks_why = self.questions and words[:2] == ["para", "que"] and len(words) >= 3
        if words[0] in _CANCEL_WORDS and not asks_why:
            return Interpretation(utterance, Mode.CONTROL, CANCEL, "cancelar")
        if _REPEAT.search(text):
            return Interpretation(utterance, Mode.CONTROL, REPEAT, "repetir")
        if _CALIBRATE.search(text):
            from visionx.app.calibrate import parse_meters

            z_m = parse_meters(utterance)  # sin limpiar: «1,5» pierde la coma al normalizar
            if z_m is None:
                reply = "Decime a cuántos metros está. Por ejemplo: calibrá, está a 2 metros."
                return Interpretation(
                    utterance, Mode.CONTROL, None, "calibrar-sin-distancia", reply
                )
            return Interpretation(utterance, Mode.CONTROL, f"{CALIBRATE}:{z_m:g}", "calibrar")

        if _DISTANCE.search(text):
            cls = find_object(text, self.catalog)
            # Pronombres solos remiten al objetivo reciente; un nombre desconocido sigue
            # pidiendo aclaración ("esa puerta" no debe terminar midiendo otra cosa).
            object_words = _object_words(text)
            implicit = {"eso", "ese", "esa", "esto", "este", "esta", "objeto"}
            wanted = " ".join(object_words) if set(object_words) - implicit else ""
            if cls is None and wanted:
                # Nombró algo que no conocemos ("la puerta"): medir el centro en su lugar
                # sería responder otra pregunta sin avisar.
                reply = f"No reconozco {wanted}. Puedo medir lo que tenés en el centro."
                return Interpretation(utterance, Mode.DISTANCE, None, "distancia-sin-objeto", reply)
            # Sin objeto es válido: la distancia del objeto que se está guiando o del centro.
            return Interpretation(utterance, Mode.DISTANCE, cls, "distancia")
        if _READ.search(text):
            if self.questions and "todo" in words:
                return Interpretation(utterance, Mode.READ, READ_ALL, "leer-todo")
            if self.questions and _QUESTION_START.match(text) and not _READ_ONLY.search(text):
                # «¿Cuándo vence lo que dice acá?»: pregunta sobre el texto, no «leé».
                return Interpretation(utterance, Mode.DESCRIBE, utterance.strip(), "pregunta")
            return Interpretation(utterance, Mode.READ, None, "leer")
        if _DESCRIBE.search(text):
            return Interpretation(utterance, Mode.DESCRIBE, None, "describir")
        if _SEARCH.search(text):
            cls = find_object(text, self.catalog)
            if cls is not None:
                return Interpretation(utterance, Mode.SEARCH, cls, "buscar")
            if self.questions and _QUESTION_START.match(text):
                # «¿Hay algún lugar para sentarme?»: sin objeto del catálogo es una pregunta.
                return Interpretation(utterance, Mode.DESCRIBE, utterance.strip(), "pregunta")
            wanted = " ".join(_object_words(text))
            reply = (
                f"No sé buscar {wanted}. Puedo buscar objetos como taza, botella o celular."
                if wanted
                else "¿Qué querés que busque?"
            )
            return Interpretation(utterance, Mode.SEARCH, None, "buscar-sin-objeto", reply)

        choice = _choice(words)
        if choice is not None:
            return Interpretation(utterance, Mode.CHOOSE, choice, "elegir")

        # Solo el nombre de un objeto ("¿la taza?") se toma como búsqueda.
        cls = self.catalog.resolve(text)
        if cls is not None:
            return Interpretation(utterance, Mode.SEARCH, cls, "solo-objeto")
        if self.questions and len(words) >= 2:
            return Interpretation(utterance, Mode.DESCRIBE, utterance.strip(), "pregunta")
        return Interpretation(
            utterance,
            None,
            None,
            "sin-regla",
            "No entendí. Podés decir: buscá la taza, a qué distancia está, "
            "leé, describí o cancelar.",
        )


def _choice(words: list[str]) -> str | None:
    """ "la de la izquierda", "el del medio", "la más cercana" -> opción; frases cortas solamente
    para no confundir "la mesa de la izquierda tiene..." con una elección."""
    if len(words) > 5:
        return None
    for word in reversed(words):
        if word in _CHOOSE:
            return _CHOOSE[word]
    return None


def acknowledgement(interp: Interpretation, catalog: ObjectResolver) -> str | None:
    """Frase corta que confirma qué se entendió antes de ejecutar. None si no hace falta."""
    if not interp.ok:
        return interp.reply
    name = catalog.spoken(interp.target) if interp.target else None
    if interp.rule == "pregunta":
        return "sonido:ok"  # la respuesta ya arranca en ~1 s: no hacerla esperar con palabras
    match interp.mode:
        case Mode.SEARCH:
            return f"Busco {name}."
        case Mode.DISTANCE:
            return f"Mido la distancia a {name}." if name else "Mido la distancia."
        case Mode.READ:
            return "Leo."
        case Mode.DESCRIBE:
            return "Describo."
        case Mode.CHOOSE:
            return {"centro": "La del medio.", "cerca": "La más cercana."}.get(
                interp.target or "", f"La de la {interp.target}."
            )
    return None  # cancelar/repetir no se anuncian: se hacen
