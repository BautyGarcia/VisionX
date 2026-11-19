"""Asistente con Claude para leer, describir y responder preguntas sobre la imagen (T10/T11).

Decidido por Tomás el 18/11, después de comparar Opus 5.5, Sonnet 5.5 y Haiku 5.5 sobre 20 casos
de los clips (outputs/vlm/comparativa.html): Sonnet 5.5 con effort low empieza a hablar en ~0,9 s
(mediana) y no inventó contenido. Necesita internet y la key propia de VisionX (ver
`load_api_key`: NUNCA `ANTHROPIC_API_KEY`, que también usarían Claude Code y otros programas);
sin eso, cada modo
vuelve a su versión local (OCR + lectura por bloques, plantilla de describir).

- La respuesta llega por streaming y se entrega de a frases: la voz empieza con la primera.
- Texto citado: lo que va entre comillas tiene que estar en lo que leyó el OCR. Si no, «dice X»
  pasa a «parece decir X» (verify_quotes): una dosis o una fecha inventada no se afirma.
- Los metros solo salen de la cámara estéreo: el pedido le pasa a Claude los objetos medidos.
- Memoria corta: las preguntas de seguimiento («¿y qué más dice?») ven las últimas respuestas
  durante `memory_s` desde el último intercambio. Las imágenes viejas no se reenvían.
- Tiempo honesto en replay: cada frase guarda cuánto tardó en llegar (reloj real) y el modo la
  entrega recién cuando el reloj del modo avanzó eso desde el pedido (`Delivery`).
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SYSTEM_PROMPT = """Sos la voz de VisionX, un asistente para una persona ciega o con baja visión \
que lleva una cámara a la altura del pecho. Ves lo que la cámara tiene adelante en este momento.
- Respondé en español rioplatense (voseo), en frases cortas y naturales para decir en voz alta: \
sin listas, sin markdown, sin emojis. Máximo 3 frases, salvo que te pidan leer todo.
- Hablale a la persona: «Adelante tenés…», «A tu izquierda hay…». No menciones la cámara, el \
OCR ni el sistema: hablá como si vos vieras.
- Distancias: si te paso objetos medidos con metros, usá esos metros tal cual («a 1,4 metros»). \
Nunca estimes otros números de distancia; podés decir «cerca» o «más lejos». Esa lista es una \
ayuda: si un objeto no se ve en la imagen, ignoralo; si su nombre está mal (un lavarropas \
anotado como horno), usá el nombre correcto con sus metros. Nunca menciones la lista, la \
detección ni la cámara.
- Si no estás seguro de algo, decilo («parece…»). Nunca presentes como visto o escrito lo que \
no se ve o no se lee (tipo de producto, sabor, dosis, cantidad).
- Si te preguntan algo que no está escrito (para qué sirve un ingrediente, si algo es dulce), \
podés responder con conocimiento general, aclarando que es en general y no lo que dice el \
envase («la ashwagandha en general se usa para…»). Nunca des dosis que no se lean.
- Las manos que aparecen sosteniendo algo son de la persona que te habla: no digas «alguien».
- Texto: si te paso lo que leyó el OCR, entre comillas citá SOLO texto que esté en esa lista \
(podés corregir mayúsculas o tildes). Si ves texto que el OCR no tiene, decí «parece decir …» \
sin comillas. Si la letra no se lee, decilo y sugerí acercar el objeto.
- En medicamentos o suplementos priorizá nombre, para qué es y dosis o cantidad si se leen.
- El pedido viene de reconocimiento de voz y puede tener errores: si no tiene sentido, respondé \
lo más útil según la imagen o pedí que lo repita."""

KINDS = {
    "leer": "LEER: decí qué es el objeto que tiene adelante y lo más importante que dice.",
    "leer-todo": "LEER TODO: leé todo el texto que se pueda, en orden, sin resumir.",
    "describir": "DESCRIBIR: 3 frases como máximo y de no más de 15 palabras cada una, lo más "
    "cercano o importante primero (obstáculos y lo que se puede usar), sin detalles decorativos.",
    "pregunta": "PREGUNTA: respondé la pregunta con lo que ves.",
    "apoyo": "APOYO: el objeto pedido está en el centro de la imagen. En UNA frase corta (máximo "
    "12 palabras) decí sobre qué está apoyado y, si ayuda a encontrarlo, qué tiene al lado. "
    "Empezá con «Está…». Sin metros ni direcciones.",
    "opciones": "OPCIONES: hay varios objetos iguales, numerados de izquierda a derecha con su "
    "posición. Respondé SOLO estas líneas: una por objeto con el formato «N: descripción», donde "
    "la descripción son 1 a 3 palabras que lo distinguen de los otros (color, tamaño, sobre qué "
    "está), y al final una línea «PREGUNTA: …» que empiece con «Veo», los nombre por cómo son y "
    "de qué lado están y termine en «¿Cuál querés?» (p. ej. «Veo una taza blanca a la izquierda "
    "y un vaso rojo a la derecha. ¿Cuál querés?»). Sin metros.",
    "distancia": "DISTANCIA: con un objeto, UNA sola frase: qué es (color, sobre qué está o al "
    "lado de qué) y, tal cual, la distancia, el costado y la altura medidos de la lista. Con "
    "varios: primero cuántos son («Hay dos tazas.») y después una frase corta por cada uno, de "
    "izquierda a derecha, con lo que los distingue y su distancia; el costado y la altura solo si "
    "ayudan a encontrarlo. Sin números que no estén en la lista.",
}


@dataclass(frozen=True, slots=True)
class AssistantSettings:
    """Sección `assistant` de la config."""

    enabled: bool = True
    model: str = "claude-sonnet-5-5"
    effort: str = "low"
    max_tokens: int = 1024
    image_width: int = 1280
    jpeg_quality: int = 85
    timeout_s: float = 15.0
    memory_s: float = 120.0
    max_turns: int = 6
    # Cortar también en comas desde esta cantidad de caracteres, para que la voz arranque con el
    # primer tramo y no al final de una frase larga. 0 = solo en punto.
    clause_chars: int = 30
    fallbacks: bool = True  # fallback del servidor si el modelo rechaza (beta)
    # Gasto: cada consulta se anota en `ledger` (JSONL); con el acumulado >= budget_usd no se
    # consulta más y los modos usan su respaldo local. 0 = sin tope.
    ledger: str = "outputs/vlm/gasto.jsonl"
    budget_usd: float = 0.0
    # Key SOLO de VisionX: archivo privado fuera del repo. No se usa ANTHROPIC_API_KEY.
    key_file: str = "~/.visionx/anthropic_api_key"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> AssistantSettings:
        a = dict(cfg.get("assistant") or {})
        d = cls()
        return cls(**{k: type(getattr(d, k))(v) for k, v in a.items() if hasattr(d, k)})


# USD por millón de tokens (entrada, salida), precios publicados al 18/11/2026.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


class Ledger:
    """Costo de cada consulta en un JSONL; suma lo ya anotado (también de corridas previas)."""

    def __init__(self, path: str | None, budget_usd: float = 0.0) -> None:
        from visionx.shared.config import ROOT

        self.path = None if not path else (Path(path) if Path(path).is_absolute() else ROOT / path)
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self._spent = 0.0
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self._spent += float(json.loads(line).get("usd", 0.0))
                except (ValueError, AttributeError):
                    continue

    @property
    def spent(self) -> float:
        return self._spent

    def exhausted(self) -> bool:
        return self.budget_usd > 0 and self._spent >= self.budget_usd

    def add(self, model: str, tokens_in: int, tokens_out: int, kind: str) -> float:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        usd = (tokens_in * pin + tokens_out * pout) / 1e6
        with self._lock:
            self._spent += usd
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "modelo": model, "tipo": kind,
                       "in": tokens_in, "out": tokens_out, "usd": round(usd, 6)}  # fmt: skip
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        return usd


KEY_ENV = "VISIONX_ANTHROPIC_API_KEY"
KEY_FILE = "~/.visionx/anthropic_api_key"


def load_api_key(key_file: str = KEY_FILE) -> str | None:
    """La key de VisionX: `VISIONX_ANTHROPIC_API_KEY` o el archivo `key_file`.

    Nunca `ANTHROPIC_API_KEY`: ese nombre lo leen solos Claude Code y los SDK de Anthropic, y
    con la key de VisionX ahí, cualquiera de ellos gasta sus créditos (pasó el 19/11: la CLI se
    cobró 5 USD). Con un nombre y un archivo propios, solo el asistente de VisionX la usa.
    """
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    path = Path(key_file).expanduser()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def has_credentials(key_file: str = KEY_FILE) -> bool:
    return load_api_key(key_file) is not None


# --- texto -------------------------------------------------------------------------------


def _norm_words(text: str) -> list[str]:
    s = unicodedata.normalize("NFKD", text.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s).split()


def _close(a: str, b: str) -> bool:
    """Igual, o a una letra de distancia en palabras de 4+ letras (el OCR lee «CRASAS»)."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= 1


def supported(quote: str, ocr_words: list[str]) -> bool:
    words = _norm_words(quote)
    return bool(words) and all(any(_close(w, o) for o in ocr_words) for w in words)


_QUOTE = re.compile(r"«([^»]+)»|\"([^\"]+)\"|“([^”]+)”")
_SAYS = re.compile(r"(?:que\s+)?\b(dice|dicen)\s+$", re.IGNORECASE)


def verify_quotes(text: str, ocr_lines: Iterable[str]) -> tuple[str, int]:
    """Citas que el OCR no confirma: «dice "X"» -> «parece decir X»; otras pierden las comillas.

    Devuelve (texto, citas no confirmadas).
    """
    ocr_words = [w for line in ocr_lines for w in _norm_words(line)]
    out, last, unconfirmed = [], 0, 0
    for m in _QUOTE.finditer(text):
        quote = next(g for g in m.groups() if g is not None)
        before = text[last : m.start()]
        if supported(quote, ocr_words):
            out.append(before + m.group(0))
        else:
            unconfirmed += 1
            says = _SAYS.search(before)
            if says:
                verb = "parecen decir" if says.group(1).lower() == "dicen" else "parece decir"
                if not before[: says.start()].strip() and says.group(0)[:1].isupper():
                    verb = verb[:1].upper() + verb[1:]  # «Dice "X"» al principio de la frase
                prefix = before[: says.start()] + (
                    "que " if says.group(0).lower().startswith("que") else ""
                )
                out.append(prefix + f"{verb} {quote}")
            else:
                out.append(before + quote)
        last = m.end()
    out.append(text[last:])
    return "".join(out), unconfirmed


_END = re.compile(r"[.!?…](?=\s)")


_CLAUSE = re.compile(r"[,;:](?=\s)")


def _inside_quote(text: str, pos: int) -> bool:
    """¿La posición cae dentro de una cita «…» o "…"? (no se corta una cita a la mitad)."""
    before = text[:pos]
    return before.count("«") > before.count("»") or before.count('"') % 2 == 1


def split_sentences(
    buffer: str, *, final: bool = False, min_chars: int = 12, clause_chars: int = 0
) -> tuple[list[str], str]:
    """Frases completas del buffer y el resto. Con final=True, el resto también sale.

    Con `clause_chars` > 0 también corta en coma, punto y coma o dos puntos cuando el tramo ya
    tiene al menos esa cantidad de caracteres: una frase larga de Claude («La taza blanca está
    sobre el taburete, a 1,4 metros, …») empieza a sintetizarse y a sonar con su primer tramo en
    vez de esperar el punto final (en vivo, 19/11, la frase entera tardaba ~4 s en sonar).
    """
    out, start = [], 0
    cuts = sorted(
        [(m.end(), True) for m in _END.finditer(buffer)]
        + ([(m.end(), False) for m in _CLAUSE.finditer(buffer)] if clause_chars > 0 else [])
    )
    for end, full_stop in cuts:
        if end <= start:
            continue
        if full_stop and end - start >= min_chars:
            out.append(buffer[start:end].strip())
            start = end
        elif not full_stop and end - start >= clause_chars and not _inside_quote(buffer, end):
            out.append(buffer[start:end].strip())
            start = end
    rest = buffer[start:]
    if final:
        if rest.strip():
            out.append(rest.strip())
        rest = ""
    return out, rest


_OPTION = re.compile(r"(?m)^\s*(\d+)\s*[:.)-]\s*(.+?)\s*$")
_QUESTION = re.compile(r"(?mi)^\s*PREGUNTA\s*:\s*(.+?)\s*$")


def parse_options(raw: str, n: int) -> tuple[dict[int, str], str | None]:
    """«1: verde / 2: transparente / PREGUNTA: …» -> ({0: "verde", 1: "transparente"}, pregunta).

    Índices desde 0, en el orden de izquierda a derecha con que se numeraron. Si no están
    todas las opciones, devuelve {} y la pregunta se arma sin descripciones.
    """
    found = {}
    for m in _OPTION.finditer(raw):
        i = int(m.group(1)) - 1
        if 0 <= i < n and not m.group(2).upper().startswith("PREGUNTA"):
            found[i] = m.group(2).strip().rstrip(".")
    q = _QUESTION.search(raw)
    question = q.group(1).strip() if q else None
    return (found if len(found) == n else {}), question


def jpeg_b64(image: np.ndarray, width: int, quality: int) -> str:
    import cv2

    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("no se pudo codificar la imagen")
    return base64.standard_b64encode(buf.tobytes()).decode()


def user_text(kind: str, request: str | None, ocr: list[str] | None, objects: list[str]) -> str:
    parts = [f"Pedido — {KINDS[kind]}"]
    if request:
        parts.append(f"La persona dijo: «{request}»")
    if objects:
        parts.append("Objetos que detectó la cámara (los metros son medidos):\n" +
                     "\n".join(f"- {o}" for o in objects))  # fmt: skip
    if ocr is not None:
        parts.append(
            "Texto que leyó el OCR (puede tener errores):\n" + "\n".join(f"- {o}" for o in ocr)
            if ocr
            else "El OCR no leyó texto en esta imagen."
        )
    return "\n".join(parts)


# --- conversación ------------------------------------------------------------------------


class Conversation:
    """Últimos intercambios en texto (sin imágenes), olvidados `memory_s` después del último."""

    def __init__(self, memory_s: float, max_turns: int) -> None:
        self.memory_s, self.max_turns = memory_s, max_turns
        self._turns: list[tuple[str, str]] = []
        self._last = -1e18
        self._lock = threading.Lock()

    def history(self, now_s: float) -> list[dict[str, Any]]:
        with self._lock:
            if now_s - self._last > self.memory_s:
                self._turns = []
            messages: list[dict[str, Any]] = []
            for user, assistant in self._turns[-self.max_turns :]:
                messages += [{"role": "user", "content": user},
                             {"role": "assistant", "content": assistant}]  # fmt: skip
            return messages

    def add(self, user: str, assistant: str, now_s: float) -> None:
        with self._lock:
            self._turns.append((user, assistant))
            self._turns = self._turns[-self.max_turns :]
            self._last = now_s

    def clear(self) -> None:
        with self._lock:
            self._turns = []


# --- pedidos -----------------------------------------------------------------------------


@dataclass
class AssistantJob:
    request_id: int
    kind: str
    sentences: list[tuple[float, str]] = field(default_factory=list)  # (s desde el pedido, frase)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    text: str = ""
    first_s: float | None = None
    total_s: float | None = None
    unconfirmed_quotes: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    usd: float = 0.0
    ocr: list[str] | None = None
    raw: str = ""  # texto tal cual llegó (sin cortar en frases), para respuestas con formato

    def cancel(self) -> None:
        self.cancelled.set()


class Assistant:
    """Pide a Claude en un hilo propio; los modos leen las frases del AssistantJob."""

    def __init__(
        self,
        settings: AssistantSettings | None = None,
        *,
        client: Any = None,
        conversation: Conversation | None = None,
        ocr: Callable[[np.ndarray], list[str]] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        ledger: Ledger | None = None,
    ) -> None:
        self.s = settings or AssistantSettings()
        self.ledger = ledger or Ledger(self.s.ledger, self.s.budget_usd)
        self._client = client
        self.conversation = conversation or Conversation(self.s.memory_s, self.s.max_turns)
        self._ocr = ocr
        self._clock = clock
        self._lock = threading.Lock()
        self._current: AssistantJob | None = None

    @property
    def available(self) -> bool:
        if not self.s.enabled:
            return False
        if self.ledger.exhausted():
            return False
        if self._client is not None:
            return True
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return has_credentials(self.s.key_file)

    def client(self) -> Any:
        if self._client is None:
            import anthropic

            key = load_api_key(self.s.key_file)
            if key is None:
                raise RuntimeError("falta la key de VisionX (~/.visionx/anthropic_api_key)")
            # api_key explícita: el SDK no cae a ANTHROPIC_API_KEY ni a otras credenciales.
            self._client = anthropic.Anthropic(api_key=key, timeout=self.s.timeout_s, max_retries=1)
        return self._client

    def submit(
        self,
        request_id: int,
        kind: str,
        image: np.ndarray,
        *,
        request: str | None = None,
        ocr: list[str] | None = None,
        objects: Iterable[str] = (),
        read_text: bool = False,
    ) -> AssistantJob:
        """Lanza el pedido y devuelve enseguida. `read_text`: correr el OCR en el hilo del
        pedido (preguntas sobre texto) si no se pasa `ocr`. Un pedido nuevo cancela el anterior."""
        job = AssistantJob(request_id, kind, ocr=ocr)
        with self._lock:
            if self._current is not None:
                self._current.cancel()
            self._current = job
        args = (job, image.copy(), request, list(objects), read_text)
        threading.Thread(target=self._run, args=args, daemon=True, name="asistente").start()
        return job

    def cancel(self) -> None:
        with self._lock:
            if self._current is not None:
                self._current.cancel()

    def _run(self, job: AssistantJob, image, request, objects, read_text) -> None:
        t0 = self._clock()
        try:
            if self.ledger.exhausted():
                raise RuntimeError(f"tope de gasto alcanzado ({self.ledger.spent:.2f} USD)")
            if job.ocr is None and read_text and self._ocr is not None:
                job.ocr = self._ocr(image)
            text = user_text(job.kind, request, job.ocr, objects)
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                 "data": jpeg_b64(image, self.s.image_width, self.s.jpeg_quality)}},
                {"type": "text", "text": text},
            ]  # fmt: skip
            messages = self.conversation.history(t0) + [{"role": "user", "content": content}]
            buffer, full = "", []
            for delta in self._stream(messages, job):
                if job.cancelled.is_set():
                    return
                if job.first_s is None:
                    job.first_s = self._clock() - t0
                buffer += delta
                job.raw += delta
                ready, buffer = split_sentences(buffer, clause_chars=self.s.clause_chars)
                for sentence in ready:
                    self._emit(job, sentence, t0, full)
            ready, _ = split_sentences(buffer, final=True)
            for sentence in ready:
                self._emit(job, sentence, t0, full)
            job.text = " ".join(full)
            job.total_s = self._clock() - t0
            if job.text and not job.cancelled.is_set():
                self.conversation.add(text, job.text, self._clock())
        except Exception as exc:  # red, timeout, key inválida, rechazo: el modo usa su respaldo
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.done.set()

    def _emit(self, job: AssistantJob, sentence: str, t0: float, full: list[str]) -> None:
        checked, bad = verify_quotes(sentence, job.ocr or [])
        job.unconfirmed_quotes += bad
        full.append(checked)
        job.sentences.append((self._clock() - t0, checked))

    def _stream(self, messages: list[dict[str, Any]], job: AssistantJob) -> Iterator[str]:
        client = self.client()
        kwargs: dict[str, Any] = dict(
            model=self.s.model,
            max_tokens=self.s.max_tokens,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.s.effort},
            messages=messages,
        )
        if self.s.fallbacks and not self.s.model.startswith("claude-haiku"):
            manager = client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            manager = client.messages.stream(**kwargs)
        with manager as stream:
            yield from stream.text_stream
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise RuntimeError("el modelo no respondió (refusal)")
        job.usage = {"in": message.usage.input_tokens, "out": message.usage.output_tokens}
        job.usd = self.ledger.add(self.s.model, job.usage["in"], job.usage["out"], job.kind)


class Delivery:
    """Entrega las frases de un AssistantJob en el reloj del modo (vivo o replay)."""

    def __init__(self, job: AssistantJob, start_ns: int) -> None:
        self.job, self.start_ns, self._next = job, start_ns, 0

    def ready(self, now_ns: int) -> list[str]:
        out = []
        sentences = self.job.sentences
        while self._next < len(sentences):
            elapsed, text = sentences[self._next]
            if self.start_ns + int(elapsed * 1e9) > now_ns:
                break
            out.append(text)
            self._next += 1
        return out

    @property
    def finished(self) -> bool:
        return self.job.done.is_set() and self._next >= len(self.job.sentences)

    @property
    def failed(self) -> bool:
        """Falló sin decir nada: el modo puede usar su respaldo local."""
        return self.job.done.is_set() and self.job.error is not None and self._next == 0


def build_assistant(cfg: dict[str, Any]) -> Assistant | None:
    """Asistente de la config, o None si está apagado, sin SDK o sin credenciales."""
    settings = AssistantSettings.from_config(cfg)
    reader_box: list[Any] = []

    def ocr(image: np.ndarray) -> list[str]:
        from visionx.reading import RapidReader, ReadingSettings, reading_order

        if not reader_box:
            reader_box.append((RapidReader(ReadingSettings.from_config(cfg)),
                               ReadingSettings.from_config(cfg)))  # fmt: skip
        reader, rs = reader_box[0]
        return [b.text for b in reading_order(reader.read(image), rs)]

    assistant = Assistant(settings, ocr=ocr)
    return assistant if assistant.available else None
