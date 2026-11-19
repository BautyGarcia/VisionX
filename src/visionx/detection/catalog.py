"""Catálogo de clases del detector y sus nombres en español (T06).

Los pesos COCO de YOLO conocen 80 clases con nombre en inglés. El usuario habla en español:
este módulo traduce lo que dice ("buscá la taza") a la clase del modelo ("cup") y la clase
detectada a la palabra que se pronuncia. Un nombre que no está acá no se puede buscar con un
detector de vocabulario fijo: pasarle "llaves" no le enseña a encontrar llaves (TECNICO T06).

Que una clase figure en el catálogo **no** prueba que se detecte bien con la ELP. Cuáles
quedan aceptadas se decide con clips anotados y se registra en BITACORA.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from itertools import product

# Clase COCO -> nombres en español. El primero es el que se dice en voz alta; los demás son
# sinónimos o variantes regionales que el usuario puede usar al pedirla. Orden = índice COCO.
COCO_ES: dict[str, tuple[str, ...]] = {
    "person": ("persona", "gente", "hombre", "mujer", "chico", "chica"),
    "bicycle": ("bicicleta", "bici"),
    "car": ("auto", "coche", "carro"),
    "motorcycle": ("moto", "motocicleta"),
    "airplane": ("avión",),
    "bus": ("colectivo", "autobús", "bondi", "micro"),
    "train": ("tren",),
    "truck": ("camión", "camioneta"),
    "boat": ("barco", "bote", "lancha"),
    "traffic light": ("semáforo",),
    "fire hydrant": ("hidrante", "boca de incendio"),
    "stop sign": ("cartel de pare", "pare"),
    "parking meter": ("parquímetro",),
    "bench": ("banco", "banca"),
    "bird": ("pájaro", "ave"),
    "cat": ("gato", "gata"),
    "dog": ("perro", "perra"),
    "horse": ("caballo",),
    "sheep": ("oveja",),
    "cow": ("vaca",),
    "elephant": ("elefante",),
    "bear": ("oso",),
    "zebra": ("cebra",),
    "giraffe": ("jirafa",),
    "backpack": ("mochila",),
    "umbrella": ("paraguas",),
    "handbag": ("cartera", "bolso"),
    "tie": ("corbata",),
    "suitcase": ("valija", "maleta"),
    "frisbee": ("frisbee", "disco volador"),
    "skis": ("esquís",),
    "snowboard": ("snowboard", "tabla de nieve"),
    "sports ball": ("pelota", "balón"),
    "kite": ("barrilete", "cometa"),
    "baseball bat": ("bate",),
    "baseball glove": ("guante de béisbol",),
    "skateboard": ("patineta", "skate"),
    "surfboard": ("tabla de surf",),
    "tennis racket": ("raqueta",),
    "bottle": ("botella",),
    "wine glass": ("copa",),
    "cup": ("taza", "vaso", "pocillo"),
    "fork": ("tenedor",),
    "knife": ("cuchillo",),
    "spoon": ("cuchara",),
    "bowl": ("bol", "cuenco", "tazón"),
    "banana": ("banana", "plátano"),
    "apple": ("manzana",),
    "sandwich": ("sándwich", "sanguche"),
    "orange": ("naranja",),
    "broccoli": ("brócoli",),
    "carrot": ("zanahoria",),
    "hot dog": ("pancho", "hot dog"),
    "pizza": ("pizza",),
    "donut": ("dona", "rosquilla"),
    "cake": ("torta", "pastel"),
    "chair": ("silla",),
    "couch": ("sillón", "sofá"),
    "potted plant": ("planta", "maceta"),
    "bed": ("cama",),
    "dining table": ("mesa",),
    "toilet": ("inodoro", "baño"),
    "tv": ("televisor", "tele", "televisión"),
    "laptop": ("computadora portátil", "notebook", "laptop", "computadora"),
    "mouse": ("mouse", "ratón", "maus", "mause"),  # Whisper escribe «maus» (vivo 19/11)
    "remote": ("control remoto", "control"),
    "keyboard": ("teclado",),
    "cell phone": ("celular", "teléfono", "móvil"),
    "microwave": ("microondas",),
    "oven": ("horno",),
    "toaster": ("tostadora",),
    "sink": ("pileta", "bacha", "lavabo", "fregadero"),
    "refrigerator": ("heladera", "refrigerador", "nevera"),
    "book": ("libro",),
    "clock": ("reloj",),
    "vase": ("florero", "jarrón"),
    "scissors": ("tijera", "tijeras"),
    "teddy bear": ("peluche", "osito de peluche"),
    "hair drier": ("secador de pelo", "secador"),
    "toothbrush": ("cepillo de dientes",),
}


def normalize(text: str) -> str:
    """Minúsculas, sin tildes ni espacios repetidos: "  Teléfono " -> "telefono"."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(plain.split())


def _singulars(word: str) -> tuple[str, ...]:
    """Singulares posibles de una palabra española, solo para comparar con el catálogo.

    No hay una regla única: "bates" -> "bate" pero "relojes" -> "reloj". Se prueban ambas.
    """
    options = [word]
    if word.endswith("s") and len(word) > 3:
        options.append(word[:-1])  # tazas -> taza, bates -> bate
    if word.endswith("es") and len(word) > 4:
        options.append(word[:-2])  # relojes -> reloj, camiones -> camion
    return tuple(options)


class ClassCatalog:
    """Traduce entre los nombres del usuario y las clases que entrega un modelo.

    `model_classes` son los nombres que trae el modelo cargado (result.names). Solo se resuelven
    pedidos a clases que el modelo realmente tiene; los alias de clases ausentes se ignoran.
    """

    def __init__(
        self,
        model_classes: Iterable[str],
        aliases: Mapping[str, Iterable[str]] = COCO_ES,
    ) -> None:
        self.model_classes = tuple(model_classes)
        known = set(self.model_classes)
        self._spoken: dict[str, str] = {}
        self._lookup: dict[str, str] = {}
        for cls in self.model_classes:
            # El nombre propio del modelo también sirve para pedirla ("cup").
            self._add(cls, cls)
        for cls, names in aliases.items():
            if cls not in known:
                continue
            names = tuple(names)
            if names:
                self._spoken[cls] = names[0]
            for name in names:
                self._add(name, cls)

    def _add(self, name: str, cls: str) -> None:
        key = normalize(name)
        previous = self._lookup.get(key)
        if previous is not None and previous != cls:
            # Un alias ambiguo haría que "buscá X" encuentre a veces una cosa y a veces otra.
            raise ValueError(f"alias {name!r} apunta a {previous!r} y a {cls!r}")
        self._lookup[key] = cls

    def resolve(self, text: str) -> str | None:
        """Clase del modelo para lo que dijo el usuario, o None si no la conoce.

        Prueba la frase completa y después en singular ("las tazas" -> "cup"). No busca
        palabras sueltas dentro de una frase larga: eso es trabajo de las reglas de voz (T08).
        """
        key = normalize(text)
        for article in ("el ", "la ", "los ", "las ", "un ", "una ", "unos ", "unas ", "mi "):
            if key.startswith(article):
                key = key[len(article) :]
                break
        if key in self._lookup:
            return self._lookup[key]
        # Cada palabra puede venir en plural ("controles remotos"): probar las combinaciones.
        # Son pocas: el catálogo tiene nombres de hasta tres palabras, y una frase más larga no
        # es un nombre de objeto (y multiplicaría las combinaciones).
        if len(key.split()) > 3:
            return None
        for words in product(*(_singulars(w) for w in key.split())):
            candidate = " ".join(words)
            if candidate in self._lookup:
                return self._lookup[candidate]
        return None

    def spoken(self, cls: str) -> str:
        """Palabra en español para una clase detectada; el nombre del modelo si no hay alias."""
        return self._spoken.get(cls, cls)

    def missing_aliases(self) -> tuple[str, ...]:
        """Clases del modelo sin nombre en español: no se pueden pedir por voz en castellano."""
        return tuple(c for c in self.model_classes if c not in self._spoken)


COCO_CLASSES: tuple[str, ...] = tuple(COCO_ES)
