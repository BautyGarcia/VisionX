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
    "cup": ("taza", "vaso", "pocillo", "tasa"),  # Whisper escribe «tasa» (vivo 19/11)
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
        aliases: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.model_classes = tuple(model_classes)
        if aliases is None:  # todos los conocidos: solo cuentan los de clases del modelo
            aliases = ALL_ES
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


# --- Vocabulario de VisionX para YOLOE (20/11) ---------------------------------------------------
# YOLOE es de vocabulario abierto: se le pasa la lista de clases como texto y sus pesos quedan
# armados con ella (`python -m visionx.detection vocabulario`). VisionX usa las COCO que tienen
# sentido en una casa o una facultad más EXTRA_ES. La clave en inglés es el nombre de la clase;
# el primer nombre en español es el que se dice.
# Reglas (las controla ClassCatalog y tests/test_vocabulario.py):
# - un nombre en español no puede ir a dos clases;
# - un nombre en inglés no puede coincidir con un alias español de otra clase (por eso «spiral
#   notebook» y «frying pan»: «notebook» es de laptop y «pan» es el pan);
# - nada de clases casi iguales («bottle» y «water bottle»): YOLOE repartiría la detección y
#   «buscá la botella» perdería las que salieron como la otra. Los sinónimos van en PROMPTS.
EXTRA_ES: dict[str, tuple[str, ...]] = {
    # --- lo que una persona lleva encima ---
    "keys": ("llaves", "llavero"),
    "wallet": ("billetera", "monedero"),
    "eyeglasses": ("anteojos", "lentes", "gafas", "anteojos de sol", "lentes de sol"),
    "wristwatch": ("reloj de pulsera", "reloj pulsera"),
    "headphones": ("auriculares", "audífonos"),
    "pen": ("lapicera", "birome", "bolígrafo", "lapicero"),
    "pencil": ("lápiz",),
    "marker pen": ("marcador", "fibrón", "resaltador"),
    "spiral notebook": ("cuaderno", "anotador", "libreta"),
    "phone charger": ("cargador",),
    "power bank": ("batería externa", "power bank"),
    "cable": ("cable",),
    "tablet": ("tablet", "tableta", "ipad"),
    "card": ("tarjeta", "credencial"),
    "thermos": ("termo",),
    "mate gourd": ("mate",),
    "lighter": ("encendedor",),
    "face mask": ("barbijo", "tapabocas"),
    "cap": ("gorra", "gorro"),
    "hat": ("sombrero",),
    "jacket": ("campera", "abrigo", "chaqueta", "buzo"),
    "scarf": ("bufanda",),
    "gloves": ("guantes",),
    "shoe": ("zapato", "zapatilla", "calzado"),
    "slippers": ("pantuflas", "ojotas"),
    "white cane": ("bastón", "bastón blanco"),
    "crutch": ("muleta",),
    "pill bottle": ("remedio", "remedios", "pastillas", "medicamento"),
    "tissue box": ("pañuelos", "carilinas", "pañuelitos"),
    "comb": ("peine",),
    "hairbrush": ("cepillo de pelo",),
    "deodorant": ("desodorante",),
    "perfume": ("perfume",),
    "coin": ("moneda", "monedas"),
    "banknote": ("billete", "plata"),
    "camera": ("cámara de fotos",),
    "flashlight": ("linterna",),
    "battery": ("pila", "pilas"),
    "plastic bag": ("bolsa", "bolsita", "bolsa de papel"),
    "lunch box": ("vianda", "tupper", "táper", "tupperware"),
    # --- casa: cuarto y living ---
    "door": ("puerta",),
    "door handle": ("picaporte", "manija"),
    "window": ("ventana",),
    "curtain": ("cortina", "cortinas"),
    "window blinds": ("persiana",),
    "lamp": ("lámpara", "velador"),
    "ceiling light": ("foco", "plafón", "lámpara de techo"),
    "light switch": ("interruptor", "llave de luz", "tecla de luz"),
    "power outlet": ("enchufe", "tomacorriente"),
    "power strip": ("zapatilla eléctrica", "alargue"),
    "fan": ("ventilador",),
    "air conditioner": ("aire acondicionado",),
    "heater": ("estufa", "calefactor", "caloventor"),
    "radiator": ("radiador",),
    "mirror": ("espejo",),
    "picture frame": ("cuadro", "portarretrato", "foto"),
    "rug": ("alfombra", "felpudo", "alfombra de baño"),
    "pillow": ("almohada", "almohadón"),
    "blanket": ("manta", "frazada", "acolchado", "cobija"),
    "mattress": ("colchón",),
    "wardrobe": ("placard", "ropero", "armario"),
    "dresser": ("cómoda",),
    "nightstand": ("mesa de luz", "mesita de luz"),
    "drawer": ("cajón",),
    "shelf": ("estante", "estantería", "repisa"),
    "bookshelf": ("biblioteca", "librero"),
    "cabinet": ("mueble", "alacena", "gabinete"),
    "coat rack": ("perchero",),
    "clothes hanger": ("percha",),
    "box": ("caja", "caja de cartón"),
    "basket": ("canasta", "cesta", "canasto", "canasto de ropa"),
    "clothes": ("ropa", "prenda", "remera", "pantalón", "camisa"),
    "toy": ("juguete",),
    "guitar": ("guitarra",),
    "speaker": ("parlante", "altavoz", "bafle"),
    "router": ("router", "módem"),
    "game controller": ("joystick", "mando"),
    "candle": ("vela",),
    "calendar": ("calendario", "almanaque"),
    "iron": ("plancha",),
    "ironing board": ("tabla de planchar",),
    "dumbbell": ("pesa", "mancuerna"),
    "helmet": ("casco",),
    "trash can": ("tacho de basura", "tacho", "basurero", "cesto", "papelero"),
    "stairs": ("escalera", "escaleras", "escalones"),
    "handrail": ("baranda", "pasamanos", "barandal"),
    # --- baño ---
    "bathtub": ("bañera", "bañadera", "tina"),
    "shower": ("ducha",),
    "bidet": ("bidet",),
    "faucet": ("canilla", "grifo"),
    "towel": ("toalla", "toallón"),
    "toilet paper": ("papel higiénico", "rollo de papel"),
    "soap": ("jabón",),
    "soap dispenser": ("dispenser de jabón", "jabón líquido"),
    "shampoo bottle": ("shampoo", "champú", "acondicionador"),
    "toothpaste": ("pasta dental", "dentífrico", "pasta de dientes"),
    "razor": ("afeitadora", "maquinita"),
    "washing machine": ("lavarropas",),
    "bucket": ("balde",),
    "mop": ("lampazo", "trapeador"),
    "broom": ("escoba",),
    "dustpan": ("pala",),
    "scale": ("balanza",),
    # --- cocina ---
    "plate": ("plato",),
    "frying pan": ("sartén",),
    "cooking pot": ("olla", "cacerola"),
    "kettle": ("pava", "hervidor", "pava eléctrica"),
    "coffee maker": ("cafetera",),
    "blender": ("licuadora",),
    "stove": ("cocina", "anafe", "hornalla"),
    "cutting board": ("tabla de picar",),
    "jar": ("frasco",),
    "can": ("lata",),
    "napkin": ("servilleta",),
    "paper towel": ("rollo de cocina", "papel de cocina"),
    "tray": ("bandeja",),
    "sponge": ("esponja",),
    "dish rack": ("escurridor",),
    "bread": ("pan",),
    "egg": ("huevo",),
    "milk carton": ("leche", "cartón de leche"),
    "cereal box": ("cereales", "caja de cereales"),
    "salt shaker": ("salero",),
    "teapot": ("tetera",),
    "drinking straw": ("sorbete", "pajita"),
    "lemon": ("limón",),
    "tomato": ("tomate",),
    "potato": ("papa",),
    "onion": ("cebolla",),
    # --- facultad, oficina y edificio ---
    "whiteboard": ("pizarrón", "pizarra", "pizarra blanca", "pizarrón negro", "pizarra negra"),
    "projector": ("proyector",),
    "projection screen": ("pantalla de proyección", "telón"),
    "computer monitor": ("monitor", "pantalla"),
    "desktop computer": ("computadora de escritorio", "cpu", "gabinete de computadora"),
    "printer": ("impresora",),
    "microphone": ("micrófono",),
    "lectern": ("atril", "podio"),
    "eraser": ("borrador", "goma"),
    "ruler": ("regla",),
    "calculator": ("calculadora",),
    "stapler": ("abrochadora", "engrapadora"),
    "folder": ("carpeta",),
    "paper": ("hoja", "hojas", "papel", "papeles"),
    "sticky note": ("post-it", "postit"),
    "tape": ("cinta", "cinta adhesiva"),
    "poster": ("afiche", "póster"),
    "bulletin board": ("cartelera", "corcho"),
    "sign": ("cartel", "letrero"),
    "exit sign": ("cartel de salida", "salida"),
    "fire extinguisher": ("matafuegos", "extintor"),
    "elevator": ("ascensor",),
    "elevator button": ("botón del ascensor",),
    "water dispenser": ("dispenser de agua", "bebedero", "dispenser"),
    "vending machine": ("máquina expendedora",),
    "locker": ("casillero", "locker"),
    "column": ("columna", "pilar"),
    "ramp": ("rampa",),
    "turnstile": ("molinete",),
    "wheelchair": ("silla de ruedas",),
    "stroller": ("cochecito",),
}

# Textos extra por clase: YOLOE promedia sus embeddings de texto (cada clase cubre sus
# sinónimos sin partirse en dos). Las clases que no figuran usan solo su nombre.
PROMPTS: dict[str, tuple[str, ...]] = {
    "whiteboard": ("whiteboard", "blackboard", "chalkboard"),
    "eyeglasses": ("eyeglasses", "sunglasses"),
    "headphones": ("headphones", "earbuds"),
    "box": ("box", "cardboard box"),
    "basket": ("basket", "laundry basket"),
    "rug": ("rug", "doormat", "bath mat"),
    "clothes": ("clothes", "t-shirt", "pants", "shirt"),
    "plastic bag": ("plastic bag", "paper bag"),
    "kettle": ("kettle", "electric kettle"),
    "trash can": ("trash can", "waste basket", "recycling bin"),
    "pillow": ("pillow", "cushion"),
}
# Las clases COCO van con su nombre solo: promediar sinónimos las empeora (20/11, Replica:
# silla 53 % -> 8 % de recall con «chair, office chair, stool»; botella 18 % -> 7 %).

# Nombres en español de más para clases COCO: lo que YOLOE no distingue bien se une (medido el
# 20/11 en Replica: con «stool», «desk» y «cushion» separados, las sillas bajaban de 55 % a 7 %
# de recall y las mesas de 19 % a 1 %, porque se repartían entre clases vecinas).
COCO_EXTRA_ES: dict[str, tuple[str, ...]] = {
    "chair": ("banqueta", "taburete", "silla de escritorio"),
    "dining table": ("escritorio", "pupitre"),
}

# COCO que no aparecen en una casa ni en una facultad: fuera del vocabulario de YOLOE (en un
# detector abierto, buscarlas solo suma falsos positivos).
COCO_FUERA = frozenset({
    "airplane", "train", "boat", "fire hydrant", "stop sign", "parking meter", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "frisbee", "skis", "snowboard", "kite",
    "baseball bat", "baseball glove", "surfboard", "hot dog", "pizza", "donut", "broccoli",
    "carrot",
})  # fmt: skip

VISIONX_ES: dict[str, tuple[str, ...]] = {
    **{c: n + COCO_EXTRA_ES.get(c, ()) for c, n in COCO_ES.items() if c not in COCO_FUERA},
    **EXTRA_ES,
}
VISIONX_CLASSES: tuple[str, ...] = tuple(VISIONX_ES)
# Todos los alias conocidos: con un modelo COCO quedan solo las COCO; con YOLOE, todas.
ALL_ES: dict[str, tuple[str, ...]] = {
    **{c: n + COCO_EXTRA_ES.get(c, ()) for c, n in COCO_ES.items()},
    **EXTRA_ES,
}


def prompts_de(cls: str) -> tuple[str, ...]:
    """Textos que YOLOE usa para una clase (el nombre solo, si no tiene sinónimos)."""
    return PROMPTS.get(cls, (cls,))
