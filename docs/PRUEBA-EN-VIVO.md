# Prueba en vivo de los 4 modos en la Mac (19/11)

Guía para probar la beta con la ELP, voz y el asistente en la Mac de Bauty (M2 Max). Todo se
corre desde la raíz del repo, en la rama `b/beta-integracion` (PR #27) hasta que se mergee.
Resultados y trazas van a `outputs/beta/<fecha>/`; lo medido se anota en BITACORA.

## 1. Preparar (una vez, ~15 min)

```bash
git fetch origin && git switch b/beta-integracion && git pull
uv sync --locked --extra ml --extra deteccion --extra voz --extra ocr --extra ia
uv run python -m visionx.voice descargar --voz es_ES-davefx-medium --whisper small
uv run python -m visionx.detection descargar --modelo yolo26s-seg
uv run python -m visionx.reading descargar
uv run python scripts/profundidad/descargar_crestereo.py      # solo si se prueba CREStereo
```

**API key de Claude** (leer, describir y preguntas libres). Sin ella la app funciona igual, con
la lectura y la descripción locales de antes. Va en un archivo privado, fuera del repo:

```bash
mkdir -p ~/.visionx && chmod 700 ~/.visionx
printf '%s' 'sk-ant-...' > ~/.visionx/anthropic_api_key && chmod 600 ~/.visionx/anthropic_api_key
```

**Nunca como `ANTHROPIC_API_KEY`** (ni en `~/.zshrc` ni en ningún lado): ese nombre lo leen solos
Claude Code y los SDK de Anthropic, y gastan los créditos de la key en lo suyo. Pasó el 19/11: la
CLI se cobró los 5 USD de la key de VisionX. Si ya está, borrarla de `~/.zshrc` y abrir una
terminal nueva. Lo ideal es una key de un *workspace* propio de VisionX en la consola de
Anthropic, con un tope de gasto mensual. Cada consulta cuesta ~0,5 centavos de dólar (Sonnet 5.5) y
queda anotada en `outputs/vlm/gasto.jsonl`; con 5 USD anotados la app deja de consultar
(`assistant.budget_usd`).

**Caché de voz (Mateo).** La voz necesita internet; las frases frecuentes ya generadas salen al
instante y también sin red. Copiar la carpeta `voz-cache` de Drive
(`VisionX-artefactos/voz/cache-mateo/`) a `data/cache/voz/` (debe quedar
`data/cache/voz/edge_es-UY-MateoNeural/v1/*.npy`). O generarla en la Mac (~5 min, con red):
`uv run python -m visionx.voice precalentar`.

**`configs/local.yaml`** de la Mac (ignorado por Git):

```yaml
camera:
  device: 0            # el índice de la ELP (`uv run python -m visionx.camera modos`)
audio:
  input_device: null   # micrófono por defecto; con AirPods, ver `python -m visionx.voice dispositivos`
  output_device: null
```

## 2. Comprobaciones rápidas (5 min)

| Qué | Comando | Pasa si |
|---|---|---|
| Voz Mateo | `uv run python -m visionx.voice decir "Busco la taza."` | Suena Mateo; la 2.ª vez «síntesis ~1 ms» |
| Sin red | Apagar el wifi y repetir con otra frase | Habla Piper (davefx); no se cuelga |
| Asistente | `ejecutar` imprime `asistente: claude-sonnet-5-5` al arrancar | Si dice «apagado», falta `~/.visionx/anthropic_api_key` o el extra `ia` |
| Cámara | `uv run python -m visionx.camera ver --config configs/local.yaml` | Ojo izquierdo a la izquierda |

## 3. Correr la app

```bash
uv run python -m visionx.app ejecutar --fuente usb --detector yolo --profundidad crestereo \
  --profundidad-precisa crestereo-preciso --voz --config configs/local.yaml \
  --traza outputs/beta/1119/vivo-1.jsonl
```

CREStereo en la Mac (BITACORA 2026-11-18): continuo 640×360 P50 120 ms, preciso 1280×720
P50 617 ms; en la cinta de 1 a 3 m erra ≤ 3,4 cm contra hasta 8,8 cm de B0. Si no carga o algo
anda mal, repetir con `--profundidad B0 --profundidad-precisa B0` (la beta de siempre) y anotar la
diferencia. Al terminar, `uv run python -m visionx.app latencias --traza
outputs/beta/1119/vivo-1.jsonl` da la tabla de tiempos por orden para la BITACORA. Con la ventana en
foco: **espacio** = hablar / terminar de hablar, **c** = cancelar, **q** = salir. Auriculares
puestos (sin parlantes: el micrófono escucharía a la app).

## 4. Qué probar y qué anotar

Anotar por prueba: OK / falla, qué dijo, y el tiempo percibido entre soltar el espacio y la
primera palabra.

| Modo | Decí | Pasa si |
|---|---|---|
| Buscar | «Buscá la taza» (fuera de cuadro), girar despacio hasta encontrarla y acercarse | Bip al verla; «Taza a la izquierda.» y después frases cortas («Un poco a la izquierda.», «Al frente, a 1,2 metros.»); «Llegaste» |
| Buscar rápido | Con la taza guiada, girar rápido de un lado al otro | La frase vieja se corta y suena la del lado correcto, sin atrasarse |
| Elegir | Dos botellas, «buscá la botella» → «la de la izquierda» | Pregunta cuál y guía a la elegida |
| Distancia | Objeto a 0,5 / 1 / 2 / 3 m de cinta: «¿a qué distancia está la taza?» | Error ≤ 10 cm hasta 2 m; abstiene ante una pared |
| Leer | Envase a 25–50 cm: «leé» | «Adelante tenés … que dice …»: objeto + lo importante. Comillas = texto confirmado por el OCR |
| Leer todo | «leé todo» | Lectura completa por bloques (sin asistente) |
| Describir | «describí» / «¿qué hay adelante?» | Frases naturales, sin objetos inventados; metros solo de lo medido |
| Pregunta | «¿para qué sirve este remedio?», «¿cuándo vence?», «¿de qué color es la taza?», «¿hay algún lugar para sentarme?» | Responde sobre lo que se ve; si no se lee, pide acercar |
| Seguimiento | Después de leer: «¿y qué más dice?» | Sigue con el mismo objeto (memoria ~2 min) |
| Control | «cancelar» a mitad de una respuesta; «repetí»; orden nueva a mitad de otra | Corta enseguida; repite; la nueva reemplaza a la vieja |
| Sin red | Apagar el wifi y pedir «leé» y «describí» | Lee/describe con la versión local; las preguntas dicen que no hay conexión |

Al terminar, subir `outputs/beta/1119/` y `outputs/vlm/gasto.jsonl` a Drive
`VisionX-artefactos/beta/1119/`.

## 5. Demos finales con CREStereo (antes de la prueba en vivo)

La interacción se ajusta con B0 en la PC de Tomás (sin GPU, CREStereo tarda 1,7 s por cuadro
ahí). Las demos finales se generan en la Mac con la misma profundidad que la prueba en vivo. Los
clips `recordings/dev-1114-*` son los de Bauty; las órdenes están en `datasets/demos/`. Cada
demo hace ~5–9 consultas al asistente (~0,03 USD). La latencia del asistente es real (cada frase
sale cuando la API la devolvió); la de profundidad se emula con `--latencia-profundidad`.

```bash
mkdir -p outputs/demos/final
for c in lectura escenas buscar-a distancias; do
  uv run python -m visionx.modes demo --fuente recordings/dev-1114-$c \
    --ordenes datasets/demos/ordenes-$c.jsonl --profundidad crestereo \
    --latencia-profundidad 0.13 --config configs/local.yaml \
    --traza outputs/demos/final/demo-$c.jsonl --salida outputs/demos/final/demo-$c.mp4 \
    > outputs/demos/final/demo-$c.txt 2>&1
done
# Elegir entre iguales describiéndolos («la blanca»), sobre el clip de distancias:
uv run python -m visionx.modes demo --fuente recordings/dev-1114-distancias \
  --ordenes datasets/demos/ordenes-elegir.jsonl --profundidad crestereo \
  --latencia-profundidad 0.13 --config configs/local.yaml \
  --traza outputs/demos/final/demo-elegir.jsonl --salida outputs/demos/final/demo-elegir.mp4
```

Subir `outputs/demos/final/` a Drive `VisionX-artefactos/demos/final/` y revisarlas juntos.

## 6. Problemas conocidos

- La primera vez que Mateo dice una frase nueva tarda ~1–2 s (red). Lo frecuente está en la caché.
- El asistente necesita red. Con el wifi del lugar inestable, tener el hotspot del teléfono.
- Latencias de la Mac sin medir con el asistente: medirlas es parte de esta prueba.
- `scripts/profundidad/descargar_crestereo.py` falla en Windows al borrar `.git` (archivos de solo
  lectura); en la Mac no pasa. En Windows: quitar el atributo de solo lectura y borrar `.git` a mano.
