# Correr VisionX en vivo en la Mac

Punto de entrada para Bauty y su agente: qué instalar, cómo correr los 4 modos con la ELP, qué
probar y qué anotar. Actualizado el **19/11** (lado B: Tomás / Claude Code). Lo hecho y medido
está en [BITACORA](BITACORA.md) (entradas del 18 y 19/11); el estado de cada tarea, en
[ROADMAP](ROADMAP.md). Todo se corre desde la raíz del repo, en `main`.

## 0. Qué hay y qué se probó (19/11)

| Pieza | Estado | Dónde se probó |
|---|---|---|
| 4 modos (buscar, distancia, leer, describir) + preguntas libres | Integrado en `main` | Clips grabados en la PC de Tomás (Windows, CPU) |
| Profundidad `crestereo` / `crestereo-preciso` en la app | Integrado; corrida completa de la app con CREStereo sin errores | PC de Tomás en CPU (lento: 1,7 s por cuadro); **nunca en la Mac dentro de la app** |
| Asistente con Claude Sonnet 5.5 (leer, describir, distancia con detalle, «¿cuál?» describiendo) | Integrado | Clips grabados; ~36 consultas, 0,19 USD |
| Voz Mateo (Edge) con caché de 866 frases | Integrado | PC de Tomás |
| Guía de buscar con **sonar** (bips estéreo) | Integrado | Demos en la PC; **nunca con auriculares en vivo** |
| Micrófono + Whisper small | Medido en la PC (96,7 % de intención) | Falta en la Mac |
| Cámara ELP en vivo | Funciona en la Mac (M2, 14/11) | — |

Lo que **nunca** corrió junto en la Mac: YOLO + CREStereo + asistente + voz + sonar con la ELP en
vivo. Eso es esta prueba.

## 1. Preparar la Mac (una vez, ~15 min)

```bash
git switch main && git pull
uv sync --locked --extra ml --extra deteccion --extra voz --extra ocr --extra ia
uv run python -m visionx.voice descargar --whisper small
uv run python -m visionx.voice descargar --whisper-mlx small   # Mac: Whisper en la GPU (MLX)
uv run python -m visionx.detection descargar --modelo yolo26s-seg
uv run python -m visionx.reading descargar
uv run python scripts/profundidad/descargar_crestereo.py   # ya hecho en la Mac de Bauty
```

En la Mac, con `--whisper-mlx` bajado, la app usa sola Whisper en la GPU: al arrancar imprime `voz a texto: whisper-small-mlx` (si dice `whisper-small@…` está en CPU; forzar con `audio.stt.backend: mlx` en `configs/local.yaml` para ver el error).

`uv sync` deja solo los extras que se nombran: si se corre sin `--extra ia`, el asistente queda
apagado.

**Key de Claude (solo para VisionX).** Tomás la pasa por privado. Va en un archivo, fuera del repo:

```bash
mkdir -p ~/.visionx && chmod 700 ~/.visionx
printf '%s' 'sk-ant-...' > ~/.visionx/anthropic_api_key && chmod 600 ~/.visionx/anthropic_api_key
```

**Nunca como `ANTHROPIC_API_KEY`** (ni `export` ni en `~/.zshrc`): ese nombre lo leen solos
Claude Code y los SDK de Anthropic y gastan los créditos en lo suyo. Pasó el 19/11: la CLI de
Tomás se cobró 5 USD de la key de VisionX. La app lee solo `~/.visionx/anthropic_api_key` (o
`VISIONX_ANTHROPIC_API_KEY`) y se la pasa explícita al SDK. Cada consulta cuesta ~0,5 centavos;
el gasto queda en `outputs/vlm/gasto.jsonl` y con 5 USD anotados la app deja de consultar
(`assistant.budget_usd`).

**Caché de voz (Mateo a 1,15).** Copiar de Drive
`VisionX-artefactos/voz/cache-mateo/edge_es-UY-MateoNeural_15_` a `data/cache/voz/` (tiene que
quedar `data/cache/voz/edge_es-UY-MateoNeural_15_/v1/*.npy`; la carpeta sin `_15_` es la de la
velocidad anterior y ya no se usa). Esas
frases salen al instante y sin red. Alternativa con red (~4 min):
`uv run python -m visionx.voice precalentar`.

**`configs/local.yaml`** (ignorado por Git), mínimo:

```yaml
camera:
  device: 0              # índice de la ELP: `uv run python -m visionx.camera modos`
  backend: avfoundation
audio:
  input_device: null     # micrófono; nombres en `uv run python -m visionx.voice dispositivos`
  output_device: null    # auriculares (sin parlantes: el micrófono escucharía a la app)
```

Opcional: `depth.crestereo.incertidumbre: <ruta a mejor.pt>` para que CREStereo devuelva la b
de Laplace (la app la acepta; por ahora solo se registra).

## 2. Comprobaciones antes de enchufar la cámara (10 min)

| Qué | Comando | Pasa si |
|---|---|---|
| Voz | `uv run python -m visionx.voice decir "Busco la taza."` | Suena Mateo; la 2.ª vez dice «síntesis ~1 ms» (caché) |
| Asistente | El arranque de `ejecutar` imprime `asistente: claude-sonnet-5-5` | Si dice «apagado»: falta el archivo de la key o el extra `ia` |
| App completa con CREStereo, sin cámara | ver abajo | Termina sin errores; `profundidad: {… errores: 0}` |
| Cámara | `uv run python -m visionx.camera ver --config configs/local.yaml` | Ojo izquierdo a la izquierda |

App completa sobre un clip, sin ventana ni micrófono (dice las respuestas por consola):

```bash
uv run python -m visionx.app ejecutar --fuente recordings/dev-1114-distancias --detector yolo \
  --profundidad crestereo --profundidad-precisa crestereo-preciso --sin-ventana \
  --ordenes datasets/demos/ordenes-elegir.jsonl --config configs/local.yaml \
  --traza outputs/beta/humo-crestereo.jsonl
```

En la PC de Tomás (CPU) dio 0 errores pero la distancia precisa no llegó en los 2 s de espera;
en la Mac (617 ms) tiene que llegar. Si no llega, subir `depth.precisa.espera_s`.

## 3. Correr en vivo

```bash
uv run python -m visionx.app ejecutar --fuente usb --detector yolo --profundidad crestereo \
  --voz --config configs/local.yaml --traza outputs/beta/vivo-1.jsonl
```

**Sin `--profundidad-precisa`.** En la primera prueba en vivo (19/11) el preciso tardó 1588 ms
con todo corriendo y no llegaba en `espera_s` (2 s): distancia y describir esperaban 2 s de más
y después usaban igual el continuo (en el dashboard: `espera-vencida`). El continuo de CREStereo
ya es preciso a corta distancia, así que en vivo va solo. Para comparar, agregar
`--profundidad-precisa crestereo-preciso` y subir `depth.precisa.espera_s` a 2,5.

Con la ventana en foco: **espacio** = empezar / terminar de hablar, **c** = cancelar, **q** =
salir. Auriculares puestos. Para verlo por Discord: compartir la ventana de la app **con audio**
(la voz y el sonar salen por los auriculares de la Mac).

Si CREStereo no carga, va lento o se queda sin memoria: repetir con `--profundidad B0
--profundidad-precisa B0` (la beta de siempre) y anotar la diferencia. Para subir los fps (YOLO
en la GPU, 768 px, 3 iteraciones) y elegir con datos: [VELOCIDAD](VELOCIDAD.md).

## 4. Qué probar y qué anotar

Por prueba: OK / falla, qué dijo y el tiempo percibido entre soltar el espacio y la primera
palabra.

| Modo | Decí / hacé | Pasa si |
|---|---|---|
| Buscar | «Buscá la taza» con la taza fuera de cuadro; seguir las instrucciones y acercarse | «Veo taza.» y después «Girá a la izquierda» → «Más a la izquierda» → «Un poco a la izquierda» → «Un poco más abajo»; al centrarla «Centrado. Está a 1,4 metros, adelante.» y una frase de dónde está apoyada; metros cada ~2,5 s al acercarse; «Llegaste… Cuando lo tengas, decí: listo.» y se queda callado (si te alejás, vuelve a guiar). **No termina sola:** termina con «la encontré», «listo», «ya está» o «terminá la búsqueda» → «Listo, terminé de buscar taza.»; sin verla, «Sigo buscando taza.» cada 15 s. En el dashboard, flecha verde al objetivo |
| Elegir | Dos objetos iguales: «buscá la taza» → «la blanca» (o «la de la izquierda») | Pregunta describiéndolos («Veo una taza blanca a la izquierda y un vaso rojo a la derecha. ¿Cuál querés?», ~2 s) y guía a la elegida |
| Distancia | Objeto a 0,5 / 1 / 2 / 3 m de cinta: «¿a qué distancia está la taza?» | En ~2 s, una frase: qué es, sobre qué está, metros, costado en cm (si está corrido) y altura respecto del cuerpo. Error ≤ 10 cm hasta 2 m |
| Leer | Envase a 25–50 cm: «leé» | «Adelante tenés … que dice …»; entre comillas solo lo que el OCR confirmó |
| Leer todo | «leé todo» | Lectura completa por bloques |
| Describir | «describí» / «¿qué hay adelante?» | 3 frases cortas, sin objetos inventados, metros solo de lo medido |
| Preguntas | «¿para qué sirve este remedio?», «¿cuándo vence?», «¿hay algún lugar para sentarme?» | Responde sobre lo que se ve; lo que no está escrito lo aclara como «en general» |
| Seguimiento | Después de leer: «¿y qué más dice?» | Sigue con el mismo objeto (memoria ~2 min) |
| Control | «cancelar» a mitad; «repetí»; orden nueva a mitad de otra | Corta enseguida; repite; la nueva reemplaza a la vieja |
| Calibrar | «calibrá, está a 2 metros» apuntando a un objeto | Hoy dice que esta profundidad no se calibra (falta `calibrar_distancia` en el estimador) |

Al terminar:

```bash
uv run python -m visionx.app latencias --traza outputs/beta/vivo-1.jsonl
```

da la tabla por orden (STT, acuse, primera frase, primer texto de Claude). Subir
`outputs/beta/` y `outputs/vlm/gasto.jsonl` a Drive `VisionX-artefactos/beta/<fecha>/` y anotar
en BITACORA: commit, config, ambiente, la tabla de latencias y qué falló.

## 5. Demos finales con CREStereo

La interacción se ajustó con B0 en la PC de Tomás. Las demos finales se generan en la Mac con la
misma profundidad que en vivo. Clips `recordings/dev-1114-*` (de Bauty); órdenes en
`datasets/demos/`. La latencia del asistente es real (cada frase sale cuando la API la
devolvió); la de profundidad se emula con `--latencia-profundidad`. ~40 consultas en total.

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

El audio de las demos es estéreo (sonar): escucharlas con auriculares. Subir
`outputs/demos/final/` a Drive `VisionX-artefactos/demos/final/`. Las de B0 del 19/11 están en
`VisionX-artefactos/demos/beta-1119c/` para comparar.

## 6. Perillas (configs/default.yaml; se pisan en configs/local.yaml)

| Clave | Por defecto | Para qué |
|---|---|---|
| `modes.guide_style` | `centrar` | `centrar` = instrucciones hasta centrar y después metros (pedido de Tomás tras el vivo del 19/11); `sonar` = bips; `palabras` = frases de dirección |
| `modes.turn_deg` / `more_deg` / `center_deg` | 30 / 15 / 8 | Límites de «Girá», «Más» y «Un poco» (grados) |
| `modes.center_v_deg` / `low_deg` | 12 / 22 | Límites de «Un poco más abajo» y «Abajo» |
| `modes.center_repeat_s` / `approach_meters_s` | 2,5 / 2,5 | Repetir la instrucción / mínimo entre avisos de metros |
| `audio.speech_rate` | 1,15 | Velocidad de Mateo (la caché está generada a esa velocidad) |
| `modes.preempt_min_s` | 0,6 | Con `palabras`: mínimo entre cortes de una frase de dirección vieja |
| `assistant.enabled` / `model` | `true` / `claude-sonnet-5-5` | Apagar el asistente o cambiar de modelo |
| `assistant.budget_usd` | 5,0 | Tope de gasto anotado en `outputs/vlm/gasto.jsonl` |
| `audio.tts.fallback` | `none` | Solo Mateo (pedido de Tomás). `piper` = voz local si Edge falla (otra voz) |
| `proximity.enabled` | `false` | Alerta de cercanía (T20): probarla con CREStereo; con B0 dio falsas alarmas |
| `depth.precisa.espera_s` | 2,0 | Espera de la distancia precisa antes de usar la continua |

## 7. Problemas conocidos

- **Voz entrecortada al preguntar distancias** (19/11, con el banco `s1`): el búfer del parlante
  era mínimo y, mientras corre la profundidad precisa, el audio llegaba tarde. Desde el PR de
  `b/voz-sin-cortes` el búfer es de 0,2 s (`audio.output_latency_s`). Si todavía se corta,
  subirlo a 0,4 en `configs/local.yaml` y mirar la línea «voz entrecortada» de
  `python -m visionx.app latencias --traza …` (cuenta los cortes por frase).

- **Sin internet:** el asistente se apaga (leer y describir vuelven a la versión local) y la voz
  solo dice lo que está en la caché; **una frase nueva sin red no suena** (no hay otra voz). Con
  wifi dudoso: hotspot del teléfono, o `audio.tts.fallback: piper` para que diga todo con otra voz.
- La primera vez que Mateo dice una frase nueva tarda ~1–2 s; la distancia con detalle tarda ~2 s.
- La altura «a la altura de tu cintura» supone la cámara al pecho y nivelada: si va inclinada,
  erra (~20 cm a 1 m con 10°).
- Con dos objetos iguales, si en ~2 s queda uno solo a la vista, guía a ese sin preguntar.
- `scripts/profundidad/descargar_crestereo.py` falla en Windows al borrar `.git` (solo lectura);
  en la Mac no pasa.
