# Medir los modos para el informe (M8 buscar, M10 voz / leer / describir)

Las métricas de TECNICO (tabla M0–M12) que faltan del lado de la app. Cada una ya tiene su
protocolo en `python -m visionx.protocolos` (plantilla CSV + guía + `evaluar`); esta guía dice
**cómo correrlas en dos sesiones** con la Mac y la ELP (Bauty en la cámara, Tomás por Discord) y
dónde queda cada número. M7 (distancia por objeto) sale del GT sintético
([GT-SINTETICO](GT-SINTETICO.md)) y del banco ELP de Bauty; M9 (latencia) de las trazas de vivo
(`python -m visionx.app latencias`, ver [VELOCIDAD](VELOCIDAD.md)).

## Instrucciones para el agente (leer primero)

1. `git pull`; la app tiene que arrancar como en [PRUEBA-EN-VIVO](PRUEBA-EN-VIVO.md) (voz, asistente,
   `--profundidad crestereo --profundidad-precisa crestereo-preciso`, config ganadora de
   VELOCIDAD en `configs/local.yaml`).
2. Para cada métrica: crear la planilla con `plantilla`, correr la sesión, completar la planilla,
   `evaluar`. Planillas y resúmenes al repo en `datasets/modos/<métrica>/`; trazas, audio y
   clips a Drive `VisionX-artefactos/modos/<fecha>/`.
3. Una traza nueva por corrida de la app (`--traza`); nunca concatenar dos.
4. Entrada en BITACORA con la tabla de resultados, contra los umbrales de abajo, y los límites
   (personas, ambientes, luz). No cambiar los umbrales después de medir.

| Métrica | Qué se mide | Umbral (TECNICO) | Sesión |
|---|---|---|---|
| M8 buscar | 30 búsquedas en 3 ambientes, ≥ 10 min con el objeto ausente, 20 cancelar, 10 pérdidas | ≥ 80 % encontrado en ≤ 60 s; ≤ 1 falso aviso por minuto; cero audio después de cancelar o vencido | 1 |
| M10 voz | 61 órdenes dichas por una persona (no Tomás: ya está medido con él) | ≥ 90 % intención y objeto correctos | 2 |
| M10 leer | 30 textos, cada uno a 0,5 y 1 m | CER y errores numéricos; cero números inventados | 2 |
| M10 describir | 30 escenas anotadas | Objetos mencionados correctos, cero metros inventados | 2 |

## Sesión 1 — M8 buscar (~1,5 h)

**Preparar.** Tres ambientes distintos (p. ej. living, cocina, un cuarto). En cada uno, 5–6
objetos del catálogo (taza, botella, mochila, libro, silla, celular…). La persona lleva la
cámara al pecho; si es Bauty con los ojos abiertos, que no mire la pantalla.

```bash
uv run python -m visionx.protocolos m8 plantilla --salida datasets/modos/m8
# una corrida por ambiente:
uv run python -m visionx.app ejecutar --fuente usb --detector yolo --profundidad crestereo \
  --profundidad-precisa crestereo-preciso --voz --config configs/local.yaml \
  --traza outputs/modos/m8/living.jsonl
```

**Guion por ambiente** (siempre arrancando de espaldas al objeto o con él fuera de cuadro):

1. **10 búsquedas**: «Buscá la taza» → seguir las instrucciones → al tenerla en la mano, «la
   encontré». Si a los 60 s no la encontró, «terminá la búsqueda» y anotarlo.
2. **≥ 200 s ausente**: pedir un objeto que NO está en el ambiente (y que tampoco haya algo
   parecido) y caminar normal por el ambiente; al final, «terminá la búsqueda». Cuenta cada
   «Veo …» que suene (falso aviso). Sumando los tres ambientes, ≥ 600 s.
3. **7 cancelar**: «Buscá la botella» y a los 2–5 s «pará». No tiene que sonar nada de esa
   búsqueda después.
4. **3–4 pérdidas**: con el objeto centrado, darse vuelta o taparlo; tiene que decir «Perdí …»;
   después «terminá la búsqueda».

**Planilla.** El borrador sale de la traza (tiempos y resultado sugerido); revisar fila por fila
con lo anotado en la sesión, en especial las `REVISAR-fallo-o-ausente` (fallo si el objeto
estaba, ausente si no):

```bash
uv run python -m visionx.protocolos m8 borrador --traza outputs/modos/m8/living.jsonl \
  --ambiente living --salida datasets/modos/m8/living.csv
# … lo mismo por ambiente; unir las filas en datasets/modos/m8/planilla.csv (columna traza)
uv run python -m visionx.protocolos m8 evaluar --planilla datasets/modos/m8/planilla.csv \
  --salida datasets/modos/m8/resultado
```

`exito_s` puede quedar vacío: `evaluar` lo toma del momento en que la persona dijo «la encontré».
El resumen trae éxito ≤ 60 s, falsos avisos por minuto (`Veo …` o bip con el objeto ausente),
audio después de cancelar o de vencer, y si cada pérdida dijo «Perdí».

## Sesión 2 — M10 voz, leer y describir (~1,5 h)

### Voz (61 órdenes, ~15 min)

Una persona que no sea Tomás, con el micrófono de la Mac, sin la app abierta:

```bash
uv run python -m visionx.protocolos voz plantilla --salida datasets/modos/voz
uv run python -m visionx.protocolos voz grabar --planilla datasets/modos/voz/planilla.csv
uv run python -m visionx.protocolos voz evaluar --planilla datasets/modos/voz/planilla.csv \
  --salida datasets/modos/voz/resultado
```

Enter graba cada frase (r repite, s saltea, q sale y retoma después). Anotar en `notas` quién
habla, el micrófono y el ambiente. Con MLX en la Mac, `evaluar` mide el motor que usa la app.

### Leer (30 textos × 2 distancias)

Textos reales variados: envases, remedios, carteles, un libro, una factura, una pantalla. Por
cada texto, un clip corto a 0,5 m y otro a 1 m (cámara quieta, texto centrado):

```bash
uv run python -m visionx.protocolos ocr plantilla --salida datasets/modos/ocr
uv run python -m visionx.camera grabar recordings/m10-ocr-07-05 --segundos 3 --nota "texto 7, 0,5 m"
```

En la planilla: `clip` = carpeta del clip **relativa a la planilla** (p. ej.
`../../../recordings/m10-ocr-07-05`), `t_s` = 1.0 (un segundo después del inicio del clip),
`texto_referencia` = TODO el texto visible, en orden de lectura, con números tal cual. Después
`ocr evaluar`. Mide el OCR local (CER y números); la lectura con Claude se audita aparte con las
mismas referencias (cero números que no estén en el texto).

### Describir (30 escenas)

Escenas distintas (mesa con cosas, estante, escritorio, cocina…), un clip de 3 s por escena. Antes
de correr nada, anotar en `objetos_json` cada objeto visible del catálogo con su zona y distancia
aproximada. `describir evaluar` audita lo que menciona el modo local; para la descripción con
Claude, revisar a mano con la misma lista: objetos correctos, objetos inventados y cero metros que
no vengan de la medición (la verificación de números de T13 ya los bloquea).

## Lo que queda registrado

- `datasets/modos/<métrica>/planilla.csv` y `resultado/resumen.json` (repo).
- Trazas, audio y clips en Drive `VisionX-artefactos/modos/<fecha>/`.
- BITACORA: tabla de resultados contra los umbrales, quién participó y los límites.
