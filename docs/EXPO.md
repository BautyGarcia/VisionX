# Cómo correr el stand en la expo

Guía para el día de la expo (12/12) y los ensayos. Todo se ve a 1920×1080 en una TV o monitor;
cada pantalla se lee a 2–3 m. Son **dos puestos**:

- **Mac de Bauty** (M2 Max, MPS): el stand en vivo, con la cámara y la voz.
- **Segunda pantalla** (monitor o laptop aparte): la presentación de las cámaras, que pasa sola:
  `python -m visionx.gtsynth camaras-ver` (necesita solo `data/stand-video/`, sin cámara ni GPU).

## Las tres pantallas

| Pantalla | Qué muestra | Cómo se abre |
|---|---|---|
| **Stand (en vivo)** | Lo que ve la cámara con detecciones y distancias con su nivel, la profundidad, la vista desde arriba, la conversación, los tiempos de respuesta, los FPS de cada etapa, la cámara en uso y en qué corre cada modelo | `python -m visionx.app ejecutar --fuente usb --detector yolo --profundidad crestereo --voz --camara elp` |
| **Cámaras (segunda pantalla)** | Portada con la comparación de las 6 cámaras; una cámara por pantalla con su recorrido y su **ficha** (datos, errores, qué corrige el clásico, rendimiento y placa); el mosaico de las 6 para cerrar | `python -m visionx.gtsynth camaras-ver`, o la tecla `k` dentro del stand |
| **Cámara nueva** | El alta de una cámara desconocida paso a paso: reconocer, mirar alrededor, estimar, alinear (antes / después), escala, perfil, con métricas | `python -m visionx.app camara-nueva --fuente usb --voz`. Sin cámara: `--fuente sim:video-zed-mini --raiz data/gt-video` |

## Teclas

| Tecla | Stand | Cámara nueva |
|---|---|---|
| espacio | hablar (apretar, decir, apretar) | decir la escala («seis centímetros», «saltear») |
| `c` | cancelar lo que se está diciendo | — |
| `k` | stand ↔ cámaras | — |
| `v` | stand ↔ diagnóstico | — |
| `f` | pantalla completa | — |
| `a` | panel de audio (micrófono y salida) | — |
| `s` | — | saltear la escala |
| `r` | — | empezar de nuevo |
| `q` / Esc / la X | salir | salir |

En la segunda pantalla (`camaras-ver`): ← → cambian de pantalla, espacio pausa, 1–9 saltan
(1 = portada), `f` pantalla completa, `q` sale. Abajo a la derecha se ve dónde está.

## Antes de la expo (checklist)

1. `git pull` de `main` y `uv sync`. En la Mac, `device: auto` usa MPS; no hace falta ONNX.
2. Datos de la segunda pantalla en `data/stand-video/`: los 18 videos `*-video-*.mp4`, los 2
   mosaicos, `resumen-camaras.png` (la portada) y `resumen.json`. Están en el Drive
   (`VisionX-Personal/stand-video`).
3. Pesos en `weights/`: CREStereo (`crestereo/crestereo_eth3d.pth`), YOLO (`yolo26s-seg.pt`) y
   YOLOE (`yoloe-26s-seg-visionx.pt`, Drive `pesos/`).
4. Voz: la key de Claude en `~/.visionx/anthropic_api_key` (nunca como variable de entorno) y
   Edge con internet; sin internet, Piper.
5. RealSense (si se usa): `uv pip install pyrealsense2-macosx`. En la Mac puede pedir permisos
   para el USB: probarlo **antes** del día. Con la cámara conectada, `camara-nueva --fuente
   realsense` tiene que decir «Reconocí la cámara: Intel RealSense D435».
6. ELP: confirmar el nombre USB con `python -m visionx.camera listar` y, si no coincide, sumarlo
   en `configs/camaras/elp.yaml` (`usb:`).
7. Ensayo completo: stand en vivo con los 4 modos, `k` para las cámaras y vuelta, «Cámara nueva»
   con una cámara simulada y con la real.

## Qué contar en cada pantalla

- **Stand:** «Le preguntás algo y te contesta en voz alta con la distancia medida, y te dice si es
  precisa o aproximada. Arriba ves la cámara que usa y en qué corre cada modelo».
- **Cámaras:** «La misma app con seis cámaras estéreo distintas. Si el sistema conoce la cámara,
  el error es del 1 al 5 %; si no la conoce, del 25 al 115 %. Por eso la red de entrada estima
  la cámara».
- **Cámara nueva:** «Enchufás una cámara que nunca vio: la reconoce o la calibra en segundos,
  te pregunta la separación de los lentes y queda lista».
