# Velocidad en vivo: de 4 a 10 fps sin perder calidad

Guía para medir y elegir cómo correr VisionX más rápido en la Mac (M2 Max). La meta es que el
sistema sea **interactivo**: que la guía de buscar reaccione al girar y que las respuestas salgan
rápido. Escrita el 19/11 (Tomás / Claude Code); se completa con los datos que mida Bauty y la
decisión va a [BITACORA](BITACORA.md). Cómo correr la app en sí: [PRUEBA-EN-VIVO](PRUEBA-EN-VIVO.md).

## Instrucciones para el agente (leer primero)

Objetivo: medir `C0`…`C4` en la Mac, registrar **todas** las métricas y proponer la config
ganadora según el criterio. No cambiar código ni el criterio; si algo no se puede medir, anotarlo
en `notas` y seguir.

1. `git switch main && git pull`; crear la rama `a/velocidad-medicion` (o `b/…` si la corre Tomás).
2. Comprobar que existan `configs/local.yaml` (cámara y audio de la Mac), los pesos de YOLO y
   CREStereo, y la key en `~/.visionx/anthropic_api_key` (ver [PRUEBA-EN-VIVO](PRUEBA-EN-VIVO.md)).
3. Una sola vez: la sección **B** (CREStereo sobre clips, 3 y 4 iteraciones).
4. Para cada `C` en `C0 C1 C2 C3 C4`, en ese orden: sección **A** (clips) y sección **C** (vivo,
   con la persona siguiendo el guion; pedirle que avise al terminar).
5. Completar **una fila por config** en `datasets/velocidad/resultados.csv` con los números
   exactos que imprimen los comandos (sin redondear de más ni estimar). Dejar vacío lo que no
   se midió y explicarlo en `notas`.
6. Copiar `outputs/velocidad/` a Drive `VisionX-artefactos/velocidad/<fecha>/`.
7. Escribir la entrada en [BITACORA](BITACORA.md): equipo, commit, la tabla completa, qué config
   cumple el criterio, cuál gana y por qué, y cualquier error. Si gana una config, proponer en el
   mismo PR el cambio en `configs/default.yaml` (o solo en `configs/local.yaml` de la Mac si
   depende de la Mac, como `device: mps`).
8. Commit y PR con la planilla, la BITACORA y el cambio de config propuesto. No subir
   `outputs/`, pesos ni la key.

Terminado = las 5 filas completas (o justificadas), los crudos en Drive y la entrada en BITACORA.

## Por qué 4 fps (primera prueba en vivo, 19/11)

| Pieza | Medido en vivo | Límite propio |
|---|---|---|
| YOLO `yolo26s-seg` a 960 px | 299 ms por cuadro | ~3,3 fps |
| CREStereo continuo 640×360, 4 iteraciones | 160 ms (120 ms solo, BITACORA 18/11) | ~6–8 fps |
| CREStereo preciso 1280×720 | 1588 ms | a pedido |

Detector y profundidad corren en paralelo; la app avanza al ritmo del más lento. El cuello es
YOLO, y por defecto corre en **CPU** (`detection.device: cpu`, elegido así porque la PC de Tomás
no tiene una GPU que sirva). En la PC de Tomás (i7, CPU) YOLO solo tarda ~150 ms por cuadro.

## Las 3 opciones y las configuraciones a medir

1. **YOLO en la GPU de la Mac** (`detection.device: mps`). La misma red: no debería perder calidad.
2. **Imagen de entrada de 768 px** en vez de 960 (`detection.imgsz`). Más rápido; puede perder
   objetos chicos o lejanos (en la PC, a 640 px un bolso a 3 m pasó de 100 % a 0 % de cuadros).
3. **CREStereo continuo con 3 iteraciones** en vez de 4 (`depth.crestereo.continuo.iters`).

Cada configuración es un archivo en `configs/velocidad/` que incluye `configs/local.yaml`
(cámara, audio) y pisa solo lo suyo. **No editar `local.yaml` por prueba.**

| Config | YOLO | imgsz | Iteraciones | Para qué |
|---|---|---|---|---|
| `C0` | cpu | 960 | 4 | Base actual, para comparar |
| `C1` | mps | 960 | 4 | Opción 1 |
| `C2` | mps | 768 | 4 | Opciones 1 + 2 |
| `C3` | mps | 960 | 3 | Opciones 1 + 3 |
| `C4` | mps | 768 | 3 | Las tres |

El preciso (1280×720, 4 iteraciones, 2 pasadas) **no se toca**: es el que se usa para las
métricas finales de la red (CREStereo + U-Net) y también a pedido en vivo para «¿a qué distancia
está?». En todas las configs su espera sube a 3 s (`depth.precisa.espera_s`), porque con todo
corriendo tardó ~1,6 s y con 2 s se vencía.

## Criterio de decisión (fijado antes de medir)

Gana **la configuración con más fps del detector en vivo** que cumpla **todo** esto:

1. **Interactividad en vivo:** detector ≥ 10 fps y profundidad continua ≥ 5 fps.
2. **Detección:** en cada clip, la presencia de cada objeto de prueba no baja más de **5 puntos**
   contra `C0`.
3. **Profundidad:** el error contra la cinta del continuo no sube más de **1 cm** contra 4
   iteraciones, a 1, 1,5 y 2 m.
4. **Sin errores** en la corrida en vivo (detector, profundidad y preciso con `errores: 0`).

Si ninguna llega a 10 fps, gana la que más se acerque cumpliendo 2–4, y se anota qué falta.

## Qué correr para cada configuración

Desde la raíz del repo, en `main`. Reemplazar `C1` por la config que se mide. Las salidas
crudas van a `outputs/velocidad/<config>/` (fuera de Git) y después a Drive.

### A. Clips: detector (≈ 5 min por config)

```bash
C=C1; mkdir -p outputs/velocidad/$C
for clip in buscar-a:bottle buscar-b:cup buscar-c:backpack \
            distancias:"cup bottle handbag backpack"; do
  name=${clip%%:*}; classes=${clip#*:}
  uv run python -m visionx.detection presencia recordings/dev-1114-$name \
    --clases $classes --config configs/velocidad/$C.yaml \
    --csv outputs/velocidad/$C/presencia.csv
done
```

De ahí salen la latencia del detector (mediana y p90 en ms) y el % de cuadros con cada objeto.
El «bolso» de distancias puede salir como `handbag` o `backpack`: usar el mayor de los dos.

### B. Clips: CREStereo (una sola vez para las 3 y 4 iteraciones)

```bash
uv run python -m visionx.depth.crestereo_adapt latencia recordings/dev-1114-distancias \
  --resoluciones 640x360 --iters 3 4 --pasadas 2 --config configs/local.yaml
uv run python -m visionx.depth.crestereo_adapt cinta <carpeta de cinta-s7> \
  --modos continuo --variantes 640x360:i3p2 --config configs/local.yaml
```

La primera da la latencia de cada variante (sin YOLO al lado); la segunda, el error contra la
cinta del continuo con 4 iteraciones y con 3 (`i3p2`). Estos valores van a todas las filas con
esas iteraciones.

### C. En vivo (≈ 3 min por config)

```bash
uv run python -m visionx.app ejecutar --fuente usb --detector yolo --profundidad crestereo \
  --profundidad-precisa crestereo-preciso --voz --config configs/velocidad/$C.yaml \
  --traza outputs/velocidad/$C/vivo.jsonl
```

Guion fijo para que las corridas sean comparables (~2 min, la misma escena y la misma taza a
1,5 m de cinta):

1. «Buscá la taza» con la taza fuera de cuadro; girar hasta centrarla y acercarse hasta «Llegaste».
2. «¿A qué distancia está la taza?» (usa el preciso).
3. «Describí».
4. Salir con **q** a los ~2 min.

Al salir, la app imprime un resumen. Los fps salen de ahí:
`vivo_det_fps = detector.procesados / segundos` y `vivo_prof_fps = profundidad.procesados /
segundos`. Anotar la latencia del preciso del dashboard o de la traza. Después:

```bash
uv run python -m visionx.app latencias --traza outputs/velocidad/$C/vivo.jsonl
```

da la primera frase de cada orden (ms desde soltar el botón): anotar la de «buscá» y la de
«¿a qué distancia…?».

## Cómo reportar

1. **Una fila por config** en `datasets/velocidad/resultados.csv` (está versionado: el agente de
   Bauty la completa y la commitea en el PR del reporte). Columnas:

   | Columna | De dónde |
   |---|---|
   | `config`, `fecha`, `equipo`, `commit` | C0…C4; `git rev-parse --short HEAD`; «Mac M2 Max» |
   | `yolo_dispositivo`, `imgsz`, `cre_iters` | De la config |
   | `det_mediana_ms_clip`, `det_p90_ms_clip` | A, del clip de distancias |
   | `presencia_*` | A, % de cuadros de cada objeto en cada clip |
   | `cre_mediana_ms_clip`, `cre_p95_ms_clip` | B, de la variante con esas iteraciones |
   | `cinta_err_1m_mm`, `cinta_err_1_5m_mm`, `cinta_err_2m_mm` | B, continuo con esas iteraciones |
   | `vivo_*` | C (segundos, fps, preciso, primera frase, errores) |
   | `cumple` | `sí` / `no` según el criterio, y cuál falla en `notas` |

2. **Crudos a Drive:** `outputs/velocidad/` → `VisionX-artefactos/velocidad/<fecha>/`.
3. **BITACORA:** una entrada con la tabla, la config elegida y por qué. Con la decisión, el
   ganador pasa a `configs/default.yaml` (o a `configs/local.yaml` de la Mac si es solo de ella).

## Si algo falla

- **YOLO en MPS da error o resultados raros:** anotarlo en `notas`, marcar `cumple: no` y seguir
  con las demás; volver a `device: cpu`.
- **Se queda sin memoria la GPU:** anotarlo; probar sin `--profundidad-precisa` para separar.
- **La app no arranca con `--config configs/velocidad/C1.yaml`:** revisar que exista
  `configs/local.yaml` (la incluye).
