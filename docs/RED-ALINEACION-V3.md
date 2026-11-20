# Red de alineación v3: corregir al clásico, medida en profundidad y con confianza

Guía para entrenar la próxima red de alineación (la «red de entrada» de CREStereo adaptable,
T12). Está escrita para que Bauty, o su Claude, la pueda implementar y entrenar sin contexto
previo. Decisión de Tomás y Bauty del 20/11. Hasta que la v3 cumpla los criterios de la
sección 6, **el sistema sigue con el clásico con lente y distancia conocida** (decisión (a)).

## 1. Por qué una v3

La v2 (`outputs/crestereo_alin/v2-45ep/mejor.pt`, PR #37) **le gana al clásico en el campo y
pierde en profundidad**. Cambio de la profundidad con distancia conocida, en %, sobre los
mismos 6 pares de la ELP (entrada de la BITÁCORA «red de alineación promediada y encima del
clásico», 19/11):

| Rangos | Clásico con lente | Red v2 promedio | Lente + red v2 | Oráculo |
|---|---:|---:|---:|---:|
| ×0,25 | **2,63** | 2,73 | 5,09 | 0,56 |
| ×1 | **5,57** | 6,66 | 13,11 | 0,75 |

De ahí salen los tres cambios:
1. **Corregir al clásico en vez de reemplazarlo.** Poner la v2 encima del clásico empeora
   porque nunca vio pares ya corregidos: mete un error de columnas que el clásico ya había
   sacado. La v3 se entrena en ese régimen. En el peor caso iguala al clásico.
2. **Medirla en profundidad.** El error del campo en píxeles no predice el error en metros: la
   v2 ganó en una métrica y perdió en la otra. La pérdida y la elección del checkpoint tienen
   que mirar la profundidad.
3. **Que diga cuánto confía.** Así, donde duda, queda el clásico. Además, esa confianza se suma
   a la incertidumbre que llega a la voz (tres niveles,
   [INCERTIDUMBRE-Y-GT](INCERTIDUMBRE-Y-GT.md)).

Lo que se mantiene de la v2: codificador, correlación, cabeza con FiLM y mapa de rayos, ajuste
geométrico de 6 parámetros sin pesos, resto denso, **un solo campo por cámara** (mediana de los
pares, `pooled_field`). Tampoco se le pide el corrimiento constante de columnas: lo arregla la
distancia conocida.

## 2. Qué tiene que hacer la v3

### 2.1 Corregir al clásico (resto)

- **Entrada:** el par de una cámara **ya corregido por el clásico con lente y regla**
  (`lens.py`, el modo `clasico` de `align_apply.Alineador`, con los puntos de todos los pares de
  esa cámara juntos), más su calibración nominal (vector FiLM y rayos, como hoy).
- **Salida:** un campo de resto `R` (de dónde tomar cada píxel en la derecha corregida por el
  clásico), más la confianza σ de la sección 2.3.
- **Objetivo exacto.** Si `F` es el campo exacto que deshace la desalineación (de dónde tomar en
  la derecha que llegó) y `M` es el mapa del clásico (la derecha corregida es `der(M(q))`),
  hace falta `M(R(q)) = F(q)`, o sea **`R = M⁻¹ ∘ F`**. `M` es suave: se invierte en la grilla
  de 4 px (`COARSE`) con unas iteraciones de punto fijo y se interpola, como el campo exacto de
  hoy.
- **Arranque en cero.** La última capa del resto y los 6 parámetros arrancan en cero:
  `R = identidad`, o sea, exactamente el clásico. Con eso, el entrenamiento solo puede alejarse
  del clásico si baja la pérdida.
- **Aplicación:** se compone con el mapa del clásico y la imagen se remuestrea **una sola vez**
  (`apply_field` con `base_maps`, ya existe).
- **Caché del clásico.** El clásico necesita varios pares de la misma cámara. Para entrenar:
  sortear K cámaras «reales» (hermanas o sorteadas, sección 3), aplicar cada una a un grupo de
  6 vistas, correr el clásico **una vez por cámara** y guardar sus parámetros (JSON por
  cámara). Cada muestra de entrenamiento es (vista, cámara): el mapa `M` sale del caché. Sin el
  caché, el clásico domina el tiempo del entrenamiento.

### 2.2 Medirla en profundidad

**Pérdida.** Los renders traen la Z exacta, así que se conoce la disparidad de cada píxel:
`d = fx · b / Z`. Un error de columnas `e_u` (en px) cambia la profundidad en `e_u / d` relativo;
uno de filas `e_v` hace fallar la correlación de CREStereo. Pérdida por píxel válido
(0,3 m ≤ Z ≤ 6 m, sin oclusión entre ojos):

```
L = λ_prof · ρ( (e_u − mediana(e_u)) / max(d, d_min) )   # error relativo de profundidad
  + λ_filas · ρ( e_v )                                     # filas, en px
  + λ_conf · NLL_Laplace(e, σ)                             # confianza (2.3)
```

- `ρ` = Charbonnier o Huber.
- Restar la mediana de `e_u` por cámara: el corrimiento constante lo arregla la distancia
  conocida, igual que en la v2.
- `d_min` = disparidad a 6 m, para que lo lejano no pese infinito.
- Para empezar: λ_prof = 1, λ_filas = 0,5, λ_conf = 0,1. Ajustarlos en val, nunca en test.

Las fuentes sin Z (InStereo2K, clips de la ELP) siguen con la pérdida de campo de la v2 y con
menos peso. Sirven para que la red vea texturas reales, pero no definen nada.

**Elección del checkpoint.** Cada N épocas, correr la **profundidad de verdad** (CREStereo
preciso + distancia conocida) sobre un subconjunto fijo de val (por ejemplo 40 vistas de
`elp-margen`, 6 por cámara) y quedarse con el checkpoint de **menor error de distancia por
objeto**. No con el de menor pérdida de campo: ese fue el error de la v2.

**Opcional:** si la pérdida de arriba no sigue al error real de CREStereo, un ajuste fino de
punta a punta con CREStereo congelado. Es caro: solo si hace falta.

### 2.3 Confianza

- **Qué predice:** una escala de Laplace σ por celda y por eje (filas, columnas) del error que le
  queda al campo final. Se entrena con `|e|/σ + log σ`, la misma verosimilitud que ya usan la
  cabeza auxiliar y la U-Net.
- **Por cámara:** mediana de los σ de los pares (como el campo).
- **Usos:**
  1. **Cuánto aplicar del resto.** Peso por celda `w = |R|² / (|R|² + σ²)`. Si la corrección es
     chica frente a la duda, `w → 0` y queda el clásico. Si es grande y segura, `w → 1`. No tiene
     parámetros que ajustar.
  2. **A la voz.** En disparidad, `σ_d² = σ_U-Net² + σ_col²`. Pasa a profundidad relativa como
     `σ_d / d` y entra en los tres niveles (precisa / aproximada / no se pudo medir).
     Esa parte la integra Tomás en `shared/uncertainty.py` cuando la v3 exista.
- **Cómo se verifica (val):**
  - fracción de errores con `|e| ≤ σ`, que para Laplace es ≈ 63 %, y con `|e| ≤ 3σ`, ≈ 95 %;
  - curva por deciles de σ;
  - correlación de Spearman entre σ y el error real de distancia por objeto.

## 3. Datos

Todo sale de `data/gt-sintetico-elp/` (copia en el Drive `VisionX-Personal/gt-sintetico-elp/`;
detalle en [GT-SINTETICO](GT-SINTETICO.md)). Se lee con `visionx.gtsynth`.

### 3.1 Para entrenar (solo escenas `train`)

| Variante | Vistas | Qué aporta |
|---|---:|---|
| `elp-margen` (train) | 2400 | ELP a media resolución (960×540 nominal), con margen |
| `camara-sorteada` | 2400 | En cada vista, otro campo (55–130°), otra separación (3–20 cm) y otra proporción (16:9 o 4:3): lo que necesita la v3 para servir en cualquier cámara |

**Cómo se arma una muestra:**
1. Leer el lienzo: `v.izq()`, `v.der()`, `v.z()`, con la geometría del lienzo `v.geometria`.
2. Sortear (o tomar del caché) la cámara «real»: `perturb.sample_sibling` con la geometría
   nominal `v.geometria_nominal`.
3. Desalinear `der` **sobre el lienzo**: el margen de 12,5 % por lado evita bordes negros.
4. Recortar la parte nominal de las tres imágenes con `v.recortar(...)` y calcular el campo
   exacto en coordenadas nominales.
5. Llevar el recorte a múltiplos de 8 (`STRIDE`), corrigiendo `cx`, `cy`.

Verificado el 20/11: en las variantes con margen, `v.recortar` da la misma imagen que `elp`
(diferencia 0) y la misma Z salvo un 0,001 % de píxeles en bordes de objetos.

**Test mínimo antes de entrenar:** aplicar el campo exacto a la derecha desalineada y comparar
con la derecha limpia. Fuera de los bordes, el error tiene que ser ≲ 1 nivel de gris.

### 3.2 Para validar (solo escenas `val`)

| Variante | Vistas | Qué mide |
|---|---:|---|
| `elp-margen` (val) | 160 | Las mismas poses que `elp`: comparable con todas las tablas anteriores |
| 6 perfiles: `angosta-60`, `base-corta-3`, `media-90`, `cuatro-tercios-90`, `base-larga-20`, `gran-angular-130` | 6 × 160 | ¿Generaliza a otras cámaras? |
| `elp-arriba-1`, `elp-arriba-2`, `elp-adelante-2`, `elp-arriba-1-adelante-2` | 4 × 160 | Ojo derecho **realmente** corrido 1–2 mm: con paralaje de verdad, cosa que un remuestreo no simula |
| `pose-altura`, `pose-poca-textura`, `pose-cerca` | 3 × 80 | Casos difíciles para la correlación |
| `secuencias` | 320 | 10 recorridos de 8 cuadros por escena: el campo por cámara con cuadros seguidos, como en el uso real |
| `ojo-de-pez` | 160 | **Fuera del alcance.** Solo se informa |

`test` (6 escenas) no está renderizado para estas variantes. Se renderiza y se usa **una sola
vez**, cuando la v3 esté cerrada en val.

## 4. Pasos

1. **Ver la evaluación de la v2** que dejó corriendo Bauty sobre los renders (¿pierde también en
   otras cámaras, o ahí aguanta mejor?). Anotar la tabla en la BITÁCORA.
2. **Fuente `gtsynth` en `align_data.py`** (sección 3.1), con su test de campo exacto.
3. **Caché del clásico** por cámara (sección 2.1).
4. **v3 en `alignnet.py` / `align_train.py`:** entrada corregida por el clásico, objetivo
   `R = M⁻¹ ∘ F`, salida en cero, cabeza σ, pérdida de 2.2 y elección por profundidad.
   Primero una corrida corta con `--limite` para medir el tiempo por época en la Mac.
5. **Modo `clasico+red-v3`** en `align_apply.py` (campo por cámara y peso `w`), y su opción en
   `gtsynth evaluar --alinear`.
6. **Evaluar en val** (sección 5) y decidir con la sección 6.
7. Si pasa: **capturas reales con la ELP** (GT C, en unos días) para confirmar que lo aprendido
   en render sirve en la cámara de verdad. Recién ahí, el test sintético.

Todo esto está en el área de Bauty (`src/visionx/depth/**`). Del lado de Tomás: los renders, las
ayudas de `gtsynth` (`nominal`, `geometria_nominal`, `recortar`, `modelo`) y, después, la
confianza en la voz.

## 5. Cómo se evalúa

Mismo protocolo que `gtsynth evaluar` del PR #54: CREStereo preciso, error de distancia por
objeto (0,5–4 m), con `--vistas-por-camara 6 --distancia-conocida`. Para cada variante de val y
cada sistema:

| Sistema | Qué es |
|---|---|
| `ninguna` | Par tal como llegó |
| `clasico` | Clásico con lente y regla: **la referencia a ganar** |
| `red` (v2) | Para ver cuánto se mejoró |
| `clasico+red-v3` | La v3 con el peso `w` |
| Oráculo | Campo exacto: el techo |

Informar la mediana y el P90 del error relativo (%) por nivel de hermanas (×0,25, ×0,5, ×1) y
por perfil, más la calibración de σ (sección 2.3).

Las hermanas se piden con `--variante hermana-<f>` (`gtsynth/variantes.py`). Hoy se arman
sobre `elp` (sin margen), así que la desalineación deja un borde inválido. Para la v3 conviene
armarlas sobre `elp-margen` y recortar con `v.recortar`: es un cambio chico en
`derecha_hermana` (área de Tomás, `gtsynth`) que se puede hacer junto con el paso 5.

## 6. Criterios para adoptarla

La v3 reemplaza al clásico solo si, **en val**:
1. En `elp-margen`, la mediana es menor o igual que la del clásico en los tres niveles, y el P90
   no es peor en más de 1 punto.
2. En los 6 perfiles, no es peor que el clásico en ninguno, y es mejor en al menos 3.
3. En `elp-arriba-*` / `elp-adelante-*`, detecta el corrimiento real: el error baja respecto del
   clásico.
4. La confianza está calibrada: entre 53 % y 73 % de errores con `|e| ≤ σ`.

Si no los cumple, se documenta como resultado negativo, con la tabla, y el sistema sigue con el
clásico. **Eso también es un resultado válido para el informe**: el clásico con lente y
distancia conocida ya queda cerca del oráculo en los niveles chicos.

## 7. Más escenas: datasets candidatos

Para esta red, la variedad que más importa es de **cámaras**, y esa se genera sin límite con
Replica. La de **escenas** importa para no aprender de memoria 8 departamentos y para texturas
difíciles. Si la evaluación de val muestra que falta, estas son las opciones que funcionan con el
mismo Habitat del render:

| Dataset | Escenas | Tamaño | Acceso | A favor | En contra |
|---|---:|---:|---|---|---|
| Replica (ya está) | 18 | ~34 GB | Libre | Escaneo real de alta calidad | Pocas escenas |
| **HSSD** (`hssd/hssd-hab`) | 211 en el paper; 161 públicas | 12,1 GB | Hugging Face: aceptar la licencia CC BY-NC 4.0 y usar un token | Sin espera, liviano; geometría limpia, Z exacta | Sintético (modelos 3D, menos fotorrealista); hay que cargarlo con su `scene_dataset_config` |
| **HM3D** | 1000 | Grande (empezar por `hm3d_minival_v0.2`, 10 escenas) | Formulario de Matterport (uso académico no comercial), aprobación en ~24 h, token de la API | Texturas reales, mucha variedad | Huecos y texturas estiradas del escaneo: la Z del mesh falla en objetos finos o reflejos |
| Matterport3D, Gibson | 90 / ~570 | Grande | Acuerdo firmado (MP3D usa un script de Python 2) | — | Más trámite, misma familia que HM3D |

**Recomendación:** HSSD primero (sin espera) y pedir HM3D en paralelo, usando solo el `minival`
hasta saber si hace falta más. Ninguno se usa en test: son para entrenar o para una validación
extra.

**Cómo se pide HM3D:**
1. Llenar el formulario de
   [matterport.com/habitat-matterport-3d-research-dataset](https://matterport.com/habitat-matterport-3d-research-dataset)
   (uso académico, no comercial).
2. Esperar el mail de aprobación, ~24 h según los mantenedores de Habitat.
3. En my.matterport.com, en la configuración de herramientas para desarrolladores, generar un
   token de la API. El secreto se muestra **una sola vez**: guardarlo fuera del repo.
4. En WSL, entorno `habitat`:

   ```bash
   python -m habitat_sim.utils.datasets_download --username <token-id> --password <token-secreto> \
     --uids hm3d_minival_v0.2 --data-path /mnt/d/datasets
   ```

**HSSD:** aceptar la licencia en
[huggingface.co/datasets/hssd/hssd-hab](https://huggingface.co/datasets/hssd/hssd-hab). Después,
`git lfs install && git clone https://huggingface.co/datasets/hssd/hssd-hab` con un token de
lectura de Hugging Face, o `datasets_download --uids hssd-hab`. `render_tanda.py` hoy carga el
`.ply` de Replica directo: para HSSD hay que pasarle el `scene_dataset_config`.

Fuentes: [DATASETS.md de habitat-sim](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md),
[HM3D](https://aihabitat.org/datasets/hm3d/), [hssd-hab](https://huggingface.co/datasets/hssd/hssd-hab),
[issue de habitat-sim sobre la aprobación](https://github.com/facebookresearch/habitat-sim/issues/2330).
