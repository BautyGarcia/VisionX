# GT-sintético-ELP: cómo se genera, se verifica y se usa

Conjunto **A** de [INCERTIDUMBRE-Y-GT](INCERTIDUMBRE-Y-GT.md#2-mini-dataset-con-ground-truth):
pares estéreo de interiores de [Replica](https://github.com/facebookresearch/Replica-Dataset)
renderizados con la geometría rectificada de la ELP, con la distancia exacta y el objeto de cada
píxel. Sirve para medir la distancia por objeto con cientos de casos, calibrar la b de la U-Net y
probar la decisión (a) con cámaras hermanas y una distinta.

## Estado (19/11, noche): generado y verificado

| | Valor |
|---|---|
| Vistas | 1440: 720 `elp` + 720 `distinta` (18 escenas × 40), 65 min en la PC de escritorio (WSL + RX 5700 XT) |
| Tamaño | 12 GB; copia en Drive `VisionX-artefactos/gt-sintetico-elp/` (con `objetos-elp.csv`) |
| Verificación | 108/108 vistas (3 por escena y cámara): con la Z exacta el error es 2,9 niveles de gris (ELP) y 3,5 (distinta), contra 13,3 y 23,9 con ±10 % de disparidad |
| Objetos `elp` | 15 088 visibles (≥ 400 px, ≤ 6 m); con clase del detector y a 0,5–4 m: train 2413, val 1115, test 1990 |
| Por rango | 0,5–1 m 1442; 1–2 m 5167; 2–3 m 4090; 3–4 m 2635; > 4 m 1754 |

**Ojo con `pose.altura_m` de `vista.json`:** es la altura sobre el punto de la navmesh de
Habitat, que queda ~18 cm por encima del piso real (en las vistas con piso etiquetado, el plano
del piso da 1,44–1,47 m donde la navmesh dice 1,26–1,29 m). La inclinación (`pitch_rad`) sí es
exacta. No afecta las distancias de los objetos (salen de la Z del render).

Límites: Replica es escaneo real pero estático y sin gente; algunos departamentos FRL tienen
ventanas abiertas o huecos del escaneo (sin geometría, Z = 0: mediana 0,1 % de la vista, hasta
~20 % en pocas vistas) que la tabla de objetos ignora. Las texturas son color por vértice de
`mesh_semantic.ply` (Habitat de conda no dibuja las PTex de `mesh.ply`): fotorrealistas a la
distancia de uso, algo más suaves de cerca. El render es pinhole ideal: no tiene ruido, desenfoque
ni el borde inválido de la rectificación real.

## Tanda del issue #53 (19–20/11): otras cámaras, margen y casos difíciles

Pedido de Bauty para que la red de alineación sirva con cualquier cámara estéreo
([RED-ALINEACION-V3](RED-ALINEACION-V3.md)). Se generó de noche con
`scripts/gt_sintetico/tanda53.sh`, que por cada tanda renderiza (`render_tanda.py`), verifica
20 vistas con `gtsynth verificar` y copia al Drive `VisionX-Personal/gt-sintetico-elp/`
(log: `outputs/gtsynth/tanda53.log`).

| Tanda | Variante | Escenas | Vistas | Cómo |
|---|---|---|---:|---|
| E1 | `elp-margen` | val | 160 | ELP con margen, poses de `elp` (el recorte nominal es idéntico a `elp`) |
| E1 | `elp-margen` | train | 2400 | ELP con margen a media resolución (960×540 nominal), 300 por escena |
| E2 | `angosta-60`, `base-corta-3`, `media-90`, `cuatro-tercios-90`, `base-larga-20`, `gran-angular-130` | val | 6 × 160 | Otras cámaras (campo en grados, base en cm), poses de `elp` |
| E3 | `camara-sorteada` | train | 2400 | Por vista: campo 55–130°, base 3–20 cm, 16:9 o 4:3 |
| E4 | `elp-arriba-1`, `elp-arriba-2`, `elp-adelante-2`, `elp-arriba-1-adelante-2` | val | 4 × 160 | Ojo derecho corrido 1–2 mm hacia arriba o adelante, poses de `elp` |
| E5 | `pose-altura`, `pose-poca-textura`, `pose-cerca` | val | 3 × 80 | 1,0 y 1,6 m con pitch −30..10°; ≥ 75 % de estructura; objeto a 0,3–0,5 m |
| E6 | `secuencias` | val | 320 | 10 recorridos de 8 cuadros por escena (pasos de 10 cm, ±3° de giro) |
| E7 | `ojo-de-pez` | val | 160 | Doble esfera de ~180°, sin margen: **solo para evaluar** (no es rectificado; no pasa por `verificar`) |

**Lienzo con margen.** Todas las variantes nuevas pinhole se renderizan en un lienzo 25 % más
grande (12,5 % por lado), con la misma focal y el centro en el medio, para poder desalinear el
ojo derecho sin bordes negros. `vista.json` guarda la cámara del lienzo en `camara`, y en
`nominal` el recorte de la cámara simulada (`ancho`, `alto`, `x0`, `y0`, `cx`, `cy`,
`hfov_deg`). En código: `v.nominal`, `v.geometria_nominal`, `v.recortar(img)` y `v.modelo`.
También guarda `perfil`, `ojo_derecho_m` (+y arriba, −z adelante), `tipo_pose` y `secuencia`.

Verificación previa (prueba en `room_1`, todas las variantes):
- consistencia del par con la Z exacta en todos los perfiles pinhole;
- el recorte de `elp-margen` es idéntico a `elp` (imagen con diferencia 0; Z igual salvo un
  0,001 % de píxeles en bordes);
- `camara-sorteada` sin vistas malas, después de corregir que el campo y la base se fijaban
  antes de la pose (`set_state` los reiniciaba).

Comando de una tanda (dentro de WSL):

```bash
~/bin/micromamba run -n habitat python /mnt/d/VisionX/scripts/gt_sintetico/render_tanda.py \
  --replica /mnt/d/datasets/replica --escenas frl_apartment_3 office_2 room_1 hotel_0 \
  --salida /mnt/d/VisionX/data/gt-sintetico-elp --variante media-90 --perfil media-90 \
  --poses-desde /mnt/d/VisionX/data/gt-sintetico-elp/elp
```

`--help` lista las opciones restantes: `--ojo-derecho DX DY DZ`, `--tipo-pose`, `--margen`,
`--ojo-de-pez`, `--vistas`. Las escenas `test` no se renderizaron para estas variantes: se hace
una sola vez, cuando la red esté cerrada en val.

## Qué hay en cada vista

`data/gt-sintetico-elp/<variante>/<escena>/<nnnn>/` (no se sube al repo; copia en Drive
`VisionX-artefactos/gt-sintetico-elp/`):

| Archivo | Contenido |
|---|---|
| `izq.png`, `der.png` | Par rectificado, RGB, 1920×1080 |
| `z.npz` | Z axial del ojo izquierdo en metros (`float32`, clave `z`) |
| `inst.png` | Id de instancia de Replica por píxel del ojo izquierdo (`uint16`, 0 = nada) |
| `vista.json` | Cámara usada, pose, objeto apuntado y tabla id → clase |

**Cámara `elp`:** fx = fy = 601,007 px (≈116° de campo horizontal), centro en el medio de la
imagen, separación 5,98 cm: la geometría de `configs/calib/elp-2026-11-11-corregida.yaml` (el
centro real difiere ~1–2 px; la vista guarda la geometría exacta usada, `ELP_RENDER` en código).
**Cámara `distinta`:** 90° de campo y 12 cm de separación. **Hermanas:** no se renderizan; se
arman al vuelo desalineando el ojo derecho de las vistas `elp` con el generador de etapa 2 de A
(`sample_sibling`, rangos × 0,5 o × 1).

Cada vista apunta a un objeto de la escena desde 0,5–4 m, a 1,2–1,4 m de altura (pecho), con un
desvío al azar de ±20° en horizontal y ±8° en vertical. Distancia de referencia de un objeto =
percentil 20 de Z en su parte visible, la misma definición que usa el ubicador.

**Partición por escena** (`visionx.gtsynth.PARTICION`): `train` 8 escenas (libre, p. ej. para
entrenar la red de entrada), `val` 4 (ajustar umbrales y el factor de b), `test` 6 (**congelado**
hasta la evaluación final).

## 1. Bajar Replica (una vez, ~34 GB)

Licencia: solo investigación o educación; no redistribuir fuera del equipo.

```bash
mkdir -p /d/datasets/replica/partes && cd /d/datasets/replica/partes
for p in a b c d e f g h i j k l m n o p q; do
  curl -sSL -C - --retry 5 -o replica_v1_0.tar.gz.parta$p \
    https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta$p
done
cat replica_v1_0.tar.gz.part?? | tar -xz -C /d/datasets/replica   # deja las 18 escenas en la raíz
curl -sSL -o additional_habitat_configs.zip http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip
unzip -qn additional_habitat_configs.zip -d /tmp/replica_cfg && cp -rn /tmp/replica_cfg/replica_v1/* /d/datasets/replica/
```

## 2. Entorno de Habitat-Sim (aparte del de VisionX)

Habitat-Sim no corre en Windows: en Windows se usa WSL (Ubuntu) con micromamba, sin `sudo`. En
la Mac o Linux, lo mismo con conda.

```bash
# dentro de WSL
mkdir -p ~/bin && curl -sSL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ~ bin/micromamba
export MAMBA_ROOT_PREFIX=~/mamba
~/bin/micromamba create -y -n habitat -c conda-forge -c aihabitat python=3.9 habitat-sim
~/bin/micromamba install -y -n habitat -c conda-forge libopengl   # faltaba libOpenGL.so.0
```

Con GPU AMD en WSL hay que dejar `gpu_device_id = -1` (el script ya lo hace): con 0, Magnum
busca una GPU CUDA y falla («unable to find CUDA device 0»).

La escena se carga desde `habitat/mesh_semantic.ply` **sin** `replica.scene_dataset_config.json`:
con esa config Habitat intenta dibujar `mesh.ply` con texturas PTex, que la build de conda no
trae, y todo sale negro. El aviso «Reloading asset … with different configuration» es esperado.

## 3. Renderizar

```bash
# dentro de WSL; rutas de Windows bajo /mnt/d
~/bin/micromamba run -n habitat python /mnt/d/VisionX/scripts/gt_sintetico/render_habitat.py \
  --replica /mnt/d/datasets/replica \
  --escenas apartment_0 apartment_1 apartment_2 frl_apartment_0 frl_apartment_1 frl_apartment_2 \
            frl_apartment_3 frl_apartment_4 frl_apartment_5 hotel_0 office_0 office_1 office_2 \
            office_3 office_4 room_0 room_1 room_2 \
  --vistas 40 --variantes elp distinta --semilla 0 \
  --salida /mnt/d/VisionX/data/gt-sintetico-elp
```

Misma semilla = mismas vistas. Las poses se sortean con `elp` y se reusan en `distinta`.

## 4. Verificar y resumir (entorno de VisionX)

```bash
uv run python -m visionx.gtsynth verificar data/gt-sintetico-elp --max 20   # Z coincide con el par
uv run python -m visionx.gtsynth resumen   data/gt-sintetico-elp            # vistas, objetos, clases
uv run python -m visionx.gtsynth objetos   data/gt-sintetico-elp --csv outputs/gtsynth/objetos.csv
```

`verificar` reproyecta el ojo derecho con la disparidad exacta y comprueba que gana contra no
moverlo y contra ±10 %: si falla, la convención de profundidad o de píxel del render está mal y
nada de lo que sigue vale.

## 5. Mostrarlo (stand y póster)

```bash
uv run python -m visionx.gtsynth visor data/gt-sintetico-elp            # pantalla: pasa sola
uv run python -m visionx.gtsynth visor data/gt-sintetico-elp --exportar outputs/gtsynth/laminas --n 12
```

Cada lámina: la vista con cada objeto pintado y su distancia exacta, la profundidad del render,
qué es el conjunto y el error de CREStereo por cámara (de `datasets/gtsynth/val-*.json`). Elige
vistas sin huecos de escaneo; n / p avanzan, f pantalla completa, q sale.

## 6. Evaluar la distancia por objeto

Cada objeto se mide con su **máscara exacta** (no con el detector), así el error es el de la
profundidad y el ubicador. El sistema siempre cree tener la geometría de la ELP.

```bash
uv run --extra ml python -m visionx.gtsynth evaluar data/gt-sintetico-elp --particion val \
  --variante elp --profundidad crestereo-preciso --csv outputs/gtsynth/val-elp.csv
# variantes: elp, hermana-0.5, hermana-1, distinta
```

Escribe una fila por objeto (CSV) y un resumen (`.json` al lado): error por rango (MAE, sesgo,
P95), políticas del ubicador y, si el estimador trae b, la cobertura del intervalo del 90 % con
la b tal cual (`cobertura_90_k1`) y el factor que la lleva a 90 % (`factor_b_para_90`). El factor
se ajusta **solo con val**; `test` se corre una vez, al final, con todo congelado.
