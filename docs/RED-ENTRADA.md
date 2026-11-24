# Red de entrada: historia de mejora

La red de entrada (E2 de [PLUG-AND-PLAY](PLUG-AND-PLAY.md), código de Bauty en
`src/visionx/depth/camara_net/`) mira una cámara estéreo desconocida y estima su focal, centro y
distorsión, cada uno con su duda. Este documento junta **todas las versiones medidas con el mismo
protocolo**, para que se vea qué aportó cada paso. El detalle de cada corrida está en la
[BITACORA](BITACORA.md).

## Para qué sirve la red (y qué cuenta como «útil»)

1. **Estimación instantánea.** Al enchufar la cámara, antes de que la persona se mueva, la red da
   una focal. Sin ella, el sistema supone 90°.
2. **Punto de partida y respaldo de la autocalibración por movimiento.** Si la persona no gira o
   el lugar tiene poca textura, la estimación de la red es la única que hay.
3. **Centro y distorsión.** El movimiento mide la focal, pero no el centro ni la lente.

Meta de E2: focal ≤ 2 % de mediana, con la duda bien puesta (cobertura del 90 %). La meta del
sistema completo (E3): error mediano en distancia ≤ 5 %, cerca de la cámara calibrada.

## Protocolo fijo

- **Val**: 840 vistas de val de Replica y HSSD (`camara-sorteada`, `hssd-camara-sorteada`), sin
  lente.
- **gt-video**: las 9 cámaras de evaluación (6 de la expo y 3 genéricas) × 8 escenas de val (4
  de Replica, 4 de HSSD), 90 cuadros cada una: 6480 cuadros. Nunca se usa para entrenar.
- Métricas de la focal: error mediano y P90 por cuadro (`|f estimada / f real − 1|`), y
  cobertura del intervalo del 90 %.
- Sistema completo (E3): `gtsynth sin-calibrar` con otra unidad, clásico y un objeto a distancia
  conocida, error mediano en distancia por recorrido.

## Versiones

Error de focal por cuadro (mediana / P90), en %:

| Versión | Qué cambió | Val | gt-video, 8 escenas | Replica (4) / HSSD (4), todos los cuadros |
|---|---|---:|---:|---|
| Constante | siempre la cámara media del entrenamiento | 31–33 | — | — |
| v1 (Bauty, 21/11) | ResNet-18, 2 ojos, Laplace + reproyección | 9,9 / 27 | 13,5 / 41 | bien / 11–63 |
| v2 (Bauty, 21/11) | + zoom y `camara-sorteada-ojo` | 9,0 / 28 | 12,7 / 40 | 0,2–16 / 4–55 |
| v3-auto (Bauty, 22/11) | v2 + 10 épocas con las etiquetas de la geometría (duda ≤ 3 %) | 9,2 / 27 | 11,9 / 37 | por cuadro: Replica 8,0 / 22, HSSD 17,6 / 54 |
| v3 control: real | lo mismo con la focal verdadera | 9,1 / 27 | 12,3 / 37 | por cuadro: Replica 7,8 / 22, HSSD 18,5 / 53 |
| v3 control: sin | v2 + 10 épocas sin etiquetas nuevas | 9,4 / 28 | 12,6 / 40 | por cuadro: Replica 8,1 / 25, HSSD 20,0 / 55 |
| v3-auto5 | como v3-auto, con las etiquetas de duda ≤ 5 % (1,7 veces más) | 9,0 / 25 | 11,7 / 38 | por cuadro: Replica 7,4 / 21, HSSD 18,7 / 53 |
| **v2 + movimiento** (22/11) | la geometría mide la focal mientras la persona gira (60 cuadros) | — | **0,9 / 3,5** | — |

El sistema completo, error mediano en distancia por recorrido (promedio de las 9 cámaras):

| Cámara que supone el sistema | Qué corrige el objeto | Error medio | Mediana |
|---|---|---:|---:|
| Calibrada (la ideal) | corrimiento | 5,2 % | 3,1 % |
| Una ELP, sin red | escala | 9,8 % | 8,6 % |
| La red v2 | escala | 8,7 % | 6,6 % |
| La red v2 + movimiento | escala | 8,7 % | 6,7 % |
| La red v2 + separación de la hoja de datos | corrimiento | 13,5 % | 6,1 % |
| **La red v2 + movimiento + separación** | **corrimiento** | **5,4 %** | **3,4 %** |

Con la focal medida y la separación conocida, la escala ya está y el objeto corrige el
corrimiento de la unidad, como en la calibrada: **+0,2 puntos sobre la calibrada** (diferencia
mediana por recorrido +0,08). Con la focal de la red sola el mismo camino da 13,5 %.

## Lo que se aprendió en cada paso

- **v1 → v2.** El zoom mejora val, pero en gt-video la mejora es chica y el error depende de la
  escena: juntar cuadros del mismo cuarto no ayuda (sesgo por escena).
- **8 escenas en vez de 2** (22/11). La v2 anda bien en Replica y mal en las casas de HSSD de
  val (15–55 %), aunque se entrenó con HSSD. Con 2 escenas de Replica no se veía.
- **Movimiento** (22/11). La geometría mide la focal con 0,9 % de mediana, sin entrenar nada.
  Arrancando de 90° sin la red, falla en todo el gran angular de 130°: la red es el seguro
  contra cámaras raras, y aporta el centro y la duda.
- **Sistema completo** (22/11). Red + movimiento + separación + objeto queda a +0,2 puntos de
  la cámara calibrada. Clave: con la escala resuelta, el objeto tiene que corregir el
  corrimiento, no la escala.
- **v3-auto** (22/11). Las etiquetas de la geometría ayudan tanto como las verdaderas, y
  entrenar más sin ellas no ayuda: `gt-video` 12,7 → 11,9 %, HSSD 20 → 18 %. Sumar las de
  duda ≤ 5 % no cambia nada. El techo con estos datos simulados es bajo: el salto tendría
  que venir del video real.

## Próximo paso: que la geometría entrene a la red (v3-auto)

Idea: cada vez que alguien gira una cámara, la autocalibración da una focal muy buena. Esa
focal es una **etiqueta gratis**: sirve para reentrenar la red con video que nadie calibró. Una
red así acierta sin necesidad de moverse justo donde hoy falla.

1. **Etiquetas** (`gtsynth autoetiquetas`). Sobre `hssd-recorridos-sorteada-b` (122 casas de
   train, 366 recorridos de 30 cuadros, una cámara sorteada por recorrido, que la v2 nunca vio),
   red + movimiento etiqueta cada recorrido sin mirar la cámara real. Solo se usan las etiquetas
   con duda ≤ 3 %. La cámara real se guarda al lado para medir qué tan buenas son.
2. **Reentrenar** la v2 con esas etiquetas, mezcladas con los datos de siempre para que no
   olvide.
3. **Control.** El mismo reentrenamiento con la cámara real como etiqueta (el techo). Si las
   etiquetas de la geometría llegan cerca del techo, la historia se sostiene: la red mejora sin
   que nadie calibre nada.
4. **Con video real.** Recorridos grabados con la ELP en casa y con la D435i en la facultad,
   etiquetados por el movimiento, sin tablero.

Cada paso entra en la tabla de versiones con el mismo protocolo.

### Etiquetas generadas (22/11, PC de Tomás)

`gtsynth autoetiquetas` sobre los 366 recorridos de `hssd-recorridos-sorteada-b` (30 cuadros
cada uno, cámara sorteada, casas de train). La red v2 da el punto de partida; el movimiento mide
la focal; nunca se mira la cámara real (que se guarda al lado solo para medir):

| | Recorridos | Cuadros | Error de la etiqueta (mediana / P90) | La red sola, en esos mismos |
|---|---:|---:|---:|---:|
| Todas | 366 | 10 980 | 3,5 % / 20 % | 10,1 % |
| **Duda ≤ 3 %** | **97** | **2910** | **0,7 % / 2,6 %** | 9,7 % |
| Duda ≤ 5 % | 166 | 4980 | 1,2 % / 5,5 % | 9,5 % |

La duda separa bien: las aceptadas tienen 14 veces menos error que la red, y el filtro no se
queda solo con las fáciles (la red erraba lo mismo en esas). Solo pasa el 26 % porque estos
recorridos tienen 30 cuadros, la mitad de lo que usa la autocalibración en la app.

Archivos en el Drive `VisionX-Personal/autoetiquetas/`:
- `b-vistas.csv` (duda ≤ 3 %) y `b-vistas-5pct.csv` (≤ 5 %): una fila por cuadro.
  - `vista`: ruta relativa a `gt-sintetico-elp/`.
  - `log_f`: la etiqueta, igual que `camara_net`: `log(fx / ancho)` de la imagen nominal.
  - `sigma_log_f`.
  - `fx_real`: solo para el control.
- `b.csv` / `b.json`: por recorrido, con la calidad.

### Entrenar la v3-auto (Bauty, en la Mac): comandos exactos

El código ya está en `main` (`camara_net entrenar --etiquetas/--desde/--etiqueta-real`),
probado de punta a punta en la PC con pocas imágenes. Solo hace falta correrlo.

**1. Preparar** (una vez):
1. `git pull` de `main`.
2. En `data/gt-sintetico-elp/`: `camara-sorteada`, `camara-sorteada-ojo`,
   `hssd-camara-sorteada`, `hssd-recorridos-sorteada` (lo de la v2) y
   **`hssd-recorridos-sorteada-b`** (Drive `gt-sintetico-elp/`, 37 GB).
3. `outputs/camara_net/e2-v2/mejor.pt`, del Drive `pesos/camara-net/e2-v2/`.
4. `outputs/autoetiquetas/b-vistas.csv` (y `b-vistas-5pct.csv`), del Drive `autoetiquetas/`.
5. `unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN`.

**2. Tres corridas** (en este orden; ~1 h cada una en la Mac, estimado por la v2):

```bash
COMUN="--variantes camara-sorteada camara-sorteada-ojo hssd-camara-sorteada --zoom 0.55 1.2 \
  --desde outputs/camara_net/e2-v2/mejor.pt --lr 1e-4 --lr-tronco 3e-5 --calentamiento 100 --epocas 10"

# a) v3-auto: la v2 + las etiquetas de la geometría (lo que queremos probar)
uv run --extra ml python -m visionx.depth.camara_net entrenar $COMUN \
  --etiquetas outputs/autoetiquetas/b-vistas.csv --salida outputs/camara_net/e3-auto

# b) control «techo»: los mismos cuadros con la focal real
uv run --extra ml python -m visionx.depth.camara_net entrenar $COMUN \
  --etiquetas outputs/autoetiquetas/b-vistas.csv --etiqueta-real --salida outputs/camara_net/e3-real

# c) control «más épocas»: la v2 diez épocas más, sin etiquetas nuevas
uv run --extra ml python -m visionx.depth.camara_net entrenar $COMUN --salida outputs/camara_net/e3-sin
```

- a) contra c) dice cuánto aportan las etiquetas, más allá de entrenar más.
- a) contra b) dice cuánto se pierde por usar la geometría en vez de la verdad.
- Si sobra tiempo: a) con `b-vistas-5pct.csv --sigma-max 0.05` (1,7 veces más cuadros, más
  ruidosos). Hecho el 22/11 (`e3-auto5`): queda como a), no suma.

**3. El factor de b en el checkpoint.** El de la v2 era 1,79: el que lleva la cobertura de val al
90 %. Medirlo igual en cada versión y guardarlo:

```python
import torch
ck = torch.load("outputs/camara_net/e3-auto/mejor.pt", weights_only=False)
ck["factor_b"] = 1.79  # el medido en val para ESTA versión
torch.save(ck, "outputs/camara_net/e3-auto/mejor.pt")
```

La pantalla «Cámara nueva» (`app/estimador_red.py`) lo lee sola. Sin `factor_b` usa 1,79 × 2.

**4. Subir y avisar.**
1. `mejor.pt`, `meta.json` y `curvas.csv` de cada corrida al Drive `pesos/camara-net/<corrida>/`.
2. Un renglón en la BITACORA con la val de cada una.
3. Avisar a Tomás: él evalúa las tres en la PC el mismo día, con las 8 escenas × 9 cámaras + E3,
   y completa la tabla de versiones de arriba.

### La U-Net con cámaras variadas (Bauty): dos caminos

Medido el 21/11: la U-Net `v2-s0` nunca se confía de más, pero el factor que lleva su b al 90 %
crece con `f·B` de la cámara (0,24 con 34 px·m, 0,70 con 161). Ajuste en las 9 cámaras:
**factor ≈ 0,046 · (f·B)^0,51**, con ±0,06 de error.

1. **Rápido (recomendado para la entrega):** sin reentrenar, multiplicar la b por esa función de
   `f·B` en el estimador. Es una línea en `estimator.py`, y `f·B` sale del perfil o de la
   «Cámara nueva». Ojo: se ajustó con las mismas 9 cámaras; para confirmarlo hay que medirlo en
   otras (lo hace Tomás en la PC con `gtsynth evaluar --geometria vista`).
2. **Completo:** reentrenar con `cache-crestereo` (Drive, 43 GB; disparidades de CREStereo de
   `camara-sorteada`, `hssd-camara-sorteada` y `hssd-recorridos-sorteada`, formato en
   `gtsynth/disparidades.py`). Hace falta un adaptador nuevo: hoy `inc-entrenar` lee su propio
   caché de InStereo2K (`uncertainty_data.py`), no este. Es más trabajo y riesgo antes del 4/12.
