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
| **v2 + movimiento** (22/11) | la geometría mide la focal mientras la persona gira (60 cuadros) | — | **0,9 / 3,5** | — |

El sistema completo, error mediano en distancia por recorrido (promedio de las 9 cámaras):

| Cámara que supone el sistema | Error medio | Mediana |
|---|---:|---:|
| Calibrada (la ideal) | 5,2 % | 3,1 % |
| Una ELP, sin red | 9,8 % | 8,6 % |
| La red v2 | 8,7 % | 6,6 % |
| La red v2 + movimiento | *corriendo* | |

## Lo que se aprendió en cada paso

- **v1 → v2.** El zoom mejora val, pero en gt-video la mejora es chica y el error depende de la
  escena: juntar cuadros del mismo cuarto no ayuda (sesgo por escena).
- **8 escenas en vez de 2** (22/11). La v2 anda bien en Replica y mal en las casas de HSSD de
  val (15–55 %), aunque se entrenó con HSSD. Con 2 escenas de Replica no se veía.
- **Movimiento** (22/11). La geometría mide la focal con 0,9 % de mediana, sin entrenar nada. La
  red sigue aportando el centro, el respaldo y la duda. Medir cuánto aporta como punto de
  partida: autocalibración arrancando de 90° (*corriendo*).

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
