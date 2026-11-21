# Estado del proyecto al 21/11 y camino a la entrega

Análisis de la noche del 20 al 21/11 (Claude, a pedido de Tomás). Junta lo que está hecho, lo
que se midió, los riesgos y el orden de lo que falta. La evidencia de cada número está en la
[BITACORA](BITACORA.md); el diseño, en [PLUG-AND-PLAY](PLUG-AND-PLAY.md); las tareas de Bauty,
en [TAREAS-BAUTY](TAREAS-BAUTY.md).

## 1. En cinco líneas

1. **El sistema funciona de punta a punta**: los 4 modos (buscar, distancia, leer, describir)
   con voz, CREStereo en dos modos, ubicación por objeto con nivel de certeza y Claude solo para
   lo abierto.
2. **El aporte del TP quedó definido y medido**: plug and play con cualquier cámara estéreo de un
   cuerpo más incertidumbre honesta. La línea de base (E0) muestra el problema con números:
   conociendo la cámara el error es 3–8 %; sin conocerla, 33–78 % (E0 en 8 escenas, 21/11).
3. **Lo que falta es la red de entrada** (estimar la cámara), que es la pieza central y todavía
   no está entrenada. Los datos para entrenarla y evaluarla ya están.
4. **La PC de Tomás pasó a ser un equipo de cómputo de verdad**: CREStereo en la Radeon con ONNX +
   DirectML (18 veces más rápido que en CPU, a la par de la Mac) y Habitat en la GPU (3,4 veces).
5. **La demo de la expo está casi lista**: mosaico de 6 cámaras, pantalla «Cámara nueva», stand
   con texturas y métricas. Falta probarla con la cámara de la facultad.

## 2. Estado por parte

| Parte | Estado | Evidencia |
|---|---|---|
| CREStereo (continuo y preciso) | Hecho. Preciso ≤ 1,1 % con cinta (ELP) | BITACORA, M9 |
| CREStereo en GPU sin CUDA | Hecho (ONNX + DirectML, PR #70) | 0,12 s continuo, 0,51 s preciso |
| Alineación clásica | Hecha. Con distancia conocida deja 2,1–4,4 % contra 0,41 % del oráculo (E1) | E1, 20/11 |
| Distancia conocida | Arreglada para focales largas (PR #71) | D455: de 30 % a corregido |
| Perfiles de cámaras y reconocimiento USB | Hecho (6 cámaras) | PR #65 |
| Pantalla «Cámara nueva» | Hecha, con voz y métricas | PR #65, #67, #73 |
| RealSense (D435/D435i/D455) | Código hecho; **falta probar con la cámara** | PR #66 |
| Red de entrada (estimar la cámara) | **Por hacer** (Bauty, E2) | TAREAS-BAUTY |
| U-Net de incertidumbre | Hecha para la ELP; **falta reentrenar con cámaras variadas y recalibrar** | INCERTIDUMBRE-Y-GT |
| Detección | YOLO + YOLOE (215 clases) con selector | PR #58 |
| Voz y modos | Hechos; respuestas cortas con «¿Querés más detalle?» | PR #56 |
| Stand de la expo | Mosaico, láminas, «Cámara nueva», stand con texturas | PR #63, #68, #73, #76 |

## 3. Números clave

**E0, enchufar sin calibrar** (CREStereo continuo, **8 escenas de val**, 21/11; antes 2).
Error mediano de Z, promedio de las 8 escenas:

| Cámara | Calibrada | Sin calibrar (supone ELP) | Sin calibrar + clásico + distancia conocida |
|---|---|---|---|
| ELP (control) | 6,3 % | 39,6 % | 6,1 % |
| ZED 2 | 2,7 % | 57,0 % | 56,0 % |
| ZED Mini | 3,5 % | 32,7 % | 21,7 % |
| OAK-D | 3,5 % | 61,2 % | 59,8 % |
| RealSense D435 | 3,8 % | 44,7 % | 31,3 % |
| RealSense D455 | 7,0 % | 61,9 % | 60,8 % |
| Angosta 60° | 6,7 % | 65,2 % | 61,0 % |
| Gran angular 130° | 8,4 % | 76,1 % | 22,5 % |
| Base larga 20 cm | 5,0 % | 78,1 % | 77,4 % |

Con 8 escenas la calibrada sube (las casas de HSSD tienen paredes lejanas y poca textura). El
clásico solo rescata las cámaras parecidas a la ELP; con la ZED 2, la OAK-D, la D455, la de 60° y
la de 20 cm no corrige nada en ninguna escena: el corrimiento que haría falta pasa el tope de
seguridad, porque el error es de escala y no de unos píxeles.

**E1, el clásico dentro de una cámara conocida** (modo preciso, val, 3057 objetos):

| Otra unidad | Sin corregir | Clásico + distancia conocida | Oráculo |
|---|---|---|---|
| ×0,25 | 10,6 % | 2,1 % | 0,41 % |
| ×0,5 | 19,9 % | 2,7 % | 0,41 % |
| ×1 | 33,5 % | 4,4 % | 0,41 % |

**Lectura conjunta:** el mayor valor de la red está en **estimar cámaras desconocidas**
(E0: de 33–78 % a lo que logre la red). Dentro de una cámara conocida queda un margen menor pero
real (4,4 % contra 0,41 %, que a 3–4 m llega al 7 %).

## 4. Datos disponibles

| Datos | Para qué | Dónde |
|---|---|---|
| GT sintético Replica + HSSD (22 variantes, ~97 GB) | Base de todo | D: y Drive |
| `camara-sorteada` + `hssd-camara-sorteada` (5700 vistas) | Red de entrada, un cuadro | D: y Drive |
| `hssd-recorridos-sorteada` (153 casas, 4590 vistas) | Red de entrada, varios cuadros; U-Net | D: y Drive |
| `hssd-recorridos-sorteada-b` (3 recorridos por casa, noche del 21/11) | Más entrenamiento | D: (no entra en el Drive) |
| `gt-video`: 9 cámaras de evaluación en 8 casas val (2 → 8 esta noche) | E2–E4 en cámaras no vistas | D: (las 2 primeras, también en el Drive) |
| `cache-crestereo`: disparidades precalculadas | Reentrenar la U-Net sin CREStereo | D: |

## 5. Riesgos, de mayor a menor

1. **La red de entrada no llega o no estima bien.** Es el aporte central y está por empezar.
   *Mitigación:* prototipo de un cuadro con los datos que ya hay (E2) antes de lo de varios
   cuadros; criterio de corte claro (campo ±3 %). Si no llega, el plug and play se presenta con
   lo que hay: el flujo, el clásico, la escala por referencia y la incertidumbre, y la red como
   resultado parcial con sus números.
2. **La incertidumbre está descalibrada fuera de la ELP.** Con cámaras variadas no se midió.
   *Mitigación:* reentrenar con las disparidades precalculadas; pedir el checkpoint actual para
   medirlo en las cámaras nuevas en la PC de Tomás.
3. **La RealSense no se probó con la cámara.** Puede haber sorpresas (permisos USB en la Mac,
   formato, proyector). *Mitigación:* probarla en la facultad apenas se pueda, con
   `--fuente realsense` y `camara-nueva --fuente realsense`.
4. **Fechas inconsistentes en los documentos.** ROADMAP dice entrega el 10/12; lo acordado con el
   profesor es entrega el 7/12 y expo el 12/12. *Mitigación:* corregir el ROADMAP (lo decide
   Tomás).
5. **Infraestructura de la PC.** Con 32 GB, 6 renders + CREStereo agotaron la memoria dos veces.
   *Mitigación:* como mucho 3 renders y un CREStereo a la vez (así corrió esta noche).
   `onnxruntime-directml` no está en `uv.lock`: un `uv sync` lo saca y CREStereo vuelve a la CPU.
6. **Pasar datos entre máquinas.** El Drive (200 GB) no alcanza para lo de esta noche (~70 GB).
   *Mitigación:* disco externo, o que Bauty baje lo prioritario y se libere el Drive.

## 6. Camino a la entrega

| Fecha | Qué | Quién |
|---|---|---|
| 21–22/11 | Bauty: pull, revisión del diseño, PR pendientes de su área (#70, #75). Pasar los datos de esta noche | Bauty, Tomás |
| 22–26/11 | E2: prototipo de la red de entrada (un cuadro) | Bauty (Mac) |
| 22–26/11 | Prueba de la D435i en la facultad (fuente, «Cámara nueva», cinta) | Tomás |
| 26–30/11 | Red de entrada con varios cuadros; U-Net con cámaras variadas | Bauty |
| 26–30/11 | Evaluar cada checkpoint en la PC (E2, E3, E4 en cámaras no vistas) | Tomás / Claude |
| 1–4/12 | Integrar la red al contrato `EstimadorCamara`; E5 con la D435i | Todos |
| 4/12 | Congelar | — |
| 7/12 | Entrega (informe IEEE ≤ 8 páginas, póster) | Todos |
| 12/12 | Expo | Todos |

## 7. Recomendaciones

- **Priorizar E2 de un cuadro** por sobre cualquier otra cosa de redes: es la respuesta a «¿se
  puede estimar una cámara desconocida?» y define el resto.
- **No reentrenar la v3 densa:** con E1 ya medido, el margen dentro de una cámara conocida es
  chico frente al de estimar cámaras nuevas.
- **Usar la PC para evaluar** todo lo que entrene la Mac, el mismo día: CREStereo ahí ya no es
  el cuello de botella.
- **Ensayar la expo** con la Mac de Bauty: `ejecutar --camara`, la pestaña «Cámaras» (tecla
  `k`) y «Cámara nueva» con la D435i.
