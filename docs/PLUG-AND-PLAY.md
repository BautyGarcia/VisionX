# Plug and play: cualquier cámara estéreo, sin calibrarla a mano

Diseño de la parte de redes del TP desde el 20/11. Reemplaza el objetivo de la red de entrada
de [RED-ALINEACION-V3](RED-ALINEACION-V3.md): la red ya no compite con el clásico corrigiendo
un campo denso, sino que hace lo que el clásico no puede. Decisión de Tomás; **falta que Bauty
lo revise** (la red es su área) antes de generar la tanda grande de datos.

## 1. Objetivo y alcance

> Enchufar una cámara estéreo que VisionX nunca vio, y que funcione: que mida distancias y
> que sepa cuánto creerle a cada medición.

- **Entra:** cámaras estéreo **de un solo cuerpo**, con los dos ojos sincronizados por hardware
  (ELP, Intel RealSense, Stereolabs ZED, Luxonis OAK-D…). Cualquier ángulo de 40° a 140°,
  separación de 3 a 20 cm, distintas resoluciones y lentes.
- **Queda afuera:**
  - dos webcams sueltas (por ejemplo dos Logitech C922): no están sincronizadas (hasta 33 ms
    entre ojos, que caminando crea disparidad falsa), tienen autofoco y la barra se mueve. A lo
    sumo, una prueba al final como trabajo futuro;
  - ojo de pez (más de ~150°).
- **Lo que aporta el TP:** la red de entrada (estimar una cámara desconocida) **y** la red de
  incertidumbre de la salida, que además de la duda de cada píxel suma la duda de la
  calibración. Juntas permiten decir: «funciona con cualquier cámara estéreo, y cuando no está
  seguro, lo dice».

## 2. Qué hay que averiguar de una cámara nueva

| # | Qué | ¿Se ve en las imágenes? | Quién lo resuelve |
|---|---|---|---|
| 1 | Formato: un USB con los dos ojos pegados o dos transmisiones (RealSense); cuál es cada ojo | Sí: con los ojos al revés las disparidades salen negativas | Código (dashboard) |
| 2 | **Focal, centro y distorsión** de cada ojo | **No por geometría estéreo**: con los ojos casi paralelos, cualquier focal explica igual los puntos emparejados. **Sí por el contenido**: líneas rectas que se curvan, puntos de fuga, tamaños típicos, y cómo cambia la imagen cuando la persona gira | **Red de entrada** (sección 4) |
| 3 | Giro entre los ojos (pitch, roll, yaw en parte), focal relativa | Sí: desalinea las filas | **Clásico** (ya anda: SIFT, RANSAC, Huber, regla) |
| 4 | **Escala** (focal × separación) y el yaw que no deja rastro | **No**: ningún método la saca solo de las imágenes | Una **referencia** (sección 5) |
| 5 | Cuánto creerle a todo lo anterior | — | **Red de incertidumbre** (sección 6) |

Por qué el clásico solo no alcanza: en todas las pruebas hasta ahora arrancó desde la
calibración nominal **de esa misma cámara** y solo corrigió diferencias chicas entre unidades.
Con una cámara desconocida no tiene de dónde arrancar. El experimento E0 (sección 8) lo mide.

## 3. El flujo de una cámara nueva

Es también la pantalla del dashboard («Cámara nueva», `python -m visionx.app camara-nueva`;
`--fuente sim:video-zed-mini` la muestra con una cámara simulada), guiada por voz:

1. **Conectar.** Detecta el formato, la resolución y cuál es cada ojo.
2. **¿La conozco?** Busca el nombre del dispositivo en una lista de perfiles (`configs/camaras/`, a crear). Si está («Reconocí:
   Intel RealSense D435i»), carga el perfil y salta al paso 6.
3. **«Mirá alrededor despacio unos segundos».** Junta unos 30 pares de cuadros distintos.
4. **Estimar la cámara.** La red de entrada da la focal, el centro y la distorsión de cada ojo,
   con su confianza. Hasta que esté entrenada, se usa un valor por defecto (90°) marcado como
   «sin estimar».
5. **Alinear.** El clásico arranca desde esa estimación y ajusta los giros. La pantalla muestra
   las filas de los dos ojos antes y después.
6. **Escala.** Si la separación es conocida (perfil, hoja de datos o la escribe quien la
   instala), listo. Si no: «Mostrame un objeto a una distancia conocida» (sección 5).
7. **Perfil.** Guarda focal, distorsión, giros, separación y **nivel de la calibración**
   («precisa», «aproximada»); queda para la próxima vez.
8. **Arranca VisionX** con los 4 modos. Mientras se usa, el clásico sigue revisando la
   alineación (si la cámara se golpea, se nota en las filas).

## 4. Red de entrada: estimar la cámara

**Antes** (v2/v3): un campo denso que corrige el ojo derecho encima del clásico. La v2 perdió
contra el clásico en profundidad (6,7 % contra 5,6 %; oráculo 0,75 %; hermana ×1, con distancia
conocida). **Ahora:** estimar los parámetros que el clásico no puede ver.

- **Entrada:** N pares (por ejemplo 8) de la cámara, a resolución fija (640 px de ancho, con
  el campo de visión intacto: se escala, no se recorta). Más el ancho y alto originales.
- **Salida, una sola cámara para los dos ojos** (revisión #85, punto 4): `log(fx / ancho)`,
  centro `(cx/ancho − 0,5, cy/alto − 0,5)` y distorsión radial `k1, k2`. Cada uno **con su
  incertidumbre** (media y `b` de una Laplace, igual que la U-Net). En una cámara de un solo
  cuerpo los dos lentes son del mismo modelo, y lo que difiere entre ojos (focal relativa,
  giros) ya lo estima el clásico. La salida por ojo vuelve cuando existan los renders con
  «ojos levemente distintos» (7.2).
- **Campo de visión = el del pinhole sin distorsión.** Si no, campo y `k1` se confunden (un
  barril fuerte «agranda» el campo aparente). El campo se calcula de la focal solo para
  mostrarlo en pantalla; lo que se mide es la focal (sección 8, E2).
- **Arquitectura sugerida:** un codificador por cuadro (CNN chica o un ViT preentrenado),
  promedio con atención sobre los N cuadros, cabeza que regresa los parámetros. Los dos ojos
  comparten pesos. Modelos publicados que estiman focal y distorsión desde una foto (por
  ejemplo GeoCalib, 2024) muestran que el problema se puede aprender; lo nuestro suma varios
  cuadros, los dos ojos y la confianza.
- **Pérdida:** NLL de Laplace sobre cada parámetro **más** una pérdida geométrica: el error de
  reproyección de una grilla de puntos con los parámetros estimados contra los reales (así, un
  error de focal que no cambia la imagen no se castiga de más).
- **Por qué la meta va en la focal y no en el campo:** con `f = W / (2·tan(h/2))`, un error en
  el campo pasa a la focal como `df/f ≈ dh / sin(h)`. Un 3 % de error en el campo es ~3,6 % de
  focal a 60°, ~4,7 % a 90° y ~7,5 % a 130°, y con la separación conocida ese error pasa casi
  entero a la distancia.
- **Cómo se usa con el clásico:** la red da el punto de partida; el clásico ajusta. Si el
  clásico encontró suficientes puntos y alineó bien las filas (residuo vertical < 0,5 px),
  manda el clásico en lo que él estima (giros, focal relativa); la red manda en lo que el
  clásico no ve (focal absoluta, distorsión). Si la confianza de la red es baja, el nivel de la
  calibración baja a «aproximada».

## 5. La escala

Ninguna red la puede sacar de las imágenes. Opciones, de más a menos precisa:

| Referencia | Cómo | Nivel |
|---|---|---|
| Separación conocida | Perfil de la cámara, hoja de datos, o la escribe quien la instala | precisa |
| Objeto a distancia conocida | `calibrar_distancia`: con un objeto hay **una** ecuación, así que corrige **una** cosa (ver abajo) | precisa |
| Dos objetos a distancias distintas | Fija la escala **y** el corrimiento a la vez | precisa |
| Objeto de tamaño conocido | Hoja A4, tarjeta (85,6 mm) | aproximada a precisa (a medir) |
| Piso a la altura del pecho | La altura se mide una vez; el plano del piso da la escala | a medir |
| Red monocular (solo para la escala) | Ajusta un número una vez, con muchos píxeles y cuadros | aproximada |

«Sin tocar nada» funciona con distancias **aproximadas**; cualquier paso extra las vuelve
**precisas**. Encaja con los tres niveles de la voz.

**Qué corrige el objeto a distancia conocida** (revisión #85 y BITACORA 21/11). En un par
rectificado `Z = f·B / d`:

- **Escala** (multiplicar la distancia, es decir, corregir `f·B`): cuando la cámara es
  **distinta o estimada** y lo que puede estar mal es la focal o la separación. Es lo que hace
  falta en el plug and play. Restar un corrimiento en ese caso solo acierta a la distancia del
  objeto: con la cámara `distinta` da 63–258 % de error.
- **Corrimiento** (restar unos píxeles de disparidad): cuando la cámara es **conocida** y lo que
  se corre es el yaw o el centro óptico de una unidad (hermanas). Es lo que mide E1.

Con dos objetos a distancias distintas se corrigen las dos. `gtsynth evaluar
--distancia-conocida escala|corrimiento` mide cada una; el `calibrar_distancia` del estimador
(Bauty) tiene que elegir escala cuando la cámara no está «precisa».

**Qué parte de la red importa según la referencia** (revisión #85, punto 2):

| Referencia | Qué tiene que acertar la red de entrada |
|---|---|
| Separación conocida (perfil, hoja de datos) | **La focal absoluta**: su error pasa directo a la distancia |
| Objeto a distancia conocida | Distorsión y centro, y un buen punto de partida para el clásico: el objeto fija `f·B` y un error de la focal sola casi no cambia la distancia |

## 6. Red de incertidumbre: sumar la duda de la calibración

Hoy la U-Net estima `b` por píxel (el error de disparidad). Con una cámara estimada hay un
segundo error, el mismo para toda la imagen: la escala (focal × separación) y el corrimiento.
Para una distancia `Z`:

```
(σ_Z / Z)² ≈ (Z · σ_d / (f·B))²  +  (Z · σ_Δ / (f·B))²  +  σ_escala²
```

- `σ_d` sale de la U-Net, como ahora. Con una Laplace, **`σ = √2 · b`** (no mezclar `b` con
  `σ`).
- `σ_Δ` es la duda del corrimiento de columnas, en px (lo que el clásico no pudo fijar). Crece
  con la distancia igual que el término de la U-Net.
- `σ_escala` sale de la red de entrada y de la referencia usada (sección 5): casi 0 con la
  separación conocida o con un objeto a distancia conocida corrigiendo la escala; el que dé la
  red monocular si es «aproximada».
- **`σ_Δ` y `σ_escala` son el mismo error para todos los píxeles de una cámara**: no bajan al
  juntar muchos píxeles de un objeto ni muchos cuadros (el término de la U-Net sí). Por eso una
  cámara «aproximada» no dice «precisa» aunque el objeto sea grande; la app ya pone ese tope
  (`camara_nivel`).
- Los niveles de la voz (precisa ≤ 10 %, aproximada ≤ 25 %, no segura) se calculan con el
  total. Una cámara mal estimada nunca dice «precisa».
- La U-Net hay que **reentrenarla con cámaras variadas** (hoy vio solo la geometría de la ELP)
  y recalibrar el factor de `b` **por nivel de calibración**: con la ELP calibrada duda de más
  (factor 0,28) y con hermanas corregidas por el clásico se queda justa (1,0–1,3), así que un
  solo factor no sirve (BITACORA 20/11, noche). La alternativa es darle a la U-Net el residuo
  vertical del clásico como entrada.

## 7. Datos

### 7.1 Lo que ya hay (en `D:\VisionX\data`, disco de la PC de escritorio)

- `gt-sintetico-elp/camara-sorteada` (Replica, 2400 train + 240 val) y `hssd-camara-sorteada`
  (3060): campo 55–130°, separación 3–20 cm, 16:9 o 4:3, **una cámara distinta por vista**.
- Hermanas, ojo derecho corrido, 6 perfiles fijos, secuencias.
- 8 recorridos de 90 cuadros (2 escenas × 4 cámaras) y, en curso, los de las 6 cámaras reales.

### 7.2 Lo que falta generar

Se renderiza **solo la geometría**; lo demás se aplica al entrenar, así cada imagen sirve con
muchos efectos distintos y no gasta disco.

| Render | Para qué |
|---|---|
| ~300 recorridos de 30 cuadros, cada uno con una cámara sorteada, en casas `train` de HSSD | La red de entrada aprende de varios cuadros y del giro de la persona |
| Campo 40–140°, 16:9, 16:10 y 4:3, de 640×480 a 1920×1080, centro corrido ±3 % | Cubrir las cámaras estéreo comunes |
| Ojos levemente distintos entre sí (focal ±2 %, centro ±1 %) | Unidades reales |
| Altura del pecho 1,0–1,5 m | Escala por el piso |

| Al entrenar (aumento) | Para qué |
|---|---|
| Distorsión radial `k1, k2` sorteada, aplicada sobre el lienzo con margen, **sin pliegue**: se descartan las combinaciones con `d/dr [r·(1 + k1·r² + k2·r⁴)] ≤ 0` hasta el radio de la esquina más un margen (a 130°, con `k1 = −0,35` el modelo se pliega en `r ≈ 1`, dentro de la imagen) | Lentes reales sin renderizar cada una |
| Ruido, desenfoque de movimiento, compresión JPEG, viñeteo | Sensores reales |
| Exposición y color distintos entre ojos; escala de grises (RealSense, OAK-D) | Cámaras con sensores infrarrojos o monocromo |

Partición por casa, como siempre (`hssd_particion.json`); el test sigue congelado.

### 7.3 Datos reales

- **Intel RealSense D435i** de la facultad: trae calibración de fábrica (se lee con
  `librealsense`): es la respuesta correcta para medir la red. Su sensor de movimiento da la
  gravedad (ayuda con pitch, roll y el piso).
- La ELP, con su calibración ChArUco.
- Las demás cámaras que haya en la facultad (Tomás pasa la lista).
- Para entrenar sin distancia real, con consistencia entre las dos imágenes: Holopix50k (pares
  de celulares distintos). Para evaluar: Middlebury 2014, que trae versiones con rectificación
  imperfecta.

## 8. Experimentos y banco de prueba

| Id | Qué | Para qué |
|---|---|---|
| **E0** ✔ | Cámaras distintas **arrancando desde la calibración de la ELP** (como quien enchufa sin calibrar), con y sin clásico, con distancia conocida | 21/11, 8 escenas de val × 9 cámaras: calibrada 2,7–8,4 %, sin calibrar 33–78 %, sin calibrar + clásico 22–77 % (sin contar la ELP). El clásico solo rescata las cámaras parecidas a la ELP (ZED Mini, D435, gran angular); con ZED 2, OAK-D, D455, 60° y 20 cm no corrige nada en ninguna escena |
| **E1** ✔ | Clásico + distancia conocida contra el oráculo, **modo preciso**, muchas hermanas, val | 21/11: oráculo 0,41 %; clásico + distancia conocida 2,1 / 2,7 / 4,4 % (×0,25 / ×0,5 / ×1); por objeto (Bauty, Mac): ×1 deja 4–13 % medio y uno de cada cuatro objetos con más de 10 % |
| E2 | Red de entrada sola en cámaras **que no vio**: error de **focal** (`log f`, en %), centro (px) y `k1`, más la cobertura del intervalo del 90 % | ¿Se puede estimar una cámara? **Meta: error mediano de la focal ≤ 2 %** (revisión #85) |
| E3 | Flujo completo (red + clásico + cada referencia de escala) en cámaras no vistas, **con una fila por referencia**: separación conocida y objeto a distancia conocida (corrigiendo la escala) | El resultado del TP; separado, para que no se diluya el aporte de la red |
| E4 | Cobertura de los intervalos de la incertidumbre total; cuántas veces dice «no segura» cuando corresponde | Que la confianza sea honesta |
| E5 | D435i real con cinta métrica | La prueba de verdad |

Métrica principal: **error de distancia por objeto entre 0,5 y 4 m** (percentil 20, como el
ubicador). Se separa **por cámara no vista**: el conjunto de cámaras de evaluación (incluidas las
6 geometrías de la pantalla) no se usa para entrenar.

**Criterio de éxito (E3):** con cámaras no vistas y la separación conocida, error mediano por
objeto ≤ 5 % y nunca «precisa» cuando el error real pasa el 10 %.

## 9. Pantalla de la expo

Seis recorridos simulados con la geometría publicada de cámaras conocidas. En pantalla dice
«Geometría de …»: se simula el ángulo, la separación y la resolución, no el sensor.

| Cámara | Campo horizontal | Separación | Resolución (un ojo) |
|---|---|---|---|
| ELP (la nuestra) | 116° | 5,98 cm | 1920×1080 |
| Intel RealSense D435(i) | 87° | 5 cm | 1280×720 |
| Intel RealSense D455 | 86° | 9,5 cm | 1280×720 |
| Stereolabs ZED 2 | 110° | 12 cm | 1920×1080 |
| Stereolabs ZED Mini | 102° | 6,3 cm | 1920×1080 |
| Luxonis OAK-D (Pro / S2) | 80° | 7,5 cm | 1280×800 |

Fuentes: hojas de datos y páginas de los fabricantes (Stereolabs, Intel, Luxonis). Para la
OAK-D las fuentes difieren entre 72° y 80°; se usa la hoja de la S2/Pro.

Cada recorrido muestra: la distancia real; «enchufada sin calibrar» (E0); «clásico»; y
«red de entrada», reservado hasta que exista. Perfiles en `render_tanda.py` (`zed-2`,
`zed-mini`, `realsense-d435`, `realsense-d455`, `oak-d`).

## 10. Pasos y responsables

| Paso | Quién | Cuándo |
|---|---|---|
| E0 y recorridos de las 6 cámaras | Tomás / Claude, PC de escritorio | 20–21/11 |
| Revisar este documento y cerrar el diseño de las dos redes | Bauty + Tomás | 21–22/11 |
| Tanda de recorridos con cámaras sorteadas (sección 7.2) | Tomás / Claude | después de la revisión |
| E1 en modo preciso | Bauty, en la Mac (GPU) | 21–23/11 |
| Prototipo de la red de entrada con `camara-sorteada` (E2) | Bauty | 23–27/11 |
| Pantalla «Cámara nueva» del dashboard y soporte RealSense | Tomás / Claude | 22–28/11 |
| Reentrenar la U-Net con cámaras variadas | Bauty | 27/11–1/12 |
| E3, E4 y E5 (D435i en la facultad) | todos | 1–5/12 |
| Entrega | — | 7/12 |
