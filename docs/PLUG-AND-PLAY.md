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
- **Salida, por ojo:** campo de visión horizontal (o `log f` relativo al ancho), centro
  `(cx, cy)` relativo, distorsión radial `k1, k2`. Cada uno **con su incertidumbre** (media y
  `b` de una Laplace, igual que la U-Net).
- **Arquitectura sugerida:** un codificador por cuadro (CNN chica o un ViT preentrenado),
  promedio con atención sobre los N cuadros, cabeza que regresa los parámetros. Los dos ojos
  comparten pesos. Modelos publicados que estiman focal y distorsión desde una foto (por
  ejemplo GeoCalib, 2024) muestran que el problema se puede aprender; lo nuestro suma varios
  cuadros, los dos ojos y la confianza.
- **Pérdida:** NLL de Laplace sobre cada parámetro **más** una pérdida geométrica: el error de
  reproyección de una grilla de puntos con los parámetros estimados contra los reales (así, un
  error de focal que no cambia la imagen no se castiga de más).
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
| Objeto a distancia conocida | Lo de hoy (`calibrar_distancia`): fija el corrimiento de columnas | precisa |
| Dos objetos a distancias distintas | Fija la escala **y** el corrimiento a la vez | precisa |
| Objeto de tamaño conocido | Hoja A4, tarjeta (85,6 mm) | aproximada a precisa (a medir) |
| Piso a la altura del pecho | La altura se mide una vez; el plano del piso da la escala | a medir |
| Red monocular (solo para la escala) | Ajusta un número una vez, con muchos píxeles y cuadros | aproximada |

«Sin tocar nada» funciona con distancias **aproximadas**; cualquier paso extra las vuelve
**precisas**. Encaja con los tres niveles de la voz.

## 6. Red de incertidumbre: sumar la duda de la calibración

Hoy la U-Net estima `b` por píxel (el error de disparidad). Con una cámara estimada hay un
segundo error, el mismo para toda la imagen: la escala (focal × separación) y el corrimiento.
Para una distancia `Z`:

```
(σ_Z / Z)² ≈ (Z · σ_d / (f·B))²  +  σ_escala²
```

- `σ_d` sale de la U-Net (`b`), como ahora.
- `σ_escala` sale de la red de entrada y de la referencia usada (sección 5): casi 0 con la
  separación conocida; el que dé la red monocular si es «aproximada».
- Los niveles de la voz (precisa ≤ 10 %, aproximada ≤ 25 %, no segura) se calculan con el
  total. Una cámara mal estimada nunca dice «precisa».
- La U-Net hay que **reentrenarla con cámaras variadas** (hoy vio solo la geometría de la ELP)
  y recalibrar el factor de `b` (hoy 0,281).

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
| Distorsión radial `k1, k2` sorteada, aplicada sobre el lienzo con margen | Lentes reales sin renderizar cada una |
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
| **E0** ✔ | Cámaras distintas **arrancando desde la calibración de la ELP** (como quien enchufa sin calibrar), con y sin clásico, con distancia conocida | Hecho el 20/11 (BITACORA): calibrada 0,9–5,5 %, sin calibrar 25–115 %, sin calibrar + clásico 9–79 % |
| E1 | Clásico + distancia conocida contra el oráculo, **modo preciso**, muchas hermanas, val | Cuánto hueco queda dentro de una misma cámara (hoy: 5,6 % contra 0,75 %) |
| E2 | Red de entrada sola: error de campo de visión, centro y distorsión en cámaras **que no vio** | ¿Se puede estimar una cámara? Meta: campo ±3 % |
| E3 | Flujo completo (red + clásico + cada referencia de escala) en cámaras no vistas | El resultado del TP |
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
