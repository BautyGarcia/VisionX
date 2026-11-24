# Tareas de Bauty: plug and play (desde el 20/11)

> **Desde el 22/11 manda [PLAN-CIERRE](PLAN-CIERRE.md)**: evaluación del proyecto, qué hace
> cada uno y calendario hasta la entrega. Tus tareas están en su sección 4 («Bauty»):
> 1. **A:** plug and play real con la ELP (guía en la sección 6 de PLAN-CIERRE).
> 2. **D:** entrenar la v3 con las etiquetas de la geometría: **comandos exactos en
>    [RED-ENTRADA](RED-ENTRADA.md#entrenar-la-v3-auto-bauty-en-la-mac-comandos-exactos)** (tres
>    corridas de ~1 h; el código ya está en `main`). **Hecho el 22/11**, con `factor_b`
>    (BITACORA 22/11, noche Mac); falta la parte con video real (punto 3).
> 3. Video real girando la ELP para la v3.
> 4. El factor de b en el checkpoint de la red de entrada (snippet en RED-ENTRADA) y la U-Net
>    con cámaras variadas: hay un camino rápido sin reentrenar (b × 0,046·(f·B)^0,51).
> 5. Todas las pruebas con cámara (B, C y G de Tomás, D435i, prueba con personas).
> 6. Ensayo de la expo en la Mac.
>
> Lo de abajo queda como referencia (el Drive ya se bajó el 21/11; la red de entrada v1/v2 se
> integró en #90).

Para Bauty y su Claude. Se puede usar tal cual como pedido:

> «Leé `docs/PLUG-AND-PLAY.md` y `docs/TAREAS-BAUTY.md`, y hacé las tareas en orden. La Mac
> es para **entrenar** (MPS); evaluar y correr CREStereo lo hace la PC de Tomás. Cada resultado
> va a la BITACORA con su tabla.»

**Contexto en una línea:** el objetivo del TP pasó a ser **plug and play con cualquier cámara
estéreo de un solo cuerpo** más la red de incertidumbre. La red de entrada ya no compite con el
clásico: **estima la cámara desconocida** (focal, centro, lente, con confianza), que el clásico
no puede ver. Todo el porqué está en [PLUG-AND-PLAY](PLUG-AND-PLAY.md).

**Cambio del 20/11 a la noche: la PC de Tomás ya usa su GPU.** CREStereo exportado a ONNX
corre en la Radeon con DirectML: 0,12 s por cuadro en continuo y 0,51 s en preciso, igual o
mejor que la Mac, con el mismo resultado (0,002–0,004 px de diferencia). Habitat también dibuja
en la GPU. Por eso se reparte así:

| Máquina | Qué hace |
|---|---|
| **Mac de Bauty** | **Entrenar**: la red que estima la cámara y la U-Net de incertidumbre (PyTorch en Windows no entrena con AMD) |
| **PC de Tomás** | **Correr**: E1, E3, E4, videos, GT sintético, disparidades precalculadas para entrenar, evaluar los checkpoints que mande Bauty |

Entre las dos máquinas se intercambian archivos por el Drive: de la PC salen datos y
disparidades; de la Mac, checkpoints (`.pt`) para evaluar acá.

**Para no pisarse:**
- Tomás / Claude tocan `app/`, `camera/`, `gtsynth/`, `voice/`, `scripts/` y los docs de
  diseño.
- Bauty toca `depth/` y su módulo nuevo `depth/camara_net/`. La única excepción es el modo ONNX
  de CREStereo (`depth/crestereo_adapt/onnx_dml.py` + 13 líneas en `port.py`): lo hizo Claude
  en un PR aparte que **Bauty revisa**.
- BITACORA: cada uno agrega sus entradas arriba; no se editan las del otro.

## 0. Preparar

1. `git pull` de `main`.
2. Datos: los mismos de siempre, en el Drive `VisionX-Personal/gt-sintetico-elp/`. Para estas
   tareas hacen falta `camara-sorteada` (2400 train + 240 val, Replica) y
   `hssd-camara-sorteada` (2460 train + 600 val) para E2.

   Copiarlos a un disco local antes de entrenar (leer del Drive es muy lento).

   Datos nuevos (20/11):
   - `VisionX-Personal/gt-video/` (~15 GB): los recorridos de 90 cuadros de las **6 cámaras de
     la expo** (`video-elp-full`, `video-realsense-d435`, `video-realsense-d455`, `video-zed-2`,
     `video-zed-mini`, `video-oak-d`) y de 3 genéricas (`video-angosta-60`,
     `video-gran-angular-130`, `video-base-larga-20`), en `hotel_0` y `office_2` (val).
     **Son cámaras de evaluación: no entrenar con ellas.**
   - `VisionX-Personal/gt-sintetico-elp/hssd-recorridos-sorteada/` (15,3 GB, ya subido): un
     recorrido de 30 cuadros por casa de HSSD (153 casas: 123 train + 30 val; las 2 de siempre
     no se pueden dibujar), **cada recorrido con su propia cámara** sorteada (55–130°, 3–20 cm,
     16:9 o 4:3), a 960 px de ancho con lienzo de margen. Es para la versión de varios cuadros de la
     red de entrada (E2) y para reentrenar la U-Net con cámaras variadas.
3. `unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN` antes de correr nada (la key de VisionX es
   solo para el modo de habla).

### 0.1 Bajar el Drive a tu disco (importante)

El Drive de Tomás tiene un límite de 200 GB y ya está casi lleno. **Bajá a tu disco todo lo que
puedas** de `VisionX-Personal/`; cuánto, lo decidís vos según el espacio que tengas. Cuando
termines, **avisale a Tomás qué carpetas bajaste completas**: él las borra del Drive para seguir
subiendo datos nuevos. Las copias originales quedan en el disco de 2 TB de su PC, así que no se
pierde nada.

Orden sugerido (tamaños aproximados):

| Prioridad | Carpetas | GB | Para qué |
|---|---|---:|---|
| 1 | `gt-sintetico-elp/elp`, `camara-sorteada`, `hssd-camara-sorteada` | 39 | E1 y E2 |
| 2 | `gt-video` (6 cámaras de la expo + 3 genéricas) | 15 | Evaluar E2 en cámaras no vistas |
| 2 | `gt-sintetico-elp/hssd-recorridos-sorteada` (ya subido) | 16 | E2 con varios cuadros, reentrenar la U-Net |
| 3 | `camara-sorteada-ojo`, `elp-margen`, `elp-margen-ojo`, `distinta` | 39 | Ojo corrido, lentes con margen, escala |
| 4 | `secuencias`, perfiles fijos (`media-90`, `gran-angular-130`, `cuatro-tercios-90`, `base-larga-20`, `base-corta-3`, `angosta-60`), poses (`elp-arriba-*`, `elp-adelante-2`, `pose-*`), `ojo-de-pez` | 26 | Casos difíciles y otras evaluaciones |
| — | `pesos/`, `stand-camaras/`, `stand-video/` | < 1 | Pesos de YOLOE y demos del stand (`stand-video` se reemplaza el 21/11 por los videos con la ficha nueva: bajalo de nuevo después) |

**Lo que todavía no está en el Drive** (generado la noche del 20/11, en el disco de Tomás). Llega
por disco externo o por el Drive a medida que lo liberes; avisá qué preferís:

| Carpeta | Qué es | GB | Para qué |
|---|---|---:|---|
| `gt-sintetico-elp/hssd-recorridos-sorteada-b` | 10.980 vistas, 122 casas de HSSD (train), 3 recorridos por casa con una cámara sorteada cada uno (verificada) | 37 | Más cámaras distintas para E2 y la U-Net |
| `cache-crestereo/` | Disparidades de CREStereo precalculadas (ver «Disparidades precalculadas» más abajo); la parte de `-b` termina el 21/11 | ~55 | Entrenar la U-Net sin correr CREStereo en la Mac |
| `gt-video/` (escenas nuevas) | Las 9 cámaras de evaluación en 6 escenas más (`frl_apartment_3`, `room_1` y 4 casas HSSD de val), además de `hotel_0` y `office_2` | — | Evaluar E2, E3 y E4 en más casas |

## 1. Revisar el diseño (antes de entrenar)

Leer las secciones 4, 5 y 6 de [PLUG-AND-PLAY](PLUG-AND-PLAY.md) y anotar en el PR #63 si algo
no cierra. En particular:

- **Qué estima la red de entrada:** por ojo, campo horizontal, centro y distorsión `k1, k2`,
  cada uno con su `b` de Laplace.
- **Cómo se reparte el trabajo con el clásico:** la red da el punto de partida; el clásico
  ajusta los giros.
- **La fórmula para sumar la duda de la calibración** a la incertidumbre de la U-Net.

La tanda grande de datos (sección 7.2) **espera a esta revisión**.

## 2. E1: lo corre la PC de Tomás (no hace falta hacerlo)

E1 (clásico + distancia conocida contra el oráculo, en modo preciso, con hermanas ×0,25, ×0,5 y
×1, y `distinta`) **pasó a la PC de Tomás**: corre la noche del 20/11 en la Radeon y la tabla
va a la BITACORA. Bauty solo la lee para decidir cuánto puede aportar la red **dentro** de una
cámara conocida (referencia de la v2: clásico 5,6 % contra oráculo 0,75 % en ×1).

Lo que sí le toca: **revisar el PR del modo ONNX de CREStereo** (`onnx_dml.py`), porque es su
código. Con la Mac no cambia nada: ahí sigue usando MPS.

## 3. E2: prototipo de la red que estima la cámara

La pregunta: **¿una red estima el campo de visión de una cámara que no vio con ±3 % de error?**
Si da, el plug and play tiene su pieza central. Si no, lo sabemos antes de generar más datos.

**Datos.** `camara-sorteada` y `hssd-camara-sorteada`: cada vista es una cámara distinta (campo
55–130°, separación 3–20 cm, 16:9 o 4:3). Para leerlas, usar `visionx.gtsynth.dataset.vistas(raiz,
variante)`; cada `Vista` trae:
- `izq()` y `der()`, renderizadas con un lienzo 25 % más grande;
- `recortar(img)`, que devuelve la cámara nominal;
- `meta["nominal"]` (`hfov_deg`, `cx`, `cy`, `ancho`, `alto`);
- `meta["camara"]["base_m"]`.

La partición es por escena: `vistas(raiz, variante, particion="train")`, o
`dataset.escenas_de("val")` (Replica + HSSD). **No usar test.**

**Etiquetas (por ojo, iguales en los dos ojos en estos renders):**
- `hfov_deg` (o `log(fx / ancho)`);
- `cx/ancho − 0,5` y `cy/alto − 0,5`;
- `k1, k2`, que salen del aumento (ver abajo).

**Aumento al entrenar (clave para que sirva con cámaras reales):**
- **Distorsión radial sorteada** (`k1` en [−0,35, 0,1], `k2` en [−0,1, 0,1]), aplicada sobre el
  **lienzo con margen** antes de recortar, para que no aparezcan bordes negros. La etiqueta es la
  `k1, k2` sorteada.
- **Recorte descentrado** de la cámara nominal (±3 %), que mueve `cx, cy`. La etiqueta se
  ajusta.
- **Ruido, desenfoque, JPEG, viñeteo, exposición distinta entre ojos y escala de grises**
  (RealSense y OAK-D ven en gris).
- **Escalar siempre a 640 px de ancho sin recortar** (el campo de visión tiene que quedar
  intacto).

**Arquitectura sugerida.**
- Un codificador por imagen (ResNet-18 o un ViT chico preentrenado), con los dos ojos como dos
  imágenes que comparten pesos.
- Promedio de los rasgos.
- Una cabeza que regresa los parámetros con media y `b`.
- Más adelante, varios cuadros de la misma cámara (los recorridos que se renderizan después).

**Pérdida.** NLL de Laplace por parámetro, más el error de reproyección de una grilla de puntos
con los parámetros estimados contra los reales (PLUG-AND-PLAY, sección 4).

**Dónde.** Un módulo nuevo, por ejemplo `src/visionx/depth/camara_net/`
(`datos.py`, `modelo.py`, `entrenar.py`, `evaluar.py`), con tests chicos como el resto.

**Entregable:**
- error del campo de visión, del centro y de `k1` en val (mediana y p90);
- el mismo error en las 6 cámaras de la expo, sin entrenar con ellas: los recorridos
  `video-*` de `VisionX-Personal/gt-video` (las etiquetas están en `vista.json`, igual que en
  `camara-sorteada`);
- cobertura del intervalo del 90 % de cada `b`;
- una entrada en la BITACORA.

## 4. Después (no arrancar hasta cerrar 1–3)

- **Reentrenar la U-Net de incertidumbre con cámaras variadas** (`camara-sorteada`, recorridos)
  y recalibrar su factor de `b` en val. Hoy vio solo la geometría de la ELP. Si las entradas
  de CREStereo (disparidad, pasadas, vertical) le sirven **precalculadas**, decir en qué
  formato y la PC de Tomás las genera en la Radeon y las sube al Drive: la Mac queda solo para
  entrenar.
- **Disparidades precalculadas** (noche del 21/11, en la PC de Tomás): `gtsynth disparidades`
  corre CREStereo preciso una vez por vista de `camara-sorteada`, `hssd-camara-sorteada` y
  `hssd-recorridos-sorteada` y guarda `data/cache-crestereo/<variante>/<escena>/<indice>.npz`
  (float16: `disparidad`, `vertical`, `mitad`, y `meta` con fx, separación y tamaño). La
  disparidad real sale de la vista (`fx · separación / Z`). Son ~2,7 MB por vista (~27 GB):
  no entran en el Drive, Tomás los pasa en disco. Si el formato no te sirve para
  `inc-entrenar`, decilo y se adapta.
- **Subir al Drive el checkpoint actual de la U-Net de incertidumbre** (`outputs/crestereo_inc/v2-s0/mejor.pt` o el que esté en uso): en la PC de Tomás no está, y
  sin él no se puede medir cómo se porta la incertidumbre en las 9 cámaras de evaluación.
- **Mandar los checkpoints** (`.pt`) al Drive cuando haya uno para probar: la PC de Tomás los
  evalúa (E2 en las 6 cámaras de la expo, E3, E4) y devuelve la tabla.
- **E3 y E4** (PLUG-AND-PLAY, sección 8): el flujo completo en cámaras no vistas y la cobertura
  de la incertidumbre total.
- **E5:** la RealSense D435i de la facultad, con cinta métrica. Su calibración de fábrica (se
  lee con `librealsense`) es la respuesta correcta para la red.

## Lo que hace Tomás / Claude en paralelo (para no pisarse)

Hecho:
- recorridos de las 6 cámaras de la expo;
- E0;
- pantalla «Cámara nueva»;
- perfiles de cámaras;
- RealSense;
- `--camara`;
- mosaico de la expo;
- recorridos con cámara sorteada;
- Habitat y CREStereo en la GPU.

En curso (noche del 20/11, en la Radeon):
- videos de las 5 cámaras reales;
- mosaico;
- E1.

Después:
- segunda tanda sintética: las 9 cámaras de evaluación en casas val de HSSD, para que E3 y E4
  tengan más de dos escenas;
- disparidades precalculadas si la U-Net las necesita;
- evaluar los checkpoints de Bauty;
- escala por voz en la app;
- prueba de la D435i en la facultad.
