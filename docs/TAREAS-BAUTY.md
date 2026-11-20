# Tareas de Bauty: plug and play (desde el 20/11)

Para Bauty y su Claude. Se puede usar tal cual como pedido:

> «Leé `docs/PLUG-AND-PLAY.md` y `docs/TAREAS-BAUTY.md`, y hacé las tareas en orden. Corré lo
> que se pueda en la GPU de la Mac (MPS). Cada resultado va a la BITACORA con su tabla.»

**Contexto en una línea:** el objetivo del TP pasó a ser **plug and play con cualquier cámara
estéreo de un solo cuerpo** más la red de incertidumbre. La red de entrada ya no compite con el
clásico: **estima la cámara desconocida** (focal, centro, lente, con confianza), que el clásico
no puede ver. Todo el porqué está en [PLUG-AND-PLAY](PLUG-AND-PLAY.md).

**Por qué en la Mac:** es la única GPU que le sirve a CREStereo (continuo ~0,12 s por cuadro,
preciso ~0,6 s). En la PC de Tomás (sin CUDA) corre en CPU a ~1,5 s por cuadro. Los renders
(Habitat) siguen en la PC de Tomás, en Linux dentro de Windows.

## 0. Preparar

1. `git pull` de `main` (o `git checkout b/video-angosta` si el PR #63 todavía no se mergeó).
2. Datos: los mismos de siempre, en el Drive `VisionX-Personal/gt-sintetico-elp/`. Para estas
   tareas hacen falta:
   - `elp` (val y train) para E1;
   - `camara-sorteada` (2400 train + 240 val, Replica) y `hssd-camara-sorteada` (2460 train +
     600 val) para E2.

   Copiarlos a un disco local antes de entrenar (leer del Drive es muy lento).
3. `unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN` antes de correr nada (la key de VisionX es
   solo para el modo de habla).

## 1. Revisar el diseño (antes de entrenar)

Leer las secciones 4, 5 y 6 de [PLUG-AND-PLAY](PLUG-AND-PLAY.md) y anotar en el PR #63 si algo
no cierra. En particular:

- **Qué estima la red de entrada:** por ojo, campo horizontal, centro y distorsión `k1, k2`,
  cada uno con su `b` de Laplace.
- **Cómo se reparte el trabajo con el clásico:** la red da el punto de partida; el clásico
  ajusta los giros.
- **La fórmula para sumar la duda de la calibración** a la incertidumbre de la U-Net.

La tanda grande de datos (sección 7.2) **espera a esta revisión**.

## 2. E1: cuánto hueco deja el clásico dentro de una misma cámara

Es la tanda de hermanas que quedó pausada el 19/11 (BITACORA), en **modo preciso**. Cada
corrida, con 160 vistas de val, son unos minutos en la Mac.

```bash
R=<carpeta local>/gt-sintetico-elp
O=outputs/gtsynth/e1
mkdir -p $O
# Oráculo: el par perfecto
python -m visionx.gtsynth evaluar $R --variante elp --particion val \
  --profundidad crestereo-preciso --csv $O/elp.csv
for h in hermana-0.25 hermana-0.5 hermana-1; do
  # Sin corregir
  python -m visionx.gtsynth evaluar $R --variante $h --particion val \
    --profundidad crestereo-preciso --vistas-por-camara 8 --csv $O/$h-sin.csv
  # Clásico + distancia conocida (lo que usa hoy VisionX)
  python -m visionx.gtsynth evaluar $R --variante $h --particion val \
    --profundidad crestereo-preciso --alinear clasico --vistas-por-camara 8 \
    --distancia-conocida --csv $O/$h-clasico-dc.csv
done
# Otra cámara supuesta ELP (escala pura), con distancia conocida
python -m visionx.gtsynth evaluar $R --variante distinta --particion val \
  --profundidad crestereo-preciso --vistas-por-camara 8 --distancia-conocida \
  --csv $O/distinta-dc.csv
```

**Entregable:** una tabla en la BITACORA con el error mediano por objeto, de 0,5 a 4 m, por
rango, en cada fila: oráculo, sin corregir y clásico + distancia conocida, para ×0,25, ×0,5 y
×1. Decide cuánto puede aportar la red **dentro** de una cámara conocida. Referencia de la v2
(6 pares de la ELP): clásico 5,6 % contra oráculo 0,75 % en ×1.

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
  `video-*` de `data/gt-video`, que Tomás pasa por disco;
- cobertura del intervalo del 90 % de cada `b`;
- una entrada en la BITACORA.

## 4. Después (no arrancar hasta cerrar 1–3)

- **Reentrenar la U-Net de incertidumbre con cámaras variadas** (`camara-sorteada`, recorridos)
  y recalibrar su factor de `b` en val. Hoy vio solo la geometría de la ELP.
- **E3 y E4** (PLUG-AND-PLAY, sección 8): el flujo completo en cámaras no vistas y la cobertura
  de la incertidumbre total.
- **E5:** la RealSense D435i de la facultad, con cinta métrica. Su calibración de fábrica (se
  lee con `librealsense`) es la respuesta correcta para la red.

## Lo que hace Tomás / Claude en paralelo (para no pisarse)

- Recorridos de las 6 cámaras de la expo, E0 y los videos del stand, en la PC de escritorio.
- La pantalla «Cámara nueva» del dashboard y el soporte de RealSense (dos transmisiones
  infrarrojas).
- La tanda grande de recorridos con cámaras sorteadas, cuando Bauty cierre la revisión (paso 1).
