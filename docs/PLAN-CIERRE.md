# Plan de cierre: evaluación del proyecto y quién hace qué (22/11)

Evaluación del proyecto completo (Claude, a pedido de Tomás), críticas, mejoras que suman valor
y el reparto entre Tomás y Bauty hasta la entrega. Reemplaza el «Camino a la entrega» de
[ANALISIS-ESTADO](ANALISIS-ESTADO.md). Los números salen de la [BITACORA](BITACORA.md); la
historia de la red de entrada, de [RED-ENTRADA](RED-ENTRADA.md).

**Fechas:** congelar el 4/12, entrega el 7/12 (informe IEEE ≤ 8 páginas + póster), expo el
12/12.

## 1. Veredicto

El discurso es bueno y alcanza para la materia con margen. Hay tres cosas que pocos trabajos
tienen juntas:

1. **Un sistema que funciona de punta a punta.** Cuatro modos por voz, CREStereo, detección y
   Claude solo para lo abierto.
2. **Una pregunta de investigación con números:** ¿se puede usar cualquier cámara estéreo sin
   calibrarla? E0 mide el problema. La red de entrada más el movimiento lo atacan.
3. **Resultados negativos bien documentados:** Fusion, y la red de alineación contra el clásico.

**Lo que hay hoy, en números:**

| Qué | Resultado |
|---|---|
| Enchufar sin calibrar (E0) | 33–78 % de error en distancia, contra 3–8 % calibrada |
| Red de entrada v2, sola | 12,7 % de error en la focal por cuadro (8 escenas × 9 cámaras); bien en Replica y mal en las casas de HSSD |
| Red + 15 s girando la cámara (autocalibración) | **0,9 % de error en la focal** (P90 3,6 %), con la duda bien puesta (cubre el 89 %) |
| Sin la red como punto de partida | Falla en todo el gran angular de 130 °: la red es el seguro contra cámaras raras |
| Sistema completo con la focal de la red (E3) | 8,7 % de error medio en distancia, contra 9,8 % suponiendo una ELP y 5,2 % calibrada |
| **Sistema completo: red + 15 s girando + separación + objeto (E3)** | **5,4 % (mediana 3,4 %): +0,2 puntos sobre la calibrada.** Con la focal de la red sola, 13,5 % |
| Pantalla «Cámara nueva» con todo integrado | ZED 2 simulada: midió 109 ° (real 110 °), alineó los ojos de 6,8 a 0,2 px y quedó «precisa» |

### Qué se puede afirmar hoy (y qué no)

- **Sí, en simulación:** con 15 s girando la cámara, la separación de la hoja de datos y un
  objeto a distancia conocida, una cámara estéreo desconocida mide con el mismo error que una
  calibrada con tablero (+0,2 puntos; 9 cámaras × 8 escenas).
- **Sí:** en modo continuo y por píxel, el error que queda es casi todo de CREStereo.
- **Sí, con matiz, en modo preciso y por objeto** (lo que usa la voz):
  - la cámara estimada da **0,83 % de mediana** contra 0,34 % de la calibrada (9 cámaras);
  - los objetos con más de 10 % de error pasan de 2 a 4 %;
  - la diferencia es el resto de la focal (~1 %): a 2 m son ±2 cm.
  - BITACORA 22/11 (noche).
- **Todavía no:** «funciona con cámaras reales». Falta la prueba A con la ELP. La autocalibración
  aún no busca la distorsión (G), y las cámaras baratas la tienen.
- **No:** «sin hacer nada». Hacen falta los mini-setups: girar, la separación o un objeto.

## 2. Críticas, de la más grave a la menos grave

1. **Casi todo está medido en simulación.** La prueba real (E5) está en 0. Plug and play, red,
   movimiento e incertidumbre se midieron sobre renders. Es lo primero que van a preguntar.
2. **La historia está dispersa.** En 8 páginas no entra todo: Fusion, alineación, YOLOE, VLM,
   sonar, U-Net, red de entrada, movimiento. Hace falta una columna vertebral (sección 5).
3. **El pedido original del profesor** era adaptar CREStereo a pares sin rectificar con una red
   de alineación. Terminó en «el clásico le gana a la red» (19/11). Hay que contarlo como
   hallazgo que lleva al giro actual: el problema real no era alinear, era no conocer la cámara.
4. **La incertidumbre total no llega a la voz.** Hoy la voz usa solo la U-Net, entrenada con la
   ELP. No suma la duda de la cámara ni la de la escala (PLUG-AND-PLAY, sección 6). Con una
   cámara desconocida, la simulación mostró que la cobertura cae de 96 % a 47–84 %: la voz diría
   «precisa» sin serlo.
5. **La autocalibración no maneja distorsión.** Los renders son pinhole; las cámaras baratas
   tienen lente.
6. **Documentos viejos.** El ROADMAP es del 18/11, dice que no hay código y que la entrega es el
   10/12.
7. **El informe y el póster no empezaron.**

## 3. Mejoras que suman valor, por valor sobre costo

| # | Mejora | Por qué suma | Quién |
|---|---|---|---|
| **A** | **Plug and play real con la ELP:** tratarla como desconocida (`camara-nueva --fuente usb --dispositivo "Otra"`), girarla 15 s y comparar con su calibración de tablero; después, cinta con 10–15 objetos | El primer número real del aporte central | Bauty |
| **B** | **Duda total en la voz:** la duda de la cámara (focal medida, escala) se suma a la de la U-Net antes de elegir «a 2 metros», «a unos 2 metros» o «no puedo medir bien» | Cumple la promesa de honestidad con cualquier cámara | Tomás: **hecho el 22/11**, falta probarlo en la ELP |
| **C** | **Se recalibra sola:** alinea los ojos al arrancar y, si el desfase sube (golpe), vuelve a alinear sola | Útil (una cámara en el pecho se golpea) y fácil de mostrar. Además tapaba un hueco: el perfil de «Cámara nueva» no guardaba la alineación | Tomás: **hecho el 22/11**, falta probarlo en la ELP |
| **D** | **La geometría entrena a la red (v3):** etiquetas del movimiento, primero en simulación y después con video real | El costado de investigación más fuerte: auto-supervisión sin tableros | Tomás etiqueta y evalúa; Bauty entrena |
| **E** | **Prueba con personas:** 5–6 compañeros con los ojos vendados, 3 tareas (encontrar, distancia, leer), éxito y tiempo, con consentimiento | Evaluación con usuarios: muy valiosa en un trabajo de asistencia | Bauty con la cámara; Tomás organiza |
| **F** | **Demo «traé tu cámara» en la expo:** cualquier cámara estéreo se calibra en vivo | La frase de la expo; sale de A + C | Los dos |
| G | **Distorsión en la autocalibración:** buscar k1 junto con la focal | Necesario para cámaras baratas reales | Tomás (Codex) |

**Lo que no se hace:** Raspberry/portátil, Lite, Fusion-FFS, detectores nuevos, más renders. Hay
datos de sobra; el cuello de botella es validar en real y escribir.

## 4. Quién hace qué

| | **Bauty** | **Tomás** |
|---|---|---|
| Equipos | Mac M2 Max (entrena con MPS; corre todo muy bien), **la ELP** | PC con Radeon (CREStereo ONNX, Habitat), **todos los datos en `D:`** |
| Planes | Claude 100 USD | 2 × Claude 20 USD, 1 × Codex 20 USD |

### Bauty (todo lo que necesita la cámara o entrenar)

1. **A: plug and play real con la ELP.** Guía en la sección 6. Incluye la medición con cinta
   grabada, que reemplaza al banco ELP (paso 4).
2. **D: entrenar la v3** con las etiquetas de la geometría (`outputs/autoetiquetas/`, en el
   Drive), más el control con la cámara verdadera. Mezclar con los datos de siempre, partir de
   la v2. Detalle en [RED-ENTRADA](RED-ENTRADA.md).
3. **Video real para la v3:** grabar recorridos girando la ELP en 5–10 lugares distintos.
   Después, la D435i cuando se consiga. Tomás los etiqueta en la PC.
4. **U-Net con cámaras variadas** (`cache-crestereo`, en el Drive), y guardar el factor de b en
   el checkpoint de la red de entrada (×1,79 en val; fuera de val hace falta ×3,6).
5. **Todas las pruebas con cámara:** probar en la ELP real lo que programe Tomás (B, C, G); la
   D435i (calibración de fábrica, «Cámara nueva», cinta); la prueba con personas (E).
6. **Ensayar la expo en la Mac**, que es la máquina de la demo.

### Tomás (código sin cámara, evaluación, coordinación)

1. **B y C**: hechas y probadas en simulación el 22/11 (guía para la ELP en la sección 6).
2. **D:** etiquetas de la geometría (`gtsynth autoetiquetas`) y **evaluar cada checkpoint** que
   entrene Bauty con el mismo protocolo (8 escenas × 9 cámaras + E3 en la Radeon), el mismo día.
3. **G: distorsión** en la autocalibración (con Codex y una especificación cerrada).
4. **Martes 25/11 en la facultad:** pedir la otra ELP y ver si hay RealSense para la expo.
5. **Organizar E:** tareas, planilla y consentimiento.
6. **Poner al día ROADMAP y TECNICO.**

### Sin dueño todavía

- **Informe y póster.** Se decide después, pero conviene arrancarlos antes del 26/11.

## 5. El discurso (columna vertebral del informe y del póster)

> Enchufá cualquier cámara estéreo, girala 15 segundos y VisionX la calibra sola: mide la lente
> con menos de 1 % de error, alinea los dos ojos y te dice a qué distancia están las cosas, con
> un margen honesto. Y cada cámara que se calibra así le enseña a la red a reconocer la
> siguiente.

El número que lo sostiene (simulación, 72 recorridos, 9 cámaras): con 15 s girando, la
separación de la hoja de datos y un objeto a distancia conocida, una cámara desconocida mide con
**+0,2 puntos** de error sobre una calibrada con tablero (5,4 % contra 5,2 % por píxel).

Para que sea verdad: **A** (real) y **B** (margen honesto en la voz). «Le enseña a la red» es
**D**. Lo demás ya está.

Orden del informe propuesto: problema (E0) → por qué la red de alineación no alcanzaba (19/11) →
red de entrada + movimiento (E2, autocalibración) → sistema completo (E3) → incertidumbre →
prueba real (A, E5) → prueba con personas (E) → la geometría entrena a la red (D).

## 6. Guía para A (Bauty, con la ELP)

1. `git pull` de `main`. Tener `outputs/camara_net/e2-v2/mejor.pt` (Drive
   `pesos/camara-net/e2-v2/`).
2. Tratar la ELP como desconocida, girarla despacio 15 s hacia los costados y decir la
   separación (6 cm):

       python -m visionx.app camara-nueva --fuente usb --dispositivo "ELP de prueba" --escala 6 --captura elp-alta.png --guardar

   (Sin `--dispositivo` la reconoce y no hace el alta.)
3. Anotar el campo que midió (pantalla y `configs/camaras/locales/elp-de-prueba.yaml`, campo
   `campo_horizontal_deg` y `extra.sigma_focal`) contra la calibración de tablero de la ELP.
   Repetir 3 veces en lugares distintos.
4. Con ese perfil, correr la app. La **alineación en vivo (C)** se activa sola con perfiles de
   «Cámara nueva»: alinea los ojos al arrancar y vigila el desfase.

       python -m visionx.app ejecutar --fuente usb --camara elp-de-prueba --voz

   Cinta con 10–15 objetos entre 0,5 y 4 m (modo preciso), igual que M9. Anotar también qué
   nivel dice la voz: con la **duda total (B)**, una cámara dada de alta no debería decir
   «precisa» si no lo es.

   **Esta medición reemplaza al banco ELP** (decisión del 22/11). En vez de 40 escenas en 5
   sesiones con stickers, es una sola sesión que sirve para tres cosas:
   - el número real del plug and play (A);
   - la duda de la U-Net en la ELP;
   - la prueba de B.

   Para eso, además de anotar, se **graban los pares**. Se reutiliza del protocolo de TECNICO
   («Protocolo de grabación del banco ELP») la planilla, la cinta y el comando de grabación.

   - **Antes:**
     - ELP transmitiendo 40 min (`camera ver`): la disparidad deriva con la temperatura.
     - Cámara nivelada.
     - Una línea de cinta de papel en el piso, bajo el vidrio de las lentes.
   - **Dónde:** dos lugares distintos, con 1–2 escenas cada uno.
   - **Qué objetos:** 10–15 en total, entre 0,5 y 4 m, de clases de la app (botella, taza, silla,
     mochila…). Al menos 2 sobre superficies lisas o con reflejo.
   - **Stickers:** uno chico (~1 cm) y numerado en la cara de cada objeto que mira a la cámara.
     Ahí se mide.
   - **Medir** cada sticker con cinta:
     - la distancia **perpendicular** desde la línea del piso (profundidad axial, no en diagonal);
     - en metros con milímetros, sin sumar los 13 mm;
     - un sticker por lugar se mide tres veces.
   - **Grabar** 10 s con la cámara quieta, por escena:

         uv run python -m visionx.camera grabar recordings/banco-s1-eMM --segundos 10 --cada 4 --config configs/local.yaml --nota "s1 eMM: lugar, luz, qué hay"

     Completar `datasets/banco-elp-plantilla.yaml` y guardarla como
     `recordings/banco-s1-eMM/escena.yaml`, con `sesion: s1`. Sacar una foto del armado con el
     teléfono.
   - **Voz:**
     - Con la app andando con el perfil `elp-de-prueba`, apuntar a cada sticker en modo preciso.
       Girar la cámara en el lugar, sin sacarla de la línea.
     - Anotar en la planilla la distancia y el nivel que dijo (`voz_m`, `voz_nivel`).
   - **Después, sin cámara (Bauty / Claude):**
     - `bench marcar` y `bench evaluar` con la calibración de tablero (la referencia) y con el
       perfil de «Cámara nueva». Falta que `bench evaluar` acepte un perfil de cámara y el rango
       de 3,5 a 4 m.
     - El factor de b de la U-Net se ajusta en un lugar y se comprueba en el otro.
   - **Sin test real reservado:** las sesiones `s3`–`s5` ya no se graban. Sobre estas escenas no
     se ajusta nada, salvo ese factor de b.
5. **Distancia conocida en la app:** con la app andando, apuntar a algo y decir «calibrá, está a
   N metros»; después alejarse o acercarse un paso y repetirlo con la nueva distancia. La
   primera corrige el corrimiento; la segunda, escala y corrimiento. Se guarda en el perfil.
6. **Probar C:** con la app andando, golpear o torcer un poco la cámara y ver que vuelva a
   alinearse sola (mensaje en la terminal; `historial` de `FuenteAlineada`).
7. Todo a la BITACORA con tabla. Si la focal sale muy distinta de la del tablero, sospechar de
   la distorsión (G).

## 7. Calendario

| Fecha | Bauty | Tomás |
|---|---|---|
| 22–24/11 | A con la ELP; entrenar la v3 con las etiquetas | B y C; evaluar la v3; ROADMAP al día |
| 25/11 | Video real girando la ELP | Facultad: ELP y RealSense para la expo |
| 26–29/11 | U-Net con cámaras variadas; probar B y C en la ELP; E (con personas) | G; etiquetar el video real; evaluar la v3 real |
| 30/11–4/12 | Ensayo de la expo en la Mac | Figuras; cerrar documentos |
| 4/12 | Congelar | Congelar |
| hasta el 7/12 | Informe y póster (dueño a definir) | |
