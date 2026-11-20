# Insumos para el póster (A0) y el informe

Todo lo que el póster tiene que poder contar, con su número y de dónde sale. Cada dato cita la
entrada de [BITACORA](BITACORA.md) por fecha y tema; si un número no está acá, no se pone en el
póster sin buscarlo antes en BITACORA. **Pendiente** = todavía no medido.

**Para quien compila el póster (Bauty / su agente):** la historia es «una red que sabe cuándo no
sabe y se adapta a otras cámaras, metida en un asistente por voz que una persona ciega puede usar
hoy». Las secciones 1–3 son el relato; 4–7 los datos; 8 las figuras.

## 1. Mensaje central

**VisionX:** asistente por voz para personas ciegas o con baja visión, con una cámara estéreo en
el pecho. Cuatro modos: **buscar y guiar** hasta un objeto, **decir a qué distancia** está algo,
**leer** texto y **describir** la escena, más **preguntas libres**.

Tres afirmaciones que el póster tiene que sostener con números:

1. **Mide distancias reales con estéreo, y sabe cuándo no puede.** CREStereo congelado + una U-Net
   propia que estima la incertidumbre por píxel; la voz dice «a 1,4 metros», «a unos 2,5» o «no
   puedo medir bien», y Claude no puede inventar un número (60/60).
2. **Se adapta a otras cámaras** (decisión (a)): una red de entrada corrige la diferencia con la
   ELP calibrada; si no alcanza, pide un objeto a distancia conocida.
3. **Es interactivo de verdad:** la primera palabra sale en ~1–2 s; se le habla en castellano
   rioplatense y responde con una voz uruguaya natural.

## 2. Línea de tiempo (hitos)

| Fecha | Hito | Fuente BITACORA |
|---|---|---|
| 28–30/10 | Base, contratos, alcance y entrega USB | 2026-10-28, 10-29, 10-30 |
| 04–12/11 | Calibración ELP: s1–s6; el precinto rompe la del 04/11; s6 elegida; cinta 1–3 m | 11-04, 11-09, 11-11, 11-12 |
| 13/11 | Corrección de disparidad (δ = 0,5275 px): error a 1,5 m de +28,8 a −5,0 mm | 11-13 |
| 13–14/11 | Captura, replay, rectificación; prueba M2 de la ELP en la Mac; clips de desarrollo | 11-13, 11-14 (T02) |
| 14/11 | Profundidad clásica (B0–B2) y con monocular (B3–B5, Depth Anything V2) | 11-14 (T05) |
| 14–17/11 | Detector YOLO-seg, voz (reglas, Whisper, Piper), 4 modos, beta integrada | 11-14 a 11-17 |
| 16–17/11 | **Fusion** (red propia sobre B4/B5): mejora en val, **no generaliza** fuera de dominio | 11-16, 11-17 |
| 18/11 | **Pivote a CREStereo adaptable** (idea del profesor); compuerta: latencia, cinta, robustez | 11-18 |
| 18/11 | Voz de la app: **Mateo** (Edge, es-UY); Claude Sonnet 5.5 elegido midiendo 3 modelos | 11-18, 11-19 |
| 19/11 | Regla de decisión, cámaras hermanas, **U-Net de incertidumbre entrenada** | 11-19 (T12) |
| 19/11 | Primera prueba en vivo en la Mac; buscar rehecho (centrar → distancia) | 11-19 (noche) |
| 19/11 | Decisión (a) y devolución de la cátedra; incertidumbre → voz; GT sintético | 11-19 (noche) |
| 19/11 | Velocidad (C1: YOLO en GPU), Whisper en GPU (MLX), voz sin cortes | 11-19 (noche) |
| 4/12 · 7/12 · 12/12 | Congelar · entrega · presentación | ROADMAP |

## 3. Arquitectura e infraestructura

```
ELP estéreo USB (2×1920×1080, 60 fps)
  → rectificación con la calibración (s6 + corrección de disparidad)
  → [red de entrada (a): corrige otra cámara hacia la ELP + confianza]        ← Bauty, pendiente
  → CREStereo congelado (continuo 640×360 · preciso 1280×720 a pedido)
  → U-Net de incertidumbre: b de Laplace por píxel
  → ubicador por objeto (máscara YOLO-seg, capa más cercana, percentil 20)
  → nivel de certeza (precisa / aproximada / no segura)
  → modos (buscar, distancia, leer, describir, preguntas)
  → Claude Sonnet 5.5 (imagen + OCR + medidas con su nivel; verificación de citas y números)
  → voz Mateo (Edge, caché) ← Whisper small en la GPU de Apple (MLX) ← botón + micrófono
```

| Pieza | Elegida | Por qué (fuente) |
|---|---|---|
| Cámara | ELP estéreo USB, 5,98 cm, 116° | Disponible; calibrada (11-11) |
| Profundidad | CREStereo congelado (5,43 M parámetros) | Fusion no generalizaba; CREStereo mide ±2 mm a 1–2 m (11-18) |
| Detector | yolo26s-seg @960, en la GPU de la Mac | Objetos difíciles con n@640 (11-15); MPS 34 ms contra CPU 164 ms (11-19) |
| Voz a texto | Whisper small; MLX en Mac | 96,7 % contra 78,7 % de base (11-17); CPU cargada tardaba 4,2 s (11-19) |
| Texto a voz | Edge es-UY-MateoNeural, +15 % | Elegida por Tomás entre varias; caché en disco (11-18, 11-19) |
| Asistente | Claude Sonnet 5.5, effort low | Opus más lento y caro; Haiku con errores (11-19) |
| OCR | RapidOCR (PP-OCRv5 latino) | Local, ~300 ms (11-17) |
| Equipo en vivo | MacBook Pro M2 Max (MPS) | Todo en un equipo; GPU para YOLO, CREStereo y MLX |
| GT sintético | Replica + Habitat-Sim en WSL (PC con RX 5700 XT) | Estéreo + distancia + labels exactos (11-19 noche) |

## 4. Decisiones y su porqué (lo que el profesor va a preguntar)

| Decisión | Alternativa descartada | Por qué | Fuente |
|---|---|---|---|
| Recalibrar (s6) | Seguir con la del 04/11 | Un precinto giró un ojo ~0,24°: escala −4,8 % | 11-09, 11-11 |
| Corrección de disparidad δ | Solo la calibración | Error +140 mm a 3 m → +1 mm | 11-13 |
| CREStereo congelado + redes propias antes y después | Fusion (estéreo + monocular) | Fusion empata con la regla fija B5 fuera de dominio | 11-17, 11-18 |
| Red de salida = incertidumbre (no corregir la disparidad) | Corregir la disparidad | La corrección no cambia el error (0,625 → 0,624 px) | 11-19 (U-Net) |
| Red de entrada = campo de corrección completo (a) | Solo 4 giros clásicos | Hermanas: 4 números dejan 3–11 %; el campo exacto, 0,3–0,9 % | 11-19 (noche) |
| Niveles de certeza decididos por código | Que Claude interprete b | Claude no hace cuentas; se verifica lo que dice | 11-19 (noche, T13) |
| Claude por API (Sonnet) | Modelo local / Opus / Haiku | Latencia, costo y calidad medidos en 20 casos | 11-19 |
| Mateo (Edge) sin respaldo Piper | Dos voces | «No quiero dos voces»: el respaldo sonaba en frases largas | 11-19 (tarde) |
| Buscar: centrar y después distancia, sin bip | Sonar con bip | En vivo el bip no gustó y la guía se pisaba | 11-19 (noche) |
| La búsqueda la termina la persona | Terminar al llegar o a los 30 s | Pedido de Tomás tras el vivo | 11-19 (noche, T09) |
| YOLO en la GPU (C1) | CPU, 768 px, 3 iteraciones | Detector 3,9 → 9,3 fps sin perder detección | 11-19 (noche, T07) |
| Whisper en MLX + respuesta por tramos | faster-whisper en CPU | Voz = ~8 de los 11 s de una pregunta | 11-19 (noche, T08) |
| GT sintético de Replica + banco ELP | Medir solo con cinta; Booster | Pocos casos con cinta; Booster sin labels (devolución de la cátedra) | 11-19 (noche, T13/T15) |

## 5. Resultados (con número y fuente)

### Calibración y profundidad

| Resultado | Valor | Fuente |
|---|---|---|
| Calibración s6 | RMS 0,215 px; vertical mediana 0,122 px (P95 0,379); escala +0,29 %; fx 602 px; base 59,81 mm | 11-11 |
| Corrección de disparidad (control 1,5 m) | +28,8 mm → **−5,0 mm (0,3 %)** | 11-13 |
| CREStereo preciso contra cinta | +1,9 / +2,0 / −2,1 / +25,6 / +34,2 mm a 1 / 1,5 / 2 / 2,5 / 3 m (SGBM B0: hasta −87,7 mm) | 11-18 |
| Latencia CREStereo (Mac M2 Max) | continuo 640×360: P50 120 ms; preciso 1280×720: P50 617 ms | 11-18 |
| Robustez a desalineación (8 pares) | roll 0,25°: 5,3 % sin corregir → 2,0 % clásico → 0,25 % exacta; roll 1°: 20,9 / 2,9 / 0,25 % | 11-18 |
| Descalibración real (precinto) | escala −4,8 % → −2,3 % con la regla clásica | 11-19 |
| Cámaras hermanas + distancia conocida | sin corregir 14,5 / 33,3 / 28,1 %; regla + distancia 4,5 / 3,1 / 10,9 %; campo exacto 0,35 / 0,29 / 0,24 % | 11-19 |
| U-Net de incertidumbre: AUSE (InStereo2K val, limpio / desalineado) | U-Net 0,276–0,291 / 0,361–0,369 (2 semillas); izquierda-derecha 0,276 / 0,341; fotométrico 1,023 / 0,647 | 11-19 |
| Cobertura del intervalo del 90 % | 0,95–0,96 limpio, 0,82–0,85 desalineado | 11-19 |
| U-Net: costo | +25 ms sobre una pasada (izquierda-derecha cuesta el doble) | 11-19 |
| Fusion (resultado negativo) | val: EPE 1,735 contra 1,801 de B5 (−3,7 %); Middlebury: empata con B5 | 11-16, 11-17 |
| **GT sintético, ELP calibrada** (val, 3376 objetos, máscara exacta) | error medio **0,7 / 0,9 / 1,2 / 1,6 %** a 0,5–1 / 1–2 / 2–3 / 3–4 m | 11-19 (noche, robustez) |
| GT sintético, **otra unidad sin corregir** (hermana ×0,5 / ×1) | 24–50 % / 44–63 % de error medio | 11-19 (noche, robustez) |
| GT sintético, **cámara distinta** (90°, 12 cm) supuesta ELP | −68,8 % en todos los rangos (error de escala puro) | 11-19 (noche, robustez) |
| Cámara distinta + **un objeto a distancia conocida** | 68,8 % → **1,2–3,7 %** | 11-19 (noche, robustez) |
| Inclinación de la cámara estimada del piso (GT) | ~1° de error cuando se ve el piso | 11-19 (noche, altura) |
| Red de entrada (a) en hermanas / distinta | **Pendiente** (Bauty entrenando) | — |
| Banco ELP s1 (cinta, por objeto) | **Pendiente** (Bauty) | — |

### Interacción

| Resultado | Valor | Fuente |
|---|---|---|
| Whisper (61 órdenes, Tomás) | small 96,7 % intención / 95,8 % objeto; base 78,7 / 70,8 % | 11-17 |
| Comparación de modelos de Claude (20 casos) | primera palabra: Opus 1,79 s · Sonnet 0,91 · Haiku 0,55; USD/consulta 0,011 · 0,0047 · 0,00025 | 11-19 |
| Banco incertidumbre → voz (60 casos) | **60/60 sin números inventados, 60/60 nivel respetado**, cámara 10/10; primer texto 0,64 s | 11-19 (noche, T13) |
| Velocidad en vivo | detector 3,9 → **9,3 fps** (C1); profundidad continua ~4,4 fps | 11-19 (noche, T07) |
| Latencia de una pregunta de distancia, antes | Whisper 4,2 s + preciso 1,9 + Claude 1,3 → primera palabra a 11,4 s | 11-19 (noche, T07) |
| Latencia después de MLX + tramos | «Responde mucho más rápido» (Bauty); **pendiente el número** (`latencias`) | — |
| Leer (T10) | precisión de lo hablado 95,8–96,0 %; OCR ~300 ms | 11-17 |
| Describir local (T11) | 0 afirmaciones sin respaldo en R2/R3 | 11-17 |
| M1 (contrato de la interacción) | 8 casos × 3 PASAN; cancelación P95 ≤ 200 ms | 11-17 (beta) |
| M8 buscar / M10 voz-leer-describir | **Pendiente** (dos sesiones, [MEDICION-MODOS](MEDICION-MODOS.md)) | — |

## 6. Datasets

| Dataset | Para qué | Tamaño usado | Fuente |
|---|---|---|---|
| InStereo2K | Ajuste de B0–B5, entrenar Fusion y la U-Net (con desalineación sintética) | 2238 train / 408 val (U-Net) | 11-13, 11-19 |
| Middlebury 2014 extra | Validación fuera de dominio | 13 pares | 11-17 |
| MiddEval3 / ETH3D | Test público (guardado para el final) | — | 11-17 (decisión) |
| Clips ELP de desarrollo | Modos, demos, latencia (`dev-1114-*`) | 12 clips | 11-14 |
| `cinta-s7` | Corrección de disparidad y cinta | 5 distancias × 3 pares | 11-13 |
| **GT sintético (Replica)** | Distancia por objeto con verdad exacta; cámaras hermanas y distinta | 1440 vistas, 15 088 objetos | 11-19 (noche) |
| Banco ELP (s1–s2) | La prueba real por objeto | **Pendiente** | — |

## 7. Lo que no funcionó (y por qué cuenta)

- **Fusion** (estéreo + Depth Anything): mejoraba en val y empató fuera de dominio → pivote.
- **Corregir la disparidad con la U-Net:** no cambió el error; la red aporta saber cuánto creerle.
- **Cuatro números para otra cámara:** no alcanzan; faltan centro y distorsión → red de entrada (a).
- **Whisper base:** 78,7 %, no llega al 90 %. **Haiku:** errores de contenido.
- **10 fps de todo el sistema:** no se alcanzó; el detector sí (9,3), pero lo que se sentía lento
  era la voz.
- **Bip del sonar** y **dos voces**: no gustaron en vivo → guía hablada y una sola voz.
- **Vencer la orden cortaba la respuesta** antes de los metros → plazo para empezar, no para cortar.

### Probar → corregir el 19/11 (ejemplo del método de trabajo)

| Prueba | Qué pasó | Corrección | Fuente |
|---|---|---|---|
| Demos con Claude | «alfajores Tita», «alguien tiene la caja» | Reglas del prompt | 11-19 |
| Demo de buscar | Dos voces; la guía se pisaba; flecha no siempre | Una voz; corte; flecha en vivo | 11-19 (tarde) |
| Vivo 1 (Mac) | ~5 s por respuesta; bip; sin flecha | Centrar → distancia; sin bip; voz +15 % | 11-19 (noche) |
| Vivo, banco s1 | Voz entrecortada al medir | Búfer de 0,2 s y cortes contados | 11-19 (noche, T08) |
| Vivo, velocidad | Whisper 4,2 s | MLX + respuesta por tramos | 11-19 (noche, T08) |
| Vivo, distancia | Frase cortada antes de los metros; «maus» | Plazo para empezar; metros primero; sinónimo | 11-19 (noche, T08/T09) |

## 8. Figuras sugeridas

| Figura | De dónde sale |
|---|---|
| Diagrama del sistema (sección 3) | Este documento |
| Persona con el arnés + qué escucha (globos de diálogo reales) | Demo en vivo; frases de las trazas |
| Par ELP → profundidad CREStereo → mapa de incertidumbre | `outputs/profundidad/` (Bauty) |
| **Curva de robustez**: ELP 1 % / hermanas 25–65 % / distinta 69 % → con distancia conocida 1–4 % (falta la fila «con la red de entrada») | 11-19 (noche, robustez); `datasets/gtsynth/val-*.json` |
| Captura de la pantalla del stand (cámara rotulada, radar, subtítulos, tiempos) | `docs/stand-ejemplo.jpg`; nueva con `python -m visionx.app ejecutar … --captura` |
| AUSE / riesgo-cobertura: U-Net vs izquierda-derecha | 11-19 (U-Net) |
| Barras de latencia: antes / después (Whisper, Claude, voz) | 11-19 (T07) + traza nueva |
| GT sintético: render, profundidad exacta, objetos | `python -m visionx.gtsynth visor … --exportar` (láminas listas) |
| Tabla «sabe cuándo no sabe»: 3 niveles con ejemplos y el 60/60 | 11-19 (noche, T13) |
| Error contra cinta: CREStereo vs SGBM | 11-18 |
| Línea de tiempo (sección 2) | Este documento |
