# Incertidumbre hasta la voz y evaluación con ground truth

Definido por Tomás el 19/11 (con Bauty), a partir de la devolución de la cátedra: *«intentaría
hacer la conexión con la LLM, ya que no creo que sea trivial hacer que entienda el resultado de
la red»* y *«algún mini dataset de escenas cuyo input sea similar a lo que verían en la realidad
pero tengan ground truth de las distancias y labels»*. Complementa la decisión (a) de T12
(BITACORA 2026-11-19, «Decisión T12»).

**Para quien diseña la red (Bauty / su agente):** lo que la app necesita de la red está en la
[sección 1.2](#12-contrato-qué-tiene-que-entregar-la-profundidad); con qué se la va a evaluar, en
la [sección 2](#2-mini-dataset-con-ground-truth).

## 1. Conexión de la salida de la red con Sonnet

**Estado (19/11, noche): implementada del lado de la app y medida** (BITACORA, «T13: la
incertidumbre llega a la voz»). Falta que el ubicador llene `z_bajo_m` / `z_alto_m` con
`visionx.shared.uncertainty.laplace_interval` y el factor de b ajustado en val: hasta entonces
todo se dice como «medida precisa», igual que antes.

### 1.1 Principio

**El código decide qué tan segura es una distancia; Sonnet solo la dice con palabras.** Sonnet
nunca recibe b, píxeles ni confianzas numéricas, nunca hace cuentas y no puede volver más precisa
una medida. Lo que dice se verifica antes de reproducirlo.

```
b por píxel (U-Net) ──► intervalo del 90 % por objeto (metros) ──► nivel de certeza ──► frase / dato
confianza de la cámara (red de entrada) ─────────────────────────────┘                    │
                                                                                Sonnet ◄──┘ ──► verificación ──► voz
```

### 1.2 Contrato: qué tiene que entregar la profundidad

| Dato | Dónde | Quién | Estado |
|---|---|---|---|
| b de Laplace por píxel, en px de disparidad | `DepthResult.uncertainty`, `UncertaintyKind.LAPLACE_B` | Bauty | Existe (modo preciso); sin calibrar en la ELP |
| Factor de calibración de b (para que el intervalo del 90 % cubra ~90 %) | config del estimador | Bauty, con los datos de la sección 2 | Falta |
| Intervalo del 90 % por objeto: `z_bajo_m`, `z_alto_m` | `SpatialObject` (campos nuevos, opcionales) | Ubicador (`depth/spatial`, Bauty) | Falta |
| Confianza de la cámara / de la alineación | `DepthResult.alignment_quality` | Red de entrada (a), Bauty | Falta |
| Estado de la cámara: `alineando`, `camara-desconocida`, `no-medible` | `DepthResult.reason` (prefijo, ver `app/camera_status.py`) | Bauty | La app ya los dice; falta emitirlos |
| `calibrar_distancia(frame, mask, z_m)` | estimador | Bauty | La app ya lo llama si existe |

**Cálculo sugerido del intervalo** (Laplace: P(|error| > t) = e^(−t/b), así que el 90 % central
es ±b·ln 10 ≈ ±2,30·b):

```
b_obj  = mediana de b en la capa del objeto × factor_calibración
z_bajo = f·B / (d_frente + doffs + 2,30·b_obj)
z_alto = f·B / (d_frente + doffs − 2,30·b_obj)     # si el denominador ≤ 0: sin cota superior
```

La mediana de b por píxel es conservadora para un objeto (promedia muchos píxeles, pero sus
errores están correlacionados): por eso el factor se ajusta **por objeto** con la sección 2, de
modo que el intervalo contenga la distancia real ~90 % de las veces.

La U-Net está validada en el modo preciso (dos pasadas). Mientras no se valide en el continuo,
el intervalo se usa en las preguntas de distancia (que esperan al preciso, `espera_s` 3) y la
guía de buscar sigue diciendo metros sin nivel.

### 1.3 Niveles de certeza (app, Tomás)

`ancho = (z_alto − z_bajo) / (2·z)`; umbrales iniciales, a ajustar en la validación de la
sección 2 (nunca en test):

| Nivel | Condición | Cómo se dice |
|---|---|---|
| **Precisa** | ancho ≤ 10 %, política `SPEAK_METERS` y cámara sin aviso | «a 1,4 metros» (al decímetro, como hoy) |
| **Aproximada** | 10 % < ancho ≤ 25 % | «a unos 2 metros y medio» (al medio metro) |
| **No segura** | ancho > 25 %, sin cota superior, política distinta de `SPEAK_METERS` o cámara con aviso | sin número: dirección y «no puedo medir bien a qué distancia está» |

Sin intervalo (estimador sin b, p. ej. SGBM) se comporta como hoy (precisa si la política lo
permite). Con `camara-desconocida` la respuesta es el pedido de la decisión (a): «No reconozco
esta cámara. Apuntá a algo a una distancia conocida y decime "calibrá, está a 2 metros"».

### 1.4 Qué recibe Sonnet

Una línea por objeto, ya redactada y con el nivel explícito (el modo distancia, `pregunta`,
`describir` y `opciones` usan la misma función):

```
- taza: a 1,4 metros (medida precisa); 20 cm a tu izquierda; a la altura del pecho
- mochila: a unos 2 metros y medio (medida aproximada: decilo con «unos»)
- silla: a tu derecha; la distancia no se pudo medir bien (no digas ningún número)
Estado de la cámara: estoy ajustando la cámara (si hay aviso)
```

Reglas nuevas del prompt de sistema: usar los metros tal cual vienen; con «aproximada», decir
«unos»; con «no se pudo medir», no dar números ni «cerca» como si fuera medido; nunca inventar ni
afinar un número.

### 1.5 Verificación (como `verify_quotes` con el OCR)

Antes de hablar, se extraen de la respuesta los números seguidos de «metro(s)» o «centímetros»:
cada uno tiene que estar en la lista de lo permitido para ese pedido (con las formas escritas
«uno y medio», «dos metros y medio»…). Si aparece otro número, o un número para un objeto
«no seguro», la respuesta de Sonnet se descarta y se dice la frase fija del modo.

### 1.6 Cómo se mide (para el informe)

Banco fijo de ~60 casos armados (precisa, aproximada, no segura, cámara desconocida, varios
objetos, pregunta libre), corrido contra Sonnet con la key de VisionX (≈0,30 USD, dentro de
`assistant.budget_usd`):

- **Fidelidad numérica:** 100 % de las respuestas sin números fuera de lo permitido (antes de la
  verificación, para saber cuánto corrige; después, tiene que ser 100 %).
- **Respeto del nivel:** «unos» en aproximadas, sin números en no seguras.
- **Latencia** a la primera frase, igual que hoy.

## 2. Mini dataset con ground truth

**Idea clave:** para evaluar la distancia por objeto alcanza con **profundidad exacta por píxel
+ máscara del objeto**. Ningún dataset real trae estéreo, profundidad densa y labels de objetos a
la vez, pero las máscaras las podemos poner nosotros (detector + revisión a mano), o salen
exactas del render.

### 2.1 Los conjuntos (decidido: A + C)

**Decisión de Tomás (19/11):** se usan **A (GT-sintético-ELP), primero y completo**, y **C (banco
ELP)**. Booster queda descartado por ahora (solo como opción si sobra tiempo). El medidor láser
para C se consigue, si se puede, en la facultad.

| Conjunto | Qué es | Para qué | Quién |
|---|---|---|---|
| **A. GT-sintético-ELP** | Escenas de [Replica](https://github.com/facebookresearch/Replica-Dataset) (18 interiores escaneados, 88 clases, instancias) renderizadas con la **geometría de la ELP**: su calibración, 6 cm de separación, cámara a 1,2–1,4 m (pecho) | Cantidad (objetivo ≥ 1000 objetos a 0,5–4 m) con distancia y label exactos; no visto por CREStereo; **cámaras hermanas y distintas** para probar la decisión (a) | Tomás / Claude Code |
| **C. Banco ELP** | Sesiones `s1`–`s2` con stickers y cinta (T04, `bench marcar/evaluar`); con medidor láser si se consigue en la facultad | La prueba real en nuestra cámara | Bauty (cámara) |
| *Opcional: Booster* | [Booster](https://cvlab.unibo.it/booster-web/) (U. Bologna, CVPR 2022): 64 interiores reales con disparidad densa, sin labels de objeto | Solo si sobra tiempo | — |

### 2.2 GT-sintético-ELP: cómo se genera

- **Cámara:** intrínsecos y separación de la calibración vigente de la ELP (solo lectura de
  `calibration/`), imagen rectificada a 1280×720; altura 1,2–1,4 m, inclinación −15° a 0°,
  posiciones al azar sobre el piso navegable de cada escena.
- **Por vista:** `izq.png`, `der.png`, `z_izq.npy` (Z en metros, float32), `inst_izq.png`
  (id de instancia, uint16) y `vista.json` (pose, cámara, objetos visibles con clase de Replica
  y su equivalente del catálogo de la app).
- **Variantes de cámara** (mismas vistas): `elp` (exacta), `hermana-0.5` y `hermana-1` (rangos
  del generador de etapa 2, `sample_sibling`), y `distinta` (otra focal, otra distorsión y otra
  separación): esta última es la que debería terminar en «pedí un objeto a distancia conocida».
- **Partición por escena** (no por vista): val para ajustar umbrales y el factor de b; test
  congelado, que no se mira hasta la evaluación final.
- **Renderizador:** a definir en el primer paso de factibilidad (Habitat-Sim corre en
  Linux/macOS; en Windows, rasterizar las mallas de Replica directamente). Requisito: Z exacta e
  id de instancia del mismo render que la imagen izquierda.
- **Dónde vive:** generado en `data/gt-sintetico-elp/` (no se sube) y en Drive
  `VisionX-artefactos/gt-sintetico-elp/`; en el repo, el script, el manifiesto y las semillas.
  Licencia de Replica: solo investigación o educación, sin redistribuir fuera del equipo.

### 2.3 Qué se mide por objeto (A y C)

Distancia de referencia = la misma definición que usa el ubicador: percentil 20 de Z en la parte
visible del objeto (frente), más la mediana como dato extra.

- **Error:** MAE, sesgo y P95 en metros y en %, por rango (0,5–1, 1–2, 2–3, 3–4 m).
- **Intervalo:** cobertura del 90 % (tiene que dar ~0,90 después de calibrar b).
- **Niveles:** para cada nivel, % de objetos y error real (las «precisas» tienen que errar
  ≤ 10 %; las «no seguras» tienen que ser las que más erran).
- **Abstención y riesgo-cobertura:** cuánto se calla el sistema y cuánto mejora el error al
  callarse.
- **Cámaras:** lo mismo por variante (`elp`, `hermana-*`, `distinta`), con y sin la red de
  entrada y con y sin distancia conocida.

## 3. Orden de trabajo

1. Tomás / Claude Code: factibilidad del render y GT-sintético-ELP completo (A).
2. Tomás / Claude Code: campos opcionales del intervalo en `SpatialObject`, niveles, prompt,
   verificación y el banco de 60 casos (funciona sin intervalo; se activa cuando llegue).
3. Bauty: intervalo en el ubicador, factor de b ajustado en val, confianza y estados de la red
   de entrada, `calibrar_distancia`.
4. Bauty: banco ELP (C), con láser si se consigue.
5. Los dos: evaluación final por objeto en A y C (test congelado) para el informe.
