# VisionX

Proyecto de I308 Visión Artificial, UdeSA, primavera 2026. Buscamos que una cámara estéreo
ayude a una persona con baja visión a buscar objetos, consultar distancias, leer y describir
escenas mediante audio. Primero funcionará por USB en computadora; después incorporará
Raspberry, batería y las capacidades del roadmap completo.

## Por dónde empezar

| Archivo | Para qué leerlo |
|---|---|
| [ROADMAP](docs/ROADMAP.md) | Entender el proyecto, sus etapas, alternativas y las 20 tareas |
| [TECNICO](docs/TECNICO.md) | Entender cada componente, sus entradas/salidas, decisiones y pruebas previstas |
| [BITACORA](docs/BITACORA.md) | Saber qué se hizo realmente, qué falló y qué falta; base del informe final |
| [AGENTS](AGENTS.md) | Trabajar con personas y agentes sin pisarse ni adelantarse al pedido |

Son documentos internos de aprendizaje y trabajo, no un texto para presentar al jurado.
Las siglas se explican en TECNICO. La entrega es el **7 de diciembre de 2026** y la presentación final el **12 de diciembre**
(confirmado por Tomás el 18/11).
Completar la entrega académica no equivale a terminar todas las capacidades de VisionX.

## Estado real

Hay captura y calibración de la ELP, profundidad B0–B5 y CREStereo congelado (T12), detección,
voz y una aplicación con los cuatro modos (buscar/guiar, distancia, leer y describir) probada
sobre clips grabados. Desde el 19/11 (PR #27): leer, describir y preguntas libres con Claude
Sonnet 5.5 (extra `ia`, necesita red y la key en `~/.visionx/anthropic_api_key`), voz Mateo por red con Piper de
respaldo, guía que corta la frase vieja y alerta de cercanía (BITACORA 2026-11-19). Falta la
prueba en vivo en la Mac ([guía](docs/PRUEBA-EN-VIVO.md)), la red propia de T12, los
comparadores y las métricas M1/M7–M10. Los detalles y la evidencia están en ROADMAP y BITACORA.

## Preparar el entorno cuando se vaya a implementar

Python 3.11.18 y [uv](https://docs.astral.sh/uv/). Desde el repositorio:

```text
uv sync --locked
git config core.hooksPath .githooks
uv run visionx doctor
```

En Windows con Smart App Control activado, los `.exe` que uv genera para los comandos
(`visionx.exe`, `pytest.exe`) pueden quedar bloqueados («os error 4551»). Usar el intérprete:
`uv run python -m visionx doctor` y `uv run python -m pytest`. Si el bloqueado es
`.venv\Scripts\python.exe`, renombrar `.venv` y volver a correr `uv sync --locked`.

Las tareas neuronales usan `uv sync --locked --extra ml`; el detector (T06), `--extra deteccion`;
la voz (T08), `--extra voz`; el OCR (T10), `--extra ocr`; el asistente con Claude, `--extra ia`.
`uv sync` deja instalados solo los extras que se le nombran: para tener ambos, pasar los dos.
Los modelos/pesos se obtienen por separado y se verifican antes de usarlos. `configs/local.yaml` solo se carga si se elige
explícitamente con el mecanismo indicado por el comando; existir no lo activa.
Una instalación distribuida usa `visionx doctor --runtime-only`.
Consultar TECNICO para las verificaciones futuras; no ejecutarlas por leer este README.

## Qué carpetas quedan

`src/` contiene código; `tests/` sus pruebas; `configs/` parámetros; `scripts/` utilidades.
`docs/` contiene los tres documentos principales y referencias originales de la materia.
`entregas/` conserva la propuesta HTML/PDF anterior: sus cifras son propuestas, no mediciones,
y hay que sincronizarla antes de una nueva entrega. Datos, videos y pesos van al almacén compartido.

El plan distribuido anterior y los registros originales se pueden recuperar desde el historial
de Git. La correspondencia de sus IDs está al final de ROADMAP. No hay que navegar por
esas fichas para empezar una tarea nueva. `docs/CONTRATOS.md` queda solo como enlace de
compatibilidad, porque hay comentarios del código publicado que apuntan a esa ruta.

Para ponerse al día, Bauty puede actualizar su `main` limpio con `git pull --ff-only` y abrir
los tres documentos. Si hay cambios locales o una tarea en curso, conservarlos y actualizar
su rama según AGENTS; no descartar trabajo para forzar el pull.

`visionx log` y `visionx novedades` todavía usan el esquema documental anterior: no muestran
ni mantienen estos archivos nuevos. Hasta adaptar esas utilidades, escribir directamente en
BITACORA y consultar ROADMAP. No perder resultados por confiar en el comando heredado.
