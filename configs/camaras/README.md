# Perfiles de cámaras estéreo

Un archivo por cámara conocida (PLUG-AND-PLAY.md, sección 3). La app reconoce la cámara por el
nombre del dispositivo USB (`usb`: partes del nombre, sin importar mayúsculas) y carga su
perfil. Las cámaras que se calibran con la pantalla «Cámara nueva» se guardan en
`configs/camaras/locales/` (fuera de Git).

| Campo | Qué es |
|---|---|
| `id`, `nombre`, `fabricante` | Identidad; `nombre` es lo que dice la voz y el dashboard |
| `formato` | `sbs` (un USB con los dos ojos pegados, como la ELP), `realsense` (dos transmisiones infrarrojas por `librealsense`) o `dos-dispositivos` |
| `usb` | Partes del nombre del dispositivo para reconocerla |
| `resolucion` | Ancho y alto de **un ojo** |
| `campo_horizontal_deg`, `separacion_m` | Geometría nominal (hoja de datos del fabricante) |
| `calibracion` | Ruta a una calibración propia (ChArUco), si existe; si no, se usa la nominal |
| `nivel` | `precisa` (calibración propia o de fábrica) o `aproximada` (solo hoja de datos) |
| `fuente` | De dónde salen los números |
