"""Planillas: nunca completar observaciones ni referencias físicas por inferencia."""

from pathlib import Path

from .common import write_csv

GUIDES = {
    "m7": "Diez objetos, 1/2/3 m y 0,5 m, tres repeticiones independientes (120 filas). "
    "Centrar un único objeto de la clase COCO; no elegir luego el frame de menor error. "
    "Si hay varios iguales, anotar instancia_x/instancia_y en [0,1] sobre la imagen "
    "RECTIFICADA (punto interior de su caja); una coincidencia ambigua se abstiene. "
    "Cinta desde el vidrio de la lente izquierda al frente del objeto; guardar origen_cinta. "
    "Hipótesis pendiente: sesgo +4–8 cm; NO corregirlo automáticamente. "
    "z_offset_m es la corrección firmada de distancia_cinta_m a Z desde el origen óptico; "
    "0 solo si se adopta vidrio_lente_izquierda como referencia común. "
    "Medir sobre el eje Z (no distancia oblicua), conservar equipo/calibración y sesión.",
    "m8": "30 búsquedas en 3 ambientes, >=600 s ausente, 20 cancelar, 10 pérdidas. "
    "Una fila por ensayo/intervalo no superpuesto y traza. inicio_s/fin_s son segundos "
    "desde el PRIMER t_ns del JSONL, no desde el primer frame del clip. "
    "resultado_real: exito/fallo/ausente/cancelar/perdida; exito_s es tiempo desde inicio_s "
    "en que la persona encuentra el objeto, no el detector; si queda vacío se toma de la "
    "traza, cuando la persona dice «la encontré». Falso aviso = bip o «Veo …» con el objeto "
    "ausente. request_id vincula audio y orden. "
    "Guardar una traza nueva por ejecución; no concatenar relojes monotónicos.",
    "ocr": "30 textos distintos, cada uno a 0,5 y 1 m (60 filas). "
    "Transcribir TODO el texto visible en orden de lectura, incluidos números. "
    "t_s desde el primer received_ns; no seleccionar por parecido al OCR.",
    "describir": "30 escenas independientes. objetos_json: lista de instancias "
    '[{"clase":"cup","zona":"a la izquierda","distancia_m":1.0}]. '
    "Anotar todas las instancias visibles antes de correr el modelo. Distancias aproximadas "
    "sirven para auditar, no como cinta M7. t_s desde el primer received_ns. "
    "La evaluación conserva texto y hechos para revisión humana de afirmaciones libres.",
    "voz": "61 frases de T08, con intención y objetivo de referencia. Hablar normal, "
    "registrar persona, equipo, micrófono, ambiente y distancia en notas. "
    "grabar no carga Whisper ni usa parlantes. Enter graba; r reintenta; s saltea; q guarda "
    "y sale. Repetir el comando retoma pendientes. evaluar procesa base y small sobre los "
    "mismos WAV. El silencio y la saturación se muestran y requieren aceptación explícita.",
}


def plantilla(metric: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "planilla.csv").exists():
        raise ValueError("ya existe planilla.csv; usar otra carpeta para no perder anotaciones")
    if metric == "m7":
        rows = [
            dict(
                objeto=f"objeto-{i:02}",
                clase_coco="",
                distancia_cinta_m=d,
                repeticion=r,
                instancia_x="",
                instancia_y="",
                clip="",
                t_s="",
                origen_cinta="vidrio_lente_izquierda",
                z_offset_m=0,
                sesion="",
                notas="",
            )
            for i in range(1, 11)
            for d in (0.5, 1, 2, 3)
            for r in (1, 2, 3)
        ]
    elif metric == "m8":
        kinds = ["fallo"] * 30 + ["ausente"] * 3 + ["cancelar"] * 20 + ["perdida"] * 10
        rows = [
            dict(
                ensayo=i,
                ambiente=f"ambiente-{(i - 1) % 3 + 1}",
                objeto="",
                resultado_real=k,
                traza="",
                request_id="",
                inicio_s="",
                fin_s="",
                exito_s="",
                notas="",
            )
            for i, k in enumerate(kinds, 1)
        ]
    elif metric == "ocr":
        rows = [
            dict(texto_id=i, clip="", t_s="", texto_referencia="", distancia_m=d, notas="")
            for i in range(1, 31)
            for d in (0.5, 1)
        ]
    elif metric == "describir":
        rows = [dict(escena=i, clip="", t_s="", objetos_json="", notas="") for i in range(1, 31)]
    elif metric == "voz":
        # Copia distribuible: no depender de tests/ al instalar el wheel.
        from .voice import phrases

        rows = phrases()
    else:
        raise ValueError(f"métrica desconocida: {metric}")
    write_csv(out / "planilla.csv", rows)
    (out / "GRABACION.md").write_text(GUIDES[metric] + "\n", encoding="utf-8")
    return out / "planilla.csv"
