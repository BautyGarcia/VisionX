"""M8: verdad humana + eventos existentes. No confundir callback final con inicio audible."""

import json
from pathlib import Path

from .common import number, read_csv, report, resolve


def evaluate(sheet: Path, out: Path, *, trace: Path | None = None):
    annotations = read_csv(sheet)
    loaded, intervals, rows = {}, {}, []
    total_absent = false_beeps = searches = successes = cancels = losses = 0
    late_cancel = late_deadline = suspects = 0
    audio_without_start = 0
    audios = missing_cancel = missing_request = 0
    for r in annotations:
        path = trace.resolve() if trace else resolve(sheet, r["traza"])
        if path not in loaded:
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not events or any("t_ns" not in e or "tipo" not in e for e in events):
                raise ValueError(f"{path}: traza vacía o sin t_ns/tipo")
            if any(b["t_ns"] < a["t_ns"] for a, b in zip(events, events[1:], strict=False)):
                raise ValueError("traza fuera de orden: no concatenar ejecuciones")
            loaded[path] = events
        events = loaded[path]
        start, end = number(r["inicio_s"], "inicio"), number(r["fin_s"], "fin")
        if end <= start:
            raise ValueError("fin_s debe superar inicio_s")
        for a, b in intervals.setdefault(path, []):
            if max(a, start) < min(b, end):
                raise ValueError("intervalos superpuestos duplican denominadores")
        intervals[path].append((start, end))
        origin = events[0]["t_ns"]
        if end > (events[-1]["t_ns"] - origin) / 1e9 + 1e-9:
            raise ValueError("fin_s fuera de la cobertura temporal de la traza")
        selected = [e for e in events if start <= (e["t_ns"] - origin) / 1e9 < end]
        kind = r["resultado_real"]
        if kind not in ("exito", "fallo", "ausente", "cancelar", "perdida"):
            raise ValueError(f"resultado_real desconocido: {kind}")
        if not r["request_id"].strip():
            raise ValueError("request_id requerido para asociar cada ensayo")
        request = int(r["request_id"])
        if not any(e.get("request_id") == request for e in events):
            raise ValueError(f"request_id {request} no existe en la traza: {path}")
        scoped = [e for e in selected if e.get("request_id") == request]
        success = False
        if kind in ("exito", "fallo"):
            searches += 1
            if kind == "exito":
                elapsed = number(r["exito_s"], "exito_s")
                if elapsed > end - start:
                    raise ValueError("exito_s fuera del ensayo")
                success = elapsed <= 60
                successes += success
        elif kind == "cancelar":
            cancels += 1
        elif kind == "perdida":
            losses += 1
        beeps = sum(
            e["tipo"] == "audio"
            and e.get("texto") == "sonido:bip"
            and e.get("resultado") == "reproducido"
            for e in scoped
        )
        if kind == "ausente":
            total_absent += end - start
            false_beeps += beeps
        cancelled = next((e["t_ns"] for e in scoped if e["tipo"] == "cancelado"), None)
        missing_cancel += int(kind == "cancelar" and cancelled is None)
        expired = next((e["t_ns"] for e in scoped if e["tipo"] == "vencido"), None)
        for e in scoped:
            if e["tipo"] != "audio" or e.get("texto") == "sonido:cancelado":
                continue
            if e.get("resultado") not in ("reproducido", "cancelado", "interrumpido"):
                continue
            audios += 1
            started = e.get("started_ns")
            if started is None:
                audio_without_start += 1
                suspects += int(cancelled is not None and e["t_ns"] > cancelled)
            else:
                late_cancel += int(cancelled is not None and started > cancelled)
                late_deadline += int(expired is not None and started > expired)
        rows.append(
            {
                **r,
                "exito_60s": success if kind in ("exito", "fallo") else None,
                "bips_reproducidos": beeps,
                "eventos": len(scoped),
                "cancelacion_en_traza": cancelled is not None,
                "perdida_en_traza": any(e["tipo"] == "buscar-perdido" for e in scoped),
            }
        )
    summary = {
        "n_busquedas": searches,
        "exitos_60s": successes,
        "exito_60s_fraccion": successes / searches if searches else None,
        "ausente_s": total_absent,
        "falsos_bips_software": false_beeps,
        "falsos_bips_por_min": false_beeps / (total_absent / 60) if total_absent else None,
        "n_cancelar": cancels,
        "n_perdidas": losses,
        "respuestas_tras_cancelar": late_cancel
        if audios and not (audio_without_start or missing_cancel or missing_request)
        else None,
        "audio_tras_vencimiento": late_deadline
        if audios and not (audio_without_start or missing_request)
        else None,
        "ensayos_sin_request_en_traza": missing_request,
        "violaciones_inicio_observadas": late_cancel + late_deadline,
        "callbacks_posteriores_cancelar_a_revisar": suspects,
        "guia_vencida": None,
        "limitaciones": "Sin edad/frame de guía no se verifica guía vencida. "
        "Sin started_ns no se verifica inicio tras cancelar/vencer. "
        "Bips son callbacks reproducidos, no medición audible; revisar bordes de intervalos.",
        "protocolo_completo": searches >= 30
        and len({r["ambiente"] for r in rows if r["resultado_real"] in ("exito", "fallo")}) >= 3
        and total_absent >= 600
        and cancels >= 20
        and losses >= 10,
    }
    return report(out, rows, summary, [sheet, *loaded])
