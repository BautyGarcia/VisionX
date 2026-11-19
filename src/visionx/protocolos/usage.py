"""M8: verdad humana + eventos existentes. No confundir callback final con inicio audible."""

import json
import os
from pathlib import Path

from .common import number, read_csv, report, resolve


def evaluate(sheet: Path, out: Path, *, trace: Path | None = None):
    annotations = read_csv(sheet)
    loaded, intervals, rows = {}, {}, []
    total_absent = false_beeps = false_sees = searches = successes = cancels = losses = 0
    late_cancel = late_deadline = suspects = stale_guides = 0
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
                if str(r.get("exito_s", "")).strip():
                    elapsed = number(r["exito_s"], "exito_s")
                else:
                    # Buscar ya no termina sola (19/11): la persona dice «la encontré» y el
                    # controlador lo registra como `terminado` con motivo `encontrado`.
                    found = next((e for e in scoped if e["tipo"] == "terminado"
                                  and e.get("motivo") == "encontrado"), None)  # fmt: skip
                    if found is None:
                        raise ValueError(
                            f"ensayo {r.get('ensayo')}: éxito sin exito_s ni "
                            "«la encontré» en la traza"
                        )
                    elapsed = (found["t_ns"] - origin) / 1e9 - start
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
        # Con la guía «centrar» no hay bip: el aviso de presencia es «Veo taza.».
        sees = sum(
            e["tipo"] == "audio"
            and str(e.get("texto", "")).startswith("Veo ")
            and e.get("resultado") == "reproducido"
            for e in scoped
        )
        if kind == "ausente":
            total_absent += end - start
            false_beeps += beeps
            false_sees += sees
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
                valid_until = e.get("valid_until_ns")
                stale_guides += int(valid_until is not None and started > valid_until)
        rows.append(
            {
                **r,
                "exito_60s": success if kind in ("exito", "fallo") else None,
                "bips_reproducidos": beeps,
                "avisos_veo_reproducidos": sees,
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
        "falsos_avisos_veo": false_sees,
        "falsos_avisos_por_min": (false_beeps + false_sees) / (total_absent / 60)
        if total_absent
        else None,
        "n_cancelar": cancels,
        "n_perdidas": losses,
        "respuestas_tras_cancelar": late_cancel
        if audios and not (audio_without_start or missing_cancel or missing_request)
        else None,
        "audio_tras_vencimiento": late_deadline
        if audios and not (audio_without_start or missing_request)
        else None,
        "ensayos_sin_request_en_traza": missing_request,
        "callbacks_posteriores_cancelar_a_revisar": suspects,
        # Frase con vigencia del hecho (guía de buscar) que empezó a sonar ya vencida.
        "guia_vencida": stale_guides if audios and not audio_without_start else None,
        "violaciones_inicio_observadas": late_cancel + late_deadline + stale_guides,
        "limitaciones": "Sin started_ns no se verifica inicio tras cancelar/vencer "
        "ni guía vencida. "
        "Bips son callbacks reproducidos, no medición audible; revisar bordes de intervalos.",
        "protocolo_completo": searches >= 30
        and len({r["ambiente"] for r in rows if r["resultado_real"] in ("exito", "fallo")}) >= 3
        and total_absent >= 600
        and cancels >= 20
        and losses >= 10,
    }
    return report(out, rows, summary, [sheet, *loaded])


REVISAR = "REVISAR-fallo-o-ausente"


def borrador(trace: Path, out: Path, ambiente: str) -> list[dict]:
    """Borrador de la planilla M8 desde una traza en vivo: una fila por «buscá…».

    El resultado se sugiere por cómo terminó cada búsqueda: «la encontré» -> exito (exito_s
    queda vacío: lo saca `evaluar` de la traza); cancelar -> cancelar; «Perdí» sin encontrar ->
    perdida; «terminá la búsqueda» -> REVISAR (fallo si el objeto estaba, ausente si no). La
    persona revisa todo antes de evaluar: `evaluar` rechaza REVISAR.
    """
    from .common import write_csv

    events = [json.loads(x) for x in trace.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not events:
        raise ValueError(f"{trace}: traza vacía")
    origin = events[0]["t_ns"]
    searches = [
        (i, e) for i, e in enumerate(events)
        if e.get("tipo") == "orden" and e.get("modo") == "search" and e.get("request_id")
    ]  # fmt: skip
    rows = []
    for k, (i, order) in enumerate(searches):
        rid = order["request_id"]
        stop = searches[k + 1][0] if k + 1 < len(searches) else len(events)
        mine = [e for e in events[i:stop] if e.get("request_id") == rid]
        ended = next((e for e in mine if e["tipo"] in ("terminado", "cancelado", "vencido")), None)
        motive = (ended or {}).get("motivo")
        if ended is not None and ended["tipo"] == "cancelado":
            result = "cancelar"
        elif motive == "encontrado":
            result = "exito"
        elif any(e["tipo"] == "buscar-perdido" for e in mine):
            result = "perdida"
        else:
            result = REVISAR
        # El ensayo dura hasta un poco después de su fin (lo que suena tras «la encontré») o
        # hasta la próxima búsqueda; nunca se superpone con la siguiente.
        last = events[stop - 1]["t_ns"] if stop == len(events) else events[stop]["t_ns"] - 1
        end_ns = min(last, ended["t_ns"] + 2_000_000_000) if ended else last
        rows.append({
            "ensayo": len(rows) + 1, "ambiente": ambiente, "objeto": order.get("objetivo") or "",
            "resultado_real": result, "request_id": rid,
            # Relativa a la planilla, como la resuelve `evaluar`.
            "traza": os.path.relpath(trace.resolve(), out.resolve().parent).replace("\\", "/"),
            "inicio_s": round((order["t_ns"] - origin) / 1e9, 3),
            "fin_s": round((end_ns - origin) / 1e9, 3), "exito_s": "",
            "notas": f"dicho: {order.get('texto', '')}",
        })  # fmt: skip
    write_csv(out, rows)
    return rows
