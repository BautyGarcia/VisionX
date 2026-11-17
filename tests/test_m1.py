"""M1 en CI: banco reducido sin pesos/dispositivos y pruebas de sus oráculos."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from visionx.app.__main__ import main
from visionx.app.controller import Say
from visionx.app.m1 import CORE_CASES, EXTRA_CASES, run_bank, write_results
from visionx.modes.distance import DistanceHandler
from visionx.modes.search import SearchHandler


@pytest.fixture(scope="module")
def reduced():
    return run_bank(1)


def test_reduced_bank_covers_contract_and_faults(reduced):
    summary, rows = reduced
    assert summary["base_total"] == 8
    assert {r["caso"] for r in summary["corridas"]} == set(CORE_CASES + EXTRA_CASES)
    # Deuda hallada por M1: AudioEvent conserva deadline de orden, no de observación.
    # CI exige que se declare FALLA; admite que el coordinador repare esa vigencia.
    for run in summary["corridas"]:
        unexpected = {key for key, passed in run["criterios"].items() if not passed}
        assert unexpected <= (
            {"guia_en_cola_vieja_no_suena"} if run["caso"] == "vencido" else set()
        )
    stale = next(r for r in summary["corridas"] if r["caso"] == "vencido")
    has_stale_audio = not stale["criterios"]["guia_en_cola_vieja_no_suena"]
    assert summary["resultado"] == ("FALLA" if has_stale_audio else "PASA")
    assert summary["base_correctos"] == (7 if has_stale_audio else 8)
    assert summary["criterios"]["cero_audio_viejo"] == ("FALLA" if has_stale_audio else "PASA")
    runs = {r["caso"]: r for r in summary["corridas"]}
    assert runs["presente"]["frases_iniciadas"] == [
        "Busco taza.",
        "sonido:bip",
        "Taza un poco a la izquierda.",
    ]
    assert runs["ausente"]["frases_iniciadas"] == ["Busco taza.", "No encontré taza."]
    assert runs["cancelar-tts"]["frases_iniciadas"] == ["sonido:cancelado"]
    assert runs["vencido-tts"]["frases_iniciadas"] == []
    assert any(r["evento"] == "cancelar-inicio" for r in rows)
    for name in ("mascara-vacia", "depth-invalida"):
        assert runs[name]["frases_iniciadas"] == [
            "sonido:ok",
            "Taza un poco a la izquierda, pero no puedo medir su distancia.",
        ]


def test_metrics_separate_virtual_and_wall_and_preserve_sample_counts(reduced):
    summary, rows = reduced
    metrics = summary["metricas"]
    assert metrics["cancelacion_logica_ms"]["n"] == 5
    assert metrics["cancelacion_logica_ms"]["p95"] == 0
    assert metrics["deteccion_bip_logica_ms"]["p95"] == 140
    assert metrics["edad_dicho_ms"]["p50"] == 20
    assert metrics["cancelacion_pared_ms"]["n"] == 5
    assert metrics["deteccion_bip_pared_ms"]["n"] > 0
    assert all(r["wall_ns"] >= 0 for r in rows)
    assert "no mide" in summary["limites"].lower()


def test_report_contains_replayable_event_details(tmp_path, reduced):
    summary, rows = reduced
    write_results(summary, rows, tmp_path)
    loaded = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert loaded["resultado"] == summary["resultado"]
    assert loaded["codigo_sin_commit"] in (True, False, None)
    assert loaded["sha256_codigo"]["app/m1.py"]
    with (tmp_path / "eventos.csv").open(encoding="utf-8", newline="") as stream:
        events = list(csv.DictReader(stream))
    assert len(events) == len(rows)
    played = [json.loads(r["datos_json"]) for r in events if r["evento"] == "audio-record"]
    assert any(p["outcome"] == "cancelado" and p["started_ns"] is None for p in played)
    assert "| presente | PASA 1/1 |" in (tmp_path / "tabla.md").read_text(encoding="utf-8")


def test_bank_detects_missing_bip(monkeypatch):
    original = SearchHandler.tick

    def no_bip(self, command, view, now_ns):
        says, done = original(self, command, view, now_ns)
        return [s for s in says if s.text_or_sound != "sonido:bip"], done

    monkeypatch.setattr(SearchHandler, "tick", no_bip)
    summary, _ = run_bank(1, cases=("presente",))
    assert summary["resultado"] == "FALLA"
    assert not summary["corridas"][0]["criterios"]["un_bip"]
    assert summary["criterios"]["deteccion_bip_p95_500_ms"] == "FALLA"


def test_bank_rejects_invented_meters(monkeypatch):
    def invent(self, command, view, now_ns):
        return [Say("Taza a 9 metros.")], True

    monkeypatch.setattr(DistanceHandler, "tick", invent)
    summary, _ = run_bank(1, cases=("depth-invalida",))
    assert summary["resultado"] == "FALLA"
    assert not summary["corridas"][0]["criterios"]["sin_metros"]


def test_cli_writes_results_and_failure_exit_code(tmp_path, reduced, monkeypatch, capsys):
    from visionx.app import m1

    summary, rows = reduced
    monkeypatch.setattr(
        m1, "run_bank", lambda repetitions: ({**summary, "resultado": "PASA"}, rows)
    )
    assert main(["m1", "--repeticiones", "1", "--salida", str(tmp_path)]) == 0
    assert "PASA" in capsys.readouterr().out
    monkeypatch.setattr(
        m1, "run_bank", lambda repetitions: ({**summary, "resultado": "FALLA"}, rows)
    )
    assert main(["m1", "--salida", str(tmp_path)]) == 1


def test_bank_rejects_invalid_repetitions():
    with pytest.raises(ValueError, match="repeticiones"):
        run_bank(0)


@pytest.mark.parametrize("option", ["--repeticiones", "--cada"])
def test_cli_rejects_nonpositive_counts(option):
    with pytest.raises(SystemExit) as exc:
        main(["m1", option, "0"])
    assert exc.value.code == 2


def test_clip_path_uses_real_controller_and_records_presence_loss(tmp_path, monkeypatch):
    from visionx.app import m1_clip
    from visionx.app.fakes import FakeDepthEstimator, ScriptedDetector, SyntheticStereoSource
    from visionx.app.spatial import ProvisionalSpatialEstimator

    def inputs(args, clock, cfg):
        class TimedSource(SyntheticStereoSource):
            def read(self):
                clock.advance(100_000_000)
                if self.count >= 5:
                    self.objects = ()
                self.count += 1
                return super().read()

        source = TimedSource(fps=None, n_frames=50, clock=clock)
        source.count = 0
        return (
            source,
            ScriptedDetector(source),
            FakeDepthEstimator(source.geometry, source=source),
            ProvisionalSpatialEstimator(source.geometry),
        )

    monkeypatch.setattr(m1_clip, "clip_inputs", inputs)
    args = SimpleNamespace(cada=1, config=None, ordenes=None, clip="fake-clip-de-test")
    result = m1_clip.run_clip(args, tmp_path)
    assert result["criterios"] == {
        "sin_errores_modelos": "PASA",
        "presente": "PASA",
        "perdido": "PASA",
    }
    assert result["frames"] == 50
    assert result["metricas"] == {}  # RecordingSink no afirma medir reproducción
    assert (tmp_path / "eventos.csv").is_file()
