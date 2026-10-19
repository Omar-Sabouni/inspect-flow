from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from inspectflow import cli
from inspectflow.benchmark import format_markdown, run_benchmark
from inspectflow.video import inspect_video, render_scenario


def test_no_subcommand_still_starts_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append(kwargs))

    cli.main(["--port", "8123", "--log-level", "warning"])
    cli.main(["serve", "--host", "127.0.0.1"])

    assert calls[0]["port"] == 8123
    assert calls[0]["log_level"] == "warning"
    assert calls[1]["port"] == 8000


def test_invalid_port_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--port", "70000"])


def test_benchmark_aggregates_and_formats_markdown() -> None:
    report = run_benchmark(
        detectors=["classical"],
        scenarios=["nominal"],
        seeds=[7, 8],
        frame_count=40,
    )

    (group,) = report["summary"]
    assert group["detector"] == "classical"
    assert group["detection_recall"]["mean"] > 0.95
    assert group["escapes"] == {"total": 0}
    assert group["good_parts"] + group["defective_parts"] == group["parts_at_gate"]
    assert len(report["cases"]) == 2
    table = format_markdown(report)
    assert table.startswith("| Detector | Scenario | Gate verdict accuracy")
    assert "| classical | nominal |" in table
    assert f"0 of {group['defective_parts']}" in table
    assert "Seeds 7, 8, 40 frames each." in table


def test_bench_command_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "bench.json"
    cli.main(
        [
            "bench",
            "--detectors", "classical",
            "--scenarios", "nominal",
            "--seeds", "7",
            "--frames", "24",
            "--threads", "1",
            "--json", str(target),
        ]
    )

    printed = capsys.readouterr()
    assert "| classical | nominal |" in printed.out
    assert "seed 7" in printed.err
    assert json.loads(target.read_text())["settings"]["opencv_threads"] == 1


@pytest.mark.parametrize("detector", ["classical", "cnn"])
def test_rendered_scene_round_trips_through_the_video_pipeline(
    tmp_path: Path, detector: str
) -> None:
    scene = tmp_path / "scene.avi"
    annotated = tmp_path / "out" / "annotated.avi"
    log = tmp_path / "out" / "log.csv"
    assert render_scenario(scene, frame_count=60) == 60

    seen: list[dict] = []
    summary = inspect_video(
        str(scene),
        detector=detector,  # type: ignore[arg-type]
        output=annotated,
        log=log,
        on_decision=seen.append,
    )

    assert summary["frames"] == 60
    assert summary["accepted"] + summary["rejected"] == len(seen) >= 2
    assert annotated.stat().st_size > 0
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    assert len(rows) == len(seen)
    assert {"track_id", "verdict", "predicted_defect"}.issubset(rows[0])


def test_video_errors_are_reported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not open"):
        inspect_video(str(tmp_path / "missing.mp4"))
    with pytest.raises(ValueError, match="gate_fraction"):
        inspect_video("0", gate_fraction=1.5)
    with pytest.raises(ValueError, match=".mp4 or .avi"):
        render_scenario(tmp_path / "scene.mkv", frame_count=8)
    with pytest.raises(SystemExit, match="inspect-flow video"):
        cli.main(["video", str(tmp_path / "missing.mp4")])
