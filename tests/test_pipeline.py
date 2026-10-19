from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np
import pytest

from inspectflow.inspection import PartInspector
from inspectflow.pipeline import (
    InspectionPipeline,
    annotate_frame,
    StageTimer,
    decision_rows,
    decision_timeline,
    make_detector,
    rows_to_csv,
)
from inspectflow.tracking import CentroidTracker


class ScriptedDetector:
    """Return prepared detections per call, ignoring the image."""

    name = "scripted"

    def __init__(self, frames: list[list[dict[str, Any]]]) -> None:
        self._frames = iter(frames)

    def inspect(self, image: np.ndarray) -> list[dict[str, Any]]:
        return [dict(item) for item in next(self._frames)]


def _detection(x: float, defect: str = "normal", y: float = 20.0) -> dict[str, Any]:
    return {
        "bbox": [int(x) - 5, int(y) - 5, 10, 10],
        "centroid": [x, y],
        "predicted_defect": defect,
        "verdict": "pass" if defect == "normal" else "fail",
        "measurements": {"width_px": 10, "height_px": 10},
    }


def _run(frames: list[list[dict[str, Any]]], gate_x: float = 100) -> InspectionPipeline:
    pipeline = InspectionPipeline(
        ScriptedDetector(frames),
        gate_x=gate_x,
        tracker=CentroidTracker(max_distance=30, max_missed=2),
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    for index in range(len(frames)):
        pipeline.process(image, index)
    return pipeline


def test_one_decision_per_confirmed_left_to_right_crossing() -> None:
    pipeline = _run(
        [
            [_detection(90), _detection(130, y=60)],
            [_detection(105), _detection(140, y=60)],
            [_detection(120), _detection(150, y=60)],
            [_detection(135), _detection(160, y=60)],
        ]
    )

    decisions = pipeline.decisions
    # The track first seen beyond the gate stays undecided.
    assert [(item["track_id"], item["decision_frame"]) for item in decisions] == [
        (1, 1)
    ]
    assert decisions[0]["centroid"] == [105.0, 20.0]
    assert pipeline.observed_tracks == 2


def test_right_to_left_motion_and_early_disappearance_are_undecided() -> None:
    pipeline = _run(
        [
            [_detection(130), _detection(50, y=60)],
            [_detection(110), _detection(80, y=60)],
            [_detection(90)],
        ]
    )

    assert pipeline.decisions == []
    assert pipeline.observed_tracks == 2


def test_decision_is_reported_on_the_crossing_frame_only() -> None:
    pipeline = InspectionPipeline(
        ScriptedDetector([[_detection(95)], [_detection(101)], [_detection(108)]]),
        gate_x=100,
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    _, first = pipeline.process(image, 0)
    _, second = pipeline.process(image, 1)
    _, third = pipeline.process(image, 2)

    assert first == []
    assert [item["decision_frame"] for item in second] == [1]
    assert third == []


def test_timeline_and_csv_rows_follow_decisions() -> None:
    pipeline = _run(
        [
            [_detection(90), _detection(60, "wrong_color", y=60)],
            [_detection(105), _detection(80, "wrong_color", y=60)],
            [_detection(120), _detection(101, "wrong_color", y=60)],
        ]
    )
    decisions = pipeline.decisions

    timeline = decision_timeline(decisions, 3)
    assert [(item["accepted"], item["rejected"]) for item in timeline] == [
        (0, 0),
        (1, 0),
        (1, 1),
    ]
    assert timeline[-1]["defect_counts"]["wrong_color"] == 1

    rows = decision_rows(decisions, [{"note": "a"}, {"note": "b"}])
    text = rows_to_csv(rows, [*rows[0].keys()])
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert [row["note"] for row in parsed] == ["a", "b"]
    assert parsed[0]["measured_width_px"] == "10"
    with pytest.raises(ValueError, match="one entry per decision"):
        decision_rows(decisions, [{}])


def test_stage_timer_summarizes_each_stage() -> None:
    timer = StageTimer()
    for value in (1.0, 2.0, 3.0, 4.0):
        timer.add("detect", value)
    with timer.measure("track"):
        pass

    summary = timer.summary()
    assert summary["detect"]["mean_ms"] == 2.5
    assert summary["detect"]["max_ms"] == 4.0
    assert summary["detect"]["p50_ms"] == 2.5
    assert summary["track"]["mean_ms"] >= 0.0


def test_make_detector_selects_backend() -> None:
    assert isinstance(make_detector("classical"), PartInspector)
    assert make_detector("cnn").name == "cnn"
    with pytest.raises(ValueError, match="unknown detector"):
        make_detector("yolo")  # type: ignore[arg-type]


def test_colliding_labels_move_below_their_box() -> None:
    image = np.full((200, 200, 3), 200, dtype=np.uint8)
    first = {**_detection(40), "bbox": [10, 40, 60, 30], "track_id": 1}
    second = {**_detection(70), "bbox": [40, 45, 60, 30], "track_id": 2}

    alone = annotate_frame(image, [second], frame_index=0, label="test", gate_x=190)
    together = annotate_frame(
        image, [first, second], frame_index=0, label="test", gate_x=190
    )

    label_background = [19, 24, 31]
    # Alone, the second label sits above its box; with a neighbour it moves
    # below the box so the two labels do not overlap.
    assert alone[38, 41].tolist() == label_background
    assert alone[80, 41].tolist() != label_background
    assert together[80, 41].tolist() == label_background
