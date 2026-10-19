from __future__ import annotations

from typing import Any

from inspectflow.evaluation import evaluate_gate
from inspectflow.generator import FrameGroundTruth


def _truth(part_id: str, x: int, defect: str, y: int = 0) -> dict[str, Any]:
    return {
        "part_id": part_id,
        "bbox": [x, y, 20, 10],
        "centroid": [x + 10.0, y + 5.0],
        "defect": defect,
        "verdict": "pass" if defect == "normal" else "fail",
    }


def _decision(x: int, defect: str, frame: int, y: int = 0) -> dict[str, Any]:
    return {
        "bbox": [x, y, 20, 10],
        "centroid": [x + 10.0, y + 5.0],
        "predicted_defect": defect,
        "verdict": "pass" if defect == "normal" else "fail",
        "decision_frame": frame,
    }


def _frames() -> list[FrameGroundTruth]:
    # Five parts on separate rows. Each centroid moves from x=95 to x=105, so
    # every part crosses a gate at x=100 on frame 1.
    defects = ["normal", "normal", "wrong_color", "missing_feature", "wrong_color"]
    return [
        FrameGroundTruth(
            index,
            [
                _truth(f"p{row}", 85 + 10 * index, defect, y=20 * row)
                for row, defect in enumerate(defects)
            ],
        )
        for index in range(2)
    ]


def test_gate_outcomes_cover_escape_false_reject_wrong_defect_and_miss() -> None:
    decisions = [
        _decision(95, "normal", 1, y=0),  # p0: correct pass
        _decision(95, "missing_feature", 1, y=20),  # p1: false reject
        _decision(95, "normal", 1, y=40),  # p2: escape
        _decision(95, "wrong_color", 1, y=60),  # p3: wrong defect
        _decision(95, "wrong_color", 1, y=60),  # p3 again: spurious
        _decision(300, "wrong_color", 1, y=0),  # nothing there: spurious
    ]

    report = evaluate_gate(decisions, _frames(), gate_x=100)

    assert report["counts"] == {
        "correct_pass": 1,
        "correct_reject": 0,
        "wrong_defect": 1,
        "escape": 1,
        "false_reject": 1,
        "spurious": 2,
        "missed": 1,
    }
    assert report["parts"] == 5
    assert report["good_parts"] == 2
    assert report["defective_parts"] == 3
    assert report["escape_rate"] == round(1 / 3, 6)
    assert report["false_reject_rate"] == 0.5
    assert report["miss_rate"] == 0.2
    # Correct pass, plus the wrong-defect reject still has the right verdict.
    assert report["verdict_accuracy"] == 0.4
    assert [item["outcome"] for item in report["outcomes"]] == [
        "correct_pass",
        "false_reject",
        "escape",
        "wrong_defect",
        "spurious",
        "spurious",
    ]
    assert report["outcomes"][2]["truth_part_id"] == "p2"
    assert report["outcomes"][5]["truth_part_id"] == ""


def test_parts_that_never_cross_are_not_counted_as_missed() -> None:
    frames = [FrameGroundTruth(0, [_truth("p0", 10, "normal")])]

    report = evaluate_gate([], frames, gate_x=100)

    assert report["parts"] == 0
    assert report["counts"]["missed"] == 0
    assert report["verdict_accuracy"] == 0.0
