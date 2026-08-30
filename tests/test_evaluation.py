from __future__ import annotations

import numpy as np
import pytest

from inspectflow.evaluation import evaluate_run
from inspectflow.generator import FrameGroundTruth, GeneratedFrame


def _truth(part_id: str, bbox: list[int], defect: str) -> dict:
    return {
        "part_id": part_id,
        "bbox": bbox,
        "centroid": [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2],
        "defect": defect,
        "verdict": "pass" if defect == "normal" else "fail",
    }


def _prediction(bbox: list[int], defect: str, track_id: int) -> dict:
    return {
        "bbox": bbox,
        "centroid": [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2],
        "predicted_defect": defect,
        "verdict": "pass" if defect == "normal" else "fail",
        "track_id": track_id,
    }


def test_metrics_and_confusion_matrix_use_ground_truth() -> None:
    image = np.zeros((20, 100, 3), dtype=np.uint8)
    frames = [
        GeneratedFrame(
            index=0,
            image=image,
            ground_truth=[
                _truth("normal", [0, 0, 10, 10], "normal"),
                _truth("color", [20, 0, 10, 10], "wrong_color"),
            ],
        ),
        GeneratedFrame(
            index=1,
            image=image,
            ground_truth=[_truth("missing", [40, 0, 10, 10], "missing_feature")],
        ),
    ]
    detections = [
        [
            _prediction([0, 0, 10, 10], "incorrect_dimensions", 1),
            _prediction([20, 0, 10, 10], "wrong_color", 2),
            _prediction([70, 0, 10, 10], "wrong_color", 3),
        ],
        [],
    ]

    report = evaluate_run(frames, detections)

    assert report["detection"] == {
        "true_positives": 2,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.666667,
        "recall": 0.666667,
        "f1": 0.666667,
    }
    assert report["classification"]["precision"] == 0.5
    assert report["classification"]["recall"] == 1.0
    assert report["classification"]["f1"] == 0.666667
    assert report["classification"]["confusion_matrix"] == {
        "labels": ["normal", "defective"],
        "matrix": [[0, 1], [0, 1]],
    }

    labels = report["defect_classification"]["labels"]
    matrix = report["defect_classification"]["confusion_matrix"]
    normal_row = matrix[labels.index("normal")]
    color_row = matrix[labels.index("wrong_color")]
    assert normal_row[labels.index("incorrect_dimensions")] == 1
    assert color_row[labels.index("wrong_color")] == 1
    assert report["defect_classification"]["ground_truth_distribution"] == {
        "normal": 1,
        "incorrect_dimensions": 0,
        "wrong_color": 1,
        "missing_feature": 1,
    }
    assert report["defect_classification"]["matched_ground_truth_distribution"] == {
        "normal": 1,
        "incorrect_dimensions": 0,
        "wrong_color": 1,
        "missing_feature": 0,
    }
    assert report["tracking"]["continuity"] is None
    assert report["totals"]["matched_objects"] == 2


def test_tracking_metric_counts_an_id_switch() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    frames = [
        GeneratedFrame(i, image, [_truth("part-a", [i, 0, 10, 10], "normal")])
        for i in range(3)
    ]
    detections = [
        [_prediction([0, 0, 10, 10], "normal", 5)],
        [_prediction([1, 0, 10, 10], "normal", 5)],
        [_prediction([2, 0, 10, 10], "normal", 9)],
    ]

    tracking = evaluate_run(frames, detections)["tracking"]

    assert tracking["tracked_transitions"] == 2
    assert tracking["id_switches"] == 1
    assert tracking["continuity"] == 0.5


def test_ambiguous_matching_maximizes_cardinality_before_iou() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    frames = [
        GeneratedFrame(
            0,
            image,
            [
                _truth("part-a", [0, 0, 10, 10], "normal"),
                _truth("part-b", [5, 0, 10, 10], "wrong_color"),
            ],
        )
    ]
    detections = [
        [
            _prediction([2, 0, 10, 10], "wrong_color", 1),
            _prediction([-2, 0, 10, 10], "normal", 2),
        ]
    ]

    report = evaluate_run(frames, detections)

    assert report["detection"]["true_positives"] == 2
    assert report["detection"]["false_negatives"] == 0
    assert report["classification"]["true_positives"] == 1
    assert report["classification"]["true_negatives"] == 1


def test_maximum_iou_breaks_ties_between_full_cardinality_matchings() -> None:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    frames = [
        GeneratedFrame(
            0,
            image,
            [
                _truth("part-a", [0, 0, 20, 10], "normal"),
                _truth("part-b", [10, 0, 20, 10], "wrong_color"),
            ],
        )
    ]
    detections = [
        [
            _prediction([1, 0, 20, 10], "normal", 1),
            _prediction([9, 0, 20, 10], "wrong_color", 2),
        ]
    ]

    report = evaluate_run(frames, detections, iou_threshold=0.3)

    assert report["classification"]["true_positives"] == 1
    assert report["classification"]["true_negatives"] == 1
    assert report["classification"]["false_positives"] == 0
    assert report["classification"]["false_negatives"] == 0


def test_image_free_truth_is_supported_and_all_inputs_are_validated_upfront() -> None:
    truth = FrameGroundTruth(
        0, [_truth("part-a", [0, 0, 10, 10], "missing_feature")]
    )
    prediction = _prediction([40, 0, 10, 10], "normal", 1)

    report = evaluate_run([truth], [[prediction]])
    assert report["totals"]["ground_truth_objects"] == 1
    assert report["defect_classification"]["ground_truth_distribution"][
        "missing_feature"
    ] == 1

    prediction["predicted_defect"] = "not-a-defect"
    with pytest.raises(ValueError, match="unknown predicted"):
        evaluate_run([truth], [[prediction]])

    prediction["predicted_defect"] = "normal"
    prediction["bbox"][0] = float("nan")
    with pytest.raises(ValueError, match="finite numeric bbox"):
        evaluate_run([truth], [[prediction]])

    with pytest.raises(ValueError, match="unknown frame"):
        evaluate_run([truth], {1: []})
