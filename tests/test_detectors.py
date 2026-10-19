from __future__ import annotations

import math

import numpy as np
import pytest

from inspectflow.detectors import (
    DEFECT_LABELS,
    MODEL_INPUT_SIZE,
    OUTPUT_CHANNELS,
    OUTPUT_STRIDE,
    SIZE_SCALE_PX,
    CnnPartDetector,
    decode_output,
    load_model_card,
    model_sha256,
)

ROWS = MODEL_INPUT_SIZE[1] // OUTPUT_STRIDE
COLUMNS = MODEL_INPUT_SIZE[0] // OUTPUT_STRIDE


def _output_with_peak(
    row: int,
    column: int,
    label: str,
    *,
    score: float = 0.9,
    size: tuple[float, float] = (80.0, 50.0),
    offset: tuple[float, float] = (0.5, 0.5),
    output: np.ndarray | None = None,
) -> np.ndarray:
    if output is None:
        output = np.zeros((OUTPUT_CHANNELS, ROWS, COLUMNS), dtype=np.float32)
    count = len(DEFECT_LABELS)
    output[DEFECT_LABELS.index(label), row, column] = score
    output[count, row, column] = offset[0]
    output[count + 1, row, column] = offset[1]
    output[count + 2, row, column] = math.log(size[0] / SIZE_SCALE_PX)
    output[count + 3, row, column] = math.log(size[1] / SIZE_SCALE_PX)
    return output


def test_decode_recovers_box_class_and_score() -> None:
    output = _output_with_peak(15, 30, "wrong_color")

    detections = decode_output(
        output, image_size=MODEL_INPUT_SIZE, score_threshold=0.4
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection["predicted_defect"] == "wrong_color"
    assert detection["verdict"] == "fail"
    assert detection["score"] == pytest.approx(0.9)
    assert detection["centroid"] == [30.5 * OUTPUT_STRIDE, 15.5 * OUTPUT_STRIDE]
    assert detection["bbox"] == [204, 99, 80, 50]
    assert detection["measurements"] == {"width_px": 80, "height_px": 50}


def test_decode_scales_boxes_to_the_original_frame_size() -> None:
    output = _output_with_peak(15, 30, "normal")

    (detection,) = decode_output(
        output, image_size=(1280, 640), score_threshold=0.4
    )

    assert detection["bbox"] == [408, 198, 160, 100]
    assert detection["verdict"] == "pass"


def test_decode_applies_threshold_border_margin_and_suppression() -> None:
    output = _output_with_peak(15, 30, "normal", score=0.3)
    # Box touching the left edge.
    _output_with_peak(15, 2, "normal", output=output)
    # Two neighbouring peaks on one part: only the stronger survives NMS.
    _output_with_peak(20, 50, "missing_feature", score=0.8, output=output)
    _output_with_peak(20, 52, "normal", score=0.7, output=output)

    detections = decode_output(
        output, image_size=MODEL_INPUT_SIZE, score_threshold=0.4
    )

    assert [item["predicted_defect"] for item in detections] == ["missing_feature"]


def test_decode_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="output must have shape"):
        decode_output(np.zeros((3, 4, 4)), image_size=(32, 32), score_threshold=0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_threshold": 0.0},
        {"score_threshold": 1.5},
        {"score_threshold": True},
        {"nms_iou": float("nan")},
        {"border_margin_px": -1},
    ],
)
def test_detector_rejects_invalid_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CnnPartDetector(**kwargs)


def test_bundled_model_matches_its_training_record() -> None:
    card = load_model_card()

    assert card["sha256"] == model_sha256()
    assert card["labels"] == list(DEFECT_LABELS)
    assert card["input"]["width"] == MODEL_INPUT_SIZE[0]
    assert card["input"]["height"] == MODEL_INPUT_SIZE[1]
    assert card["output_stride"] == OUTPUT_STRIDE


def test_bundled_model_runs_and_rejects_bad_images() -> None:
    detector = CnnPartDetector()
    blank = np.zeros((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), dtype=np.uint8)

    assert detector.forward(blank).shape == (OUTPUT_CHANNELS, ROWS, COLUMNS)
    assert detector.inspect(blank) == []
    with pytest.raises(ValueError, match="uint8"):
        detector.inspect(blank.astype(np.float32))
