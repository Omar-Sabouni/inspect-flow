"""The CNN part detector, run from ONNX through OpenCV's DNN module.

The network (``training/train_detector.py``) predicts, for every cell of a
stride-8 grid, a heatmap per defect class, a sub-cell center offset and a
log box size. Decoding keeps local heatmap peaks, so there are no anchors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np

from .evaluation import box_iou
from .generator import DEFECTS

DetectorName = Literal["classical", "cnn"]
DETECTOR_NAMES: tuple[DetectorName, ...] = ("classical", "cnn")

MODEL_PATH = Path(__file__).parent / "models" / "part_detector.onnx"
INPUT_SIZE = (640, 320)  # width, height
STRIDE = 8
SIZE_SCALE_PX = 64.0  # the size head predicts log(size / SIZE_SCALE_PX)
OUTPUT_CHANNELS = len(DEFECTS) + 4


class Detector(Protocol):
    name: str

    def inspect(self, image: np.ndarray) -> list[dict[str, Any]]: ...


def model_record(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """The training record written next to the model."""

    record: dict[str, Any] = json.loads(model_path.with_suffix(".json").read_text())
    return record


def model_sha256(model_path: Path = MODEL_PATH) -> str:
    return hashlib.sha256(model_path.read_bytes()).hexdigest()


class CnnPartDetector:
    """Threshold and border margin were tuned on validation seeds 100-104.

    The 5 px margin matches the ground truth, which skips parts within 6 px of
    the frame edge.
    """

    name = "cnn"

    def __init__(
        self, model_path: Path = MODEL_PATH, score_threshold: float = 0.4, border_margin_px: int = 5
    ) -> None:
        self.score_threshold = score_threshold
        self.border_margin_px = border_margin_px
        # A cv2.dnn.Net is not thread-safe, so each detector loads its own.
        self._net = cv2.dnn.readNetFromONNX(str(model_path))

    def forward(self, image: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(image, 1 / 255, INPUT_SIZE)
        self._net.setInput(blob)
        return np.asarray(self._net.forward())[0]

    def inspect(self, image: np.ndarray) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        return decode(
            self.forward(image), (width, height), self.score_threshold, self.border_margin_px
        )


def decode(
    output: np.ndarray,
    image_size: tuple[int, int],
    score_threshold: float = 0.4,
    border_margin_px: int = 5,
    nms_iou: float = 0.5,
) -> list[dict[str, Any]]:
    """Turn a (channels, rows, cols) network output into detections.

    Peaks are found on the per-cell maximum over classes, so one part yields
    one detection however its score is split between classes.
    """

    width, height = image_size
    scale_x, scale_y = width / INPUT_SIZE[0], height / INPUT_SIZE[1]
    n = len(DEFECTS)
    scores = output[:n].max(axis=0)
    classes = output[:n].argmax(axis=0)
    is_peak = (scores >= score_threshold) & (scores >= cv2.dilate(scores, np.ones((3, 3), np.uint8)))

    found: list[dict[str, Any]] = []
    for row, col in zip(*np.nonzero(is_peak), strict=True):
        cx = (col + output[n, row, col]) * STRIDE * scale_x
        cy = (row + output[n + 1, row, col]) * STRIDE * scale_y
        box_w = float(np.exp(output[n + 2, row, col])) * SIZE_SCALE_PX * scale_x
        box_h = float(np.exp(output[n + 3, row, col])) * SIZE_SCALE_PX * scale_y
        x, y = int(round(cx - box_w / 2)), int(round(cy - box_h / 2))
        w, h = max(1, int(round(box_w))), max(1, int(round(box_h)))
        m = border_margin_px
        if x <= m or y <= m or x + w >= width - m or y + h >= height - m:
            continue
        defect = DEFECTS[int(classes[row, col])]
        found.append(
            {
                "bbox": [x, y, w, h],
                "centroid": [round(float(cx), 3), round(float(cy), 3)],
                "predicted_defect": defect,
                "verdict": "pass" if defect == "normal" else "fail",
                "score": round(float(scores[row, col]), 4),
                "measurements": {"width_px": w, "height_px": h},
            }
        )

    kept: list[dict[str, Any]] = []
    for candidate in sorted(found, key=lambda d: -d["score"]):
        if all(box_iou(candidate["bbox"], other["bbox"]) < nms_iou for other in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda d: d["centroid"])

