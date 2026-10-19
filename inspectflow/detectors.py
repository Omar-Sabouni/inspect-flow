"""Learned part detector served through OpenCV's DNN module.

The network is a small CenterNet-style model trained by
``training/train_detector.py``. For every cell of an output grid with stride
``OUTPUT_STRIDE`` it predicts one center heatmap per defect class, a sub-cell
center offset, and the log of the box size. Decoding keeps local heatmap maxima
above a score threshold, so no anchor boxes are needed.

Inference uses ``cv2.dnn`` so the runtime needs no dependency beyond
``opencv-python-headless``. PyTorch is only needed to retrain the model.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.resources import files
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np


DetectorName = Literal["classical", "cnn"]
DETECTOR_NAMES: tuple[DetectorName, ...] = ("classical", "cnn")

DEFECT_LABELS = (
    "normal",
    "incorrect_dimensions",
    "wrong_color",
    "missing_feature",
)
# Model input as (width, height). Other frame sizes are resized to this shape
# and the boxes are scaled back, which changes the pixel meaning of part size.
MODEL_INPUT_SIZE = (640, 320)
OUTPUT_STRIDE = 8
# The size head predicts log(size_px / SIZE_SCALE_PX).
SIZE_SCALE_PX = 64.0
# Output channels: one heatmap per label, then x/y offset, then log w/h.
OUTPUT_CHANNELS = len(DEFECT_LABELS) + 4


class Detector(Protocol):
    """Anything that turns one BGR frame into detection dictionaries."""

    name: str

    def inspect(self, image: np.ndarray) -> list[dict[str, Any]]: ...


def default_model_path() -> Path:
    """Return the ONNX model bundled with the package."""

    return Path(str(files("inspectflow") / "models" / "part_detector.onnx"))


def model_card_path(model_path: Path | None = None) -> Path:
    """Return the JSON training record stored next to an ONNX model."""

    return (model_path or default_model_path()).with_suffix(".json")


def load_model_card(model_path: Path | None = None) -> dict[str, Any]:
    """Read the training record that ``train_detector.py`` writes."""

    path = model_card_path(model_path)
    with path.open(encoding="utf-8") as handle:
        card: dict[str, Any] = json.load(handle)
    return card


@lru_cache(maxsize=4)
def _model_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def model_sha256(model_path: Path | None = None) -> str:
    """Hash the model file so reports can name the exact weights they used."""

    path = model_path or default_model_path()
    return hashlib.sha256(_model_bytes(str(path))).hexdigest()


class CnnPartDetector:
    """Detect and classify parts with the bundled ONNX network.

    ``cv2.dnn.Net`` is not safe to share between threads, so each instance owns
    its own network. Loading is fast because the file bytes are cached.

    The default threshold and border margin were chosen on validation seeds
    100-104. A margin of 5 px matches the ground-truth rule, which ignores
    parts closer than 6 px to the frame edge.
    """

    name = "cnn"

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        score_threshold: float = 0.4,
        nms_iou: float = 0.5,
        border_margin_px: int = 5,
    ) -> None:
        for label, value, low, high in (
            ("score_threshold", score_threshold, 0.0, 1.0),
            ("nms_iou", nms_iou, 0.0, 1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                or not low < float(value) <= high
            ):
                raise ValueError(f"{label} must be in ({low:g}, {high:g}]")
        if (
            isinstance(border_margin_px, bool)
            or not isinstance(border_margin_px, int)
            or border_margin_px < 0
        ):
            raise ValueError("border_margin_px must be a non-negative integer")

        self.model_path = model_path or default_model_path()
        self.score_threshold = float(score_threshold)
        self.nms_iou = float(nms_iou)
        self.border_margin_px = border_margin_px
        payload = np.frombuffer(_model_bytes(str(self.model_path)), dtype=np.uint8)
        self._net = cv2.dnn.readNetFromONNX(payload)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Return the raw ``(channels, rows, cols)`` network output."""

        _validate_image(image)
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=MODEL_INPUT_SIZE,
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        output = np.asarray(self._net.forward(), dtype=np.float32)
        expected_rows = MODEL_INPUT_SIZE[1] // OUTPUT_STRIDE
        expected_cols = MODEL_INPUT_SIZE[0] // OUTPUT_STRIDE
        if output.shape != (1, OUTPUT_CHANNELS, expected_rows, expected_cols):
            raise RuntimeError(f"unexpected model output shape {output.shape}")
        return output[0]

    def inspect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect parts in one BGR frame and return JSON-safe detections."""

        output = self.forward(image)
        height, width = image.shape[:2]
        return decode_output(
            output,
            image_size=(width, height),
            score_threshold=self.score_threshold,
            nms_iou=self.nms_iou,
            border_margin_px=self.border_margin_px,
        )


def decode_output(
    output: np.ndarray,
    *,
    image_size: tuple[int, int],
    score_threshold: float,
    nms_iou: float = 0.5,
    border_margin_px: int = 5,
) -> list[dict[str, Any]]:
    """Turn one ``(channels, rows, cols)`` output into detections.

    The class with the highest heatmap value at a peak becomes the predicted
    defect, and that value is the detection score. Taking the maximum over
    classes before peak finding stops one part from producing a detection per
    class.
    """

    if output.ndim != 3 or output.shape[0] != OUTPUT_CHANNELS:
        raise ValueError(f"output must have shape ({OUTPUT_CHANNELS}, rows, cols)")
    image_width, image_height = image_size
    scale_x = image_width / MODEL_INPUT_SIZE[0]
    scale_y = image_height / MODEL_INPUT_SIZE[1]
    class_count = len(DEFECT_LABELS)

    heatmaps = output[:class_count]
    scores = heatmaps.max(axis=0)
    classes = heatmaps.argmax(axis=0)
    neighbourhood_max = cv2.dilate(scores, np.ones((3, 3), dtype=np.uint8))
    peak_rows, peak_cols = np.nonzero(
        (scores >= score_threshold) & (scores >= neighbourhood_max)
    )

    candidates: list[dict[str, Any]] = []
    for row, col in zip(peak_rows.tolist(), peak_cols.tolist()):
        offset_x, offset_y = output[class_count : class_count + 2, row, col]
        log_width, log_height = output[class_count + 2 : class_count + 4, row, col]
        center_x = (col + float(offset_x)) * OUTPUT_STRIDE * scale_x
        center_y = (row + float(offset_y)) * OUTPUT_STRIDE * scale_y
        box_width = float(np.exp(log_width)) * SIZE_SCALE_PX * scale_x
        box_height = float(np.exp(log_height)) * SIZE_SCALE_PX * scale_y
        x = int(round(center_x - box_width / 2.0))
        y = int(round(center_y - box_height / 2.0))
        box_width_px = max(1, int(round(box_width)))
        box_height_px = max(1, int(round(box_height)))
        if (
            x <= border_margin_px
            or y <= border_margin_px
            or x + box_width_px >= image_width - border_margin_px
            or y + box_height_px >= image_height - border_margin_px
        ):
            continue
        label = DEFECT_LABELS[int(classes[row, col])]
        candidates.append(
            {
                "bbox": [x, y, box_width_px, box_height_px],
                "centroid": [round(center_x, 3), round(center_y, 3)],
                "predicted_defect": label,
                "verdict": "pass" if label == "normal" else "fail",
                "score": round(float(scores[row, col]), 4),
                "measurements": {
                    "width_px": box_width_px,
                    "height_px": box_height_px,
                },
            }
        )

    kept = _non_maximum_suppression(candidates, nms_iou)
    kept.sort(key=lambda item: (item["centroid"][0], item["centroid"][1]))
    return kept


def _non_maximum_suppression(
    candidates: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: -item["score"])
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        if all(_iou(candidate["bbox"], other["bbox"]) < iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def _iou(first: list[int], second: list[int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = width * height
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")
    if image.dtype != np.uint8:
        raise ValueError("image must have uint8 pixels")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3) in BGR order")
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError("image is too small to inspect")
