"""The per-frame loop shared by the web app, the benchmark and the video tool."""

from __future__ import annotations

import csv
import io
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import cv2
import numpy as np

from .config import InspectionConfig
from .detectors import CnnPartDetector, Detector, DetectorName
from .inspection import PartInspector
from .tracking import CentroidTracker

GATE_FRACTION = 0.25
SHORT_LABELS = {
    "normal": "pass",
    "incorrect_dimensions": "wrong size",
    "wrong_color": "wrong color",
    "missing_feature": "no hole",
}
GATE_BGR = (44, 199, 255)
CSV_FIELDS = (
    "track_id",
    "decision_frame",
    "verdict",
    "predicted_defect",
    "score",
    "x",
    "y",
    "width",
    "height",
    "color_distance",
    "hole_radius",
    "truth_part_id",
    "truth_defect",
    "outcome",
)


def make_detector(name: DetectorName, config: InspectionConfig | None = None) -> Detector:
    if name == "cnn":
        return CnnPartDetector()
    return PartInspector(config)


class InspectionPipeline:
    """Detect, track, and decide pass or reject once per part at the gate.

    A track is decided on the first frame its center moves from left of
    ``gate_x`` to on or right of it. Tracks first seen past the gate, or lost
    before reaching it, get no decision.
    """

    def __init__(self, detector: Detector, gate_x: float) -> None:
        self.detector = detector
        self.gate_x = gate_x
        self.tracker = CentroidTracker()
        self.timings: dict[str, list[float]] = defaultdict(list)
        self._decisions: dict[int, dict[str, Any]] = {}
        self._last_x: dict[int, float] = {}

    @property
    def decisions(self) -> list[dict[str, Any]]:
        return [self._decisions[track_id] for track_id in sorted(self._decisions)]

    @property
    def observed_tracks(self) -> int:
        return len(self._last_x)

    @contextmanager
    def timed(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        yield
        self.timings[stage].append((time.perf_counter() - start) * 1000)

    def process(self, image: np.ndarray, frame_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return this frame's tracked detections and the decisions made on it."""

        with self.timed("detect"):
            detections = self.detector.inspect(image)
        with self.timed("track"):
            tracked = self.tracker.update(detections, frame_index)

        decided_now = []
        for detection in tracked:
            track_id, x = detection["track_id"], detection["centroid"][0]
            previous = self._last_x.get(track_id)
            self._last_x[track_id] = x
            if track_id not in self._decisions and previous is not None and previous < self.gate_x <= x:
                decision = {**detection, "decision_frame": frame_index}
                self._decisions[track_id] = decision
                decided_now.append(decision)
        return tracked, decided_now

    def timing_summary(self) -> dict[str, dict[str, float]]:
        summary = {}
        for stage, samples in self.timings.items():
            values = np.asarray(samples)
            summary[stage] = {
                "mean_ms": round(float(values.mean()), 3),
                "p50_ms": round(float(np.percentile(values, 50)), 3),
                "p95_ms": round(float(np.percentile(values, 95)), 3),
                "max_ms": round(float(values.max()), 3),
            }
        return summary


def decision_csv(decisions: list[dict[str, Any]], outcomes: list[dict[str, Any]] | None = None) -> str:
    """One CSV row per decision, with the matched true part if ``outcomes`` is given."""

    output = io.StringIO()
    writer = csv.DictWriter(output, CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for index, decision in enumerate(decisions):
        x, y, width, height = decision["bbox"]
        measurements = decision.get("measurements", {})
        writer.writerow(
            {
                **decision,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "color_distance": measurements.get("color_distance", ""),
                "hole_radius": measurements.get("feature_radius_px", ""),
                **(outcomes[index] if outcomes else {}),
            }
        )
    return output.getvalue()


def annotate_frame(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    frame_index: int,
    label: str,
    gate_x: int,
    captions: bool = True,
) -> np.ndarray:
    """Draw the gate line, boxes and short verdict labels on a copy of ``image``.

    ``captions`` adds the word GATE and a frame counter; the dashboard shows
    both itself and turns them off.
    """

    canvas = image.copy()
    height = canvas.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(canvas, (gate_x, min(53, height // 6)), (gate_x, height - 43), GATE_BGR, 1)
    if captions:
        cv2.putText(canvas, "GATE", (gate_x + 5, height - 46), font, 0.4, GATE_BGR, 1, cv2.LINE_AA)
        counter = f"{frame_index + 1:03d}  {label}"
        cv2.putText(canvas, counter, (12, height - 12), font, 0.42, (211, 221, 231), 1, cv2.LINE_AA)

    placed: list[tuple[int, int, int, int]] = []
    for detection in detections:
        x, y, w, h = (int(v) for v in detection["bbox"])
        color = (72, 92, 255) if detection["verdict"] == "fail" else (87, 214, 143)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        text = f"#{detection['track_id']} {SHORT_LABELS[detection['predicted_defect']]}"
        (text_w, text_h), baseline = cv2.getTextSize(text, font, 0.43, 1)
        # Label above the box, or below it if that would cover a neighbour's label.
        for label_y in (max(20, y - 5), y + h + text_h + baseline + 6):
            box = (x, label_y - text_h - baseline - 4, x + text_w + 8, label_y + baseline)
            if not any(_overlaps(box, other) for other in placed):
                break
        placed.append(box)
        cv2.rectangle(canvas, box[:2], box[2:], (19, 24, 31), -1)
        cv2.putText(canvas, text, (x + 4, label_y - 3), font, 0.43, color, 1, cv2.LINE_AA)
    return canvas


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]
