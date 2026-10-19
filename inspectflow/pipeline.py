"""Per-frame inspection loop shared by the web app, benchmark and video CLI."""

from __future__ import annotations

import csv
import io
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import cv2
import numpy as np

from .config import InspectionConfig
from .detectors import CnnPartDetector, Detector, DetectorName
from .inspection import PartInspector
from .tracking import CentroidTracker


INSPECTION_GATE_FRACTION = 0.25
DEFECT_REJECT_LABELS = ("incorrect_dimensions", "wrong_color", "missing_feature")
DECISION_LOG_FIELDS = (
    "track_id",
    "decision_frame",
    "representative_frame",
    "verdict",
    "predicted_defect",
    "score",
    "centroid_x_px",
    "centroid_y_px",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "measured_width_px",
    "measured_height_px",
    "color_distance",
    "feature_radius_px",
)


def make_detector(
    name: DetectorName, config: InspectionConfig | None = None
) -> Detector:
    """Build the classical rule pipeline or the CNN detector."""

    resolved = InspectionConfig() if config is None else config
    if name == "classical":
        return PartInspector(resolved)
    if name == "cnn":
        return CnnPartDetector()
    raise ValueError(f"unknown detector: {name}")


class StageTimer:
    """Collect wall-clock durations per pipeline stage in milliseconds."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter_ns() - started) / 1e6)

    def add(self, stage: str, milliseconds: float) -> None:
        self._samples.setdefault(stage, []).append(float(milliseconds))

    def summary(self) -> dict[str, dict[str, float]]:
        """Return mean, median, 95th percentile and maximum for every stage."""

        result: dict[str, dict[str, float]] = {}
        for stage, samples in self._samples.items():
            values = np.asarray(samples, dtype=np.float64)
            result[stage] = {
                "mean_ms": round(float(values.mean()), 3),
                "p50_ms": round(float(np.percentile(values, 50)), 3),
                "p95_ms": round(float(np.percentile(values, 95)), 3),
                "max_ms": round(float(values.max()), 3),
            }
        return result


class InspectionPipeline:
    """Detect, track and make one gate decision per part crossing.

    A track gets its decision on the first frame where its centroid moves from
    left of ``gate_x`` to on or right of it, using the detection from that
    frame. A track that is first seen beyond the gate, or that disappears
    before reaching it, stays undecided.
    """

    def __init__(
        self,
        detector: Detector,
        *,
        gate_x: float,
        tracker: CentroidTracker | None = None,
    ) -> None:
        self.detector = detector
        self.gate_x = float(gate_x)
        self.tracker = tracker or CentroidTracker(max_distance=70.0, max_missed=4)
        self.timer = StageTimer()
        self._last_x: dict[int, float] = {}
        self._decisions: list[dict[str, Any]] = []
        self._decided: set[int] = set()

    @property
    def decisions(self) -> list[dict[str, Any]]:
        """Gate decisions ordered by track ID."""

        return sorted(
            (dict(item) for item in self._decisions),
            key=lambda item: int(item["track_id"]),
        )

    @property
    def observed_tracks(self) -> int:
        return len(self._last_x)

    def process(
        self, image: np.ndarray, frame_index: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run one frame and return its tracked detections and new decisions."""

        with self.timer.measure("detect"):
            detections = self.detector.inspect(image)
        with self.timer.measure("track"):
            tracked = self.tracker.update(detections, frame_index)

        new_decisions: list[dict[str, Any]] = []
        for detection in tracked:
            track_id = int(detection["track_id"])
            current_x = float(detection["centroid"][0])
            previous_x = self._last_x.get(track_id)
            self._last_x[track_id] = current_x
            if (
                track_id in self._decided
                or previous_x is None
                or not previous_x < self.gate_x <= current_x
            ):
                continue
            decision = {
                "frame_index": frame_index,
                **detection,
                "track_id": track_id,
                "decision_frame": frame_index,
            }
            self._decided.add(track_id)
            self._decisions.append(decision)
            new_decisions.append(dict(decision))
        return tracked, new_decisions


def decision_counts(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Accepted, rejected and per-defect totals for a set of decisions."""

    defect_counts = {label: 0 for label in DEFECT_REJECT_LABELS}
    for item in decisions:
        label = str(item["predicted_defect"])
        if label in defect_counts:
            defect_counts[label] += 1
    return {
        "accepted": sum(item["verdict"] == "pass" for item in decisions),
        "rejected": sum(item["verdict"] == "fail" for item in decisions),
        "defect_counts": defect_counts,
    }


def decision_timeline(
    decisions: Sequence[Mapping[str, Any]], frame_count: int
) -> list[dict[str, Any]]:
    """Cumulative decision totals as of each frame."""

    ordered = sorted(decisions, key=lambda item: int(item["decision_frame"]))
    timeline: list[dict[str, Any]] = []
    position = 0
    for frame_index in range(frame_count):
        while (
            position < len(ordered)
            and int(ordered[position]["decision_frame"]) <= frame_index
        ):
            position += 1
        timeline.append(decision_counts(ordered[:position]))
    return timeline


def decision_rows(
    decisions: Sequence[Mapping[str, Any]],
    extra: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten decisions into CSV rows, merging optional per-row extras."""

    if extra is not None and len(extra) != len(decisions):
        raise ValueError("extra must have one entry per decision")
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(decisions):
        measurements = item.get("measurements", {})
        centroid = item["centroid"]
        bbox = item["bbox"]
        row = {
            "track_id": int(item["track_id"]),
            "decision_frame": int(item["decision_frame"]),
            "representative_frame": int(item["frame_index"]),
            "verdict": item["verdict"],
            "predicted_defect": str(item["predicted_defect"]),
            "score": item.get("score", ""),
            "centroid_x_px": round(float(centroid[0]), 3),
            "centroid_y_px": round(float(centroid[1]), 3),
            "bbox_x": int(bbox[0]),
            "bbox_y": int(bbox[1]),
            "bbox_width": int(bbox[2]),
            "bbox_height": int(bbox[3]),
            "measured_width_px": measurements.get("width_px", ""),
            "measured_height_px": measurements.get("height_px", ""),
            "color_distance": measurements.get("color_distance", ""),
            "feature_radius_px": measurements.get("feature_radius_px", ""),
        }
        if extra is not None:
            row.update(extra[position])
        rows.append(row)
    return rows


def rows_to_csv(
    rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] = DECISION_LOG_FIELDS
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fieldnames), restval="")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def annotate_frame(
    image: np.ndarray,
    detections: Sequence[Mapping[str, Any]],
    *,
    frame_index: int,
    label: str,
    gate_x: int,
) -> np.ndarray:
    """Draw the gate, boxes, track IDs and verdicts on a copy of ``image``."""

    canvas = image.copy()
    height, width = canvas.shape[:2]
    placed: list[tuple[int, int, int, int]] = []
    top = min(53, height // 6)
    cv2.line(canvas, (gate_x, top), (gate_x, height - 43), (130, 187, 255), 1)
    cv2.putText(
        canvas,
        "INSPECTION GATE",
        (max(0, gate_x - 66), max(12, top - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (185, 216, 255),
        1,
        cv2.LINE_AA,
    )
    for detection in detections:
        x, y, box_width, box_height = (int(value) for value in detection["bbox"])
        failed = detection["verdict"] == "fail"
        color = (72, 92, 255) if failed else (87, 214, 143)
        cv2.rectangle(canvas, (x, y), (x + box_width, y + box_height), color, 2)
        defect = str(detection["predicted_defect"]).replace("_", " ")
        verdict = "FAIL " + defect if failed else "PASS"
        text = f"ID {detection['track_id']}  {verdict}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1
        )
        # Put the label above the box, or below it if that would cover a label
        # already drawn for a neighbouring part.
        label_y = max(20, y - 5)
        for candidate in (label_y, y + box_height + text_height + baseline + 6):
            label_y = candidate
            box = (
                x,
                label_y - text_height - baseline - 4,
                min(width - 1, x + text_width + 8),
                label_y + baseline,
            )
            if not any(_overlaps(box, other) for other in placed):
                break
        placed.append(box)
        cv2.rectangle(canvas, box[:2], box[2:], (19, 24, 31), -1)
        cv2.putText(
            canvas,
            text,
            (x + 4, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"FRAME {frame_index + 1:03d}  {label.upper().replace('_', ' ')}",
        (18, height - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (211, 221, 231),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _overlaps(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    return not (
        first[2] < second[0]
        or second[2] < first[0]
        or first[3] < second[1]
        or second[3] < first[1]
    )
