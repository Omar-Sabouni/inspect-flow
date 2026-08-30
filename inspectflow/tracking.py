"""Small deterministic centroid tracker with persistent integer IDs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(slots=True)
class _Track:
    track_id: int
    centroid: tuple[float, float]
    last_frame: int
    velocity: tuple[float, float] = (0.0, 0.0)
    missed: int = 0


class CentroidTracker:
    """Associate detections with tracks using predicted-centroid distance.

    Matching is globally greedy over the sorted distance matrix, which avoids
    detection-order-dependent assignments. Short occlusions are bridged for up
    to ``max_missed`` frames.
    """

    def __init__(self, max_distance: float = 70.0, max_missed: int = 3) -> None:
        if (
            isinstance(max_distance, bool)
            or not isinstance(max_distance, Real)
            or not isfinite(float(max_distance))
            or max_distance <= 0
        ):
            raise ValueError("max_distance must be positive")
        if (
            isinstance(max_missed, bool)
            or not isinstance(max_missed, int)
            or max_missed < 0
        ):
            raise ValueError("max_missed must be a non-negative integer")
        self.max_distance: float = float(max_distance)
        self.max_missed: int = max_missed
        self._tracks: dict[int, _Track] = {}
        self._next_id: int = 1
        self._last_frame_index: int | None = None

    @property
    def active_tracks(self) -> dict[int, dict[str, Any]]:
        """Return a serializable snapshot without exposing mutable state."""

        return {
            track_id: {
                "track_id": track_id,
                "centroid": list(track.centroid),
                "velocity": list(track.velocity),
                "last_frame": track.last_frame,
                "missed": track.missed,
            }
            for track_id, track in sorted(self._tracks.items())
        }

    def reset(self) -> None:
        """Remove all active tracks and restart IDs at one."""

        self._tracks.clear()
        self._next_id = 1
        self._last_frame_index = None

    def update(
        self, detections: Iterable[Mapping[str, Any]], frame_index: int
    ) -> list[dict[str, Any]]:
        """Assign ``track_id`` to copied detections for ``frame_index``."""

        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase monotonically")

        if detections is None or isinstance(detections, (str, bytes, Mapping)):
            raise ValueError("detections must be an iterable of detection objects")
        try:
            copied = [self._normalize_detection(detection) for detection in detections]
        except TypeError as exc:
            raise ValueError(
                "detections must be an iterable of detection objects"
            ) from exc
        # Frames skipped between calls still count as missed observations. Drop
        # tracks that already exceeded the allowance before they can associate
        # with a detection in the current frame.
        self._expire_before(frame_index)
        assignments = self._associate(copied, frame_index)
        matched_tracks = set(assignments.values())

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_tracks:
                track.missed = frame_index - track.last_frame
                if track.missed > self.max_missed:
                    del self._tracks[track_id]

        for detection_index, detection in enumerate(copied):
            if detection_index in assignments:
                track_id = assignments[detection_index]
                track = self._tracks[track_id]
                current = (
                    float(detection["centroid"][0]),
                    float(detection["centroid"][1]),
                )
                elapsed = max(1, frame_index - track.last_frame)
                track.velocity = (
                    (current[0] - track.centroid[0]) / elapsed,
                    (current[1] - track.centroid[1]) / elapsed,
                )
                track.centroid = current
                track.last_frame = frame_index
                track.missed = 0
            else:
                track_id = self._next_id
                self._next_id += 1
                current = (
                    float(detection["centroid"][0]),
                    float(detection["centroid"][1]),
                )
                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    centroid=current,
                    last_frame=frame_index,
                )
            detection["track_id"] = track_id

        self._last_frame_index = frame_index
        return copied

    def _expire_before(self, frame_index: int) -> None:
        """Account for unreported frames before current-frame association."""

        for track_id, track in list(self._tracks.items()):
            track.missed = max(0, frame_index - track.last_frame - 1)
            if track.missed > self.max_missed:
                del self._tracks[track_id]

    def _associate(
        self, detections: list[dict[str, Any]], frame_index: int
    ) -> dict[int, int]:
        if not detections or not self._tracks:
            return {}

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            elapsed = max(1, frame_index - track.last_frame)
            vx, vy = track.velocity
            predicted = np.asarray(
                [track.centroid[0] + vx * elapsed, track.centroid[1] + vy * elapsed]
            )
            for detection_index, detection in enumerate(detections):
                centroid = np.asarray(detection["centroid"], dtype=float)
                distance = float(np.linalg.norm(predicted - centroid))
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, detection_index))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        assignments: dict[int, int] = {}
        for _, track_id, detection_index in candidates:
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)
            assignments[detection_index] = track_id
        return assignments

    @staticmethod
    def _normalize_detection(detection: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(detection, Mapping):
            raise ValueError("each detection must be an object")
        if "centroid" not in detection:
            raise ValueError("each detection requires a centroid")
        centroid = detection["centroid"]
        if (
            not isinstance(centroid, (list, tuple))
            or len(centroid) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                for value in centroid
            )
        ):
            raise ValueError("centroid must contain two numeric values")
        copied = dict(detection)
        copied["centroid"] = [float(centroid[0]), float(centroid[1])]
        return copied
