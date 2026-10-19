"""Centroid tracker that keeps one ID per part as it crosses the frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _Track:
    centroid: np.ndarray
    velocity: np.ndarray
    last_frame: int


class CentroidTracker:
    """Match detections to tracks by distance to each track's predicted position.

    Matching is greedy over all track/detection pairs sorted by distance, so
    the result doesn't depend on detection order. A track survives up to
    ``max_missed`` frames without a detection.
    """

    def __init__(self, max_distance: float = 70.0, max_missed: int = 4) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[dict[str, Any]], frame_index: int) -> list[dict[str, Any]]:
        """Return copies of ``detections`` with a ``track_id`` added."""

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame - 1 <= self.max_missed
        }
        centroids = [np.asarray(d["centroid"], float) for d in detections]

        pairs = []
        for track_id, track in self._tracks.items():
            predicted = track.centroid + track.velocity * (frame_index - track.last_frame)
            for index, centroid in enumerate(centroids):
                distance = float(np.linalg.norm(predicted - centroid))
                if distance <= self.max_distance:
                    pairs.append((distance, track_id, index))

        assigned: dict[int, int] = {}
        for _, track_id, index in sorted(pairs):
            if index not in assigned and track_id not in assigned.values():
                assigned[index] = track_id

        tracked = []
        for index, (detection, centroid) in enumerate(zip(detections, centroids, strict=True)):
            if index in assigned:
                track_id = assigned[index]
                track = self._tracks[track_id]
                track.velocity = (centroid - track.centroid) / (frame_index - track.last_frame)
                track.centroid, track.last_frame = centroid, frame_index
            else:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(centroid, np.zeros(2), frame_index)
            tracked.append({**detection, "track_id": track_id})
        return tracked
