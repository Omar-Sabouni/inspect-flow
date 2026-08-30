from __future__ import annotations

from inspectflow import CentroidTracker


def _detection(x: float, y: float) -> dict:
    return {
        "bbox": [x - 5, y - 5, 10, 10],
        "centroid": [x, y],
        "predicted_defect": "normal",
        "verdict": "pass",
    }


def test_track_id_continues_through_motion_and_one_missed_frame() -> None:
    tracker = CentroidTracker(max_distance=30, max_missed=2)

    first = tracker.update([_detection(20, 40)], frame_index=0)
    second = tracker.update([_detection(27, 40)], frame_index=1)
    tracker.update([], frame_index=2)
    fourth = tracker.update([_detection(41, 40)], frame_index=3)

    assert first[0]["track_id"] == 1
    assert second[0]["track_id"] == first[0]["track_id"]
    assert fourth[0]["track_id"] == first[0]["track_id"]


def test_two_tracks_keep_ids_when_detection_order_changes() -> None:
    tracker = CentroidTracker(max_distance=25, max_missed=0)
    first = tracker.update([_detection(20, 20), _detection(90, 60)], frame_index=0)
    second = tracker.update([_detection(86, 60), _detection(25, 20)], frame_index=1)

    first_ids = {tuple(item["centroid"]): item["track_id"] for item in first}
    second_ids = {tuple(item["centroid"]): item["track_id"] for item in second}
    assert second_ids[(25.0, 20.0)] == first_ids[(20.0, 20.0)]
    assert second_ids[(86.0, 60.0)] == first_ids[(90.0, 60.0)]


def test_track_expires_before_association_after_too_many_skipped_frames() -> None:
    tracker = CentroidTracker(max_distance=100, max_missed=1)

    first = tracker.update([_detection(20, 40)], frame_index=0)
    after_gap = tracker.update([_detection(20, 40)], frame_index=3)

    assert first[0]["track_id"] == 1
    assert after_gap[0]["track_id"] == 2


def test_velocity_is_normalized_by_elapsed_frames_after_a_gap() -> None:
    tracker = CentroidTracker(max_distance=10, max_missed=2)

    first = tracker.update([_detection(0, 40)], frame_index=0)
    tracker.update([_detection(8, 40)], frame_index=1)
    after_gap = tracker.update([_detection(32, 40)], frame_index=4)
    next_frame = tracker.update([_detection(40, 40)], frame_index=5)

    assert after_gap[0]["track_id"] == first[0]["track_id"]
    assert next_frame[0]["track_id"] == first[0]["track_id"]
