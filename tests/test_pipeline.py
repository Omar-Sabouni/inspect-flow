import numpy as np

from inspectflow import CentroidTracker, InspectionPipeline, PartInspector, SyntheticConveyor
from inspectflow.evaluation import evaluate_run


def part(x, y=40, defect="normal"):
    return {"bbox": [x - 5, y - 5, 10, 10], "centroid": [x, y], "predicted_defect": defect,
            "verdict": "pass" if defect == "normal" else "fail"}


def test_a_seed_always_produces_the_same_scene():
    first = list(SyntheticConveyor(seed=3, frame_count=12).iter_frames())
    second = list(SyntheticConveyor(seed=3, frame_count=12).iter_frames())
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a.image, b.image) and a.ground_truth == b.ground_truth


def test_poor_light_changes_the_pixels_but_not_the_parts():
    nominal = list(SyntheticConveyor(seed=3, frame_count=12).iter_frames())
    poor = list(SyntheticConveyor(seed=3, frame_count=12, scenario="poor_lighting").iter_frames())
    assert [f.ground_truth for f in nominal] == [f.ground_truth for f in poor]
    assert not np.array_equal(nominal[5].image, poor[5].image)


def test_rules_label_every_defect_type_correctly_in_normal_light():
    frames = list(SyntheticConveyor(seed=7, frame_count=96).iter_frames())
    inspector = PartInspector()
    report = evaluate_run([f.ground_truth for f in frames], [inspector.inspect(f.image) for f in frames])

    assert report["detection"]["recall"] > 0.99
    assert report["defect_accuracy"] == 1.0
    confusion = report["confusion_matrix"]["matrix"]
    assert all(confusion[i][i] > 0 for i in range(4)), "every defect type should appear"


def test_tracker_keeps_ids_through_a_missed_frame_and_reordered_detections():
    tracker = CentroidTracker(max_distance=30)
    first = tracker.update([part(20), part(90, y=80)], 0)
    tracker.update([part(27), part(97, y=80)], 1)
    tracker.update([], 2)
    after_gap = tracker.update([part(111, y=80), part(41)], 3)

    ids = {d["centroid"][0]: d["track_id"] for d in after_gap}
    assert ids[41] == first[0]["track_id"]
    assert ids[111] == first[1]["track_id"]


class Scripted:
    """Detector that replays prepared detections."""

    name = "scripted"

    def __init__(self, frames):
        self.frames = iter(frames)

    def inspect(self, image):
        return next(self.frames)


def test_each_part_is_decided_once_when_it_crosses_the_gate():
    frames = [
        [part(90), part(130, y=80)],  # second part starts past the gate
        [part(105), part(140, y=80)],
        [part(120), part(150, y=80)],
        [part(98, y=120)],  # moving right to left: never crosses
        [part(91, y=120)],
    ]
    pipeline = InspectionPipeline(Scripted(frames), gate_x=100)
    for index in range(len(frames)):
        pipeline.process(np.zeros((4, 4, 3), np.uint8), index)

    assert [(d["track_id"], d["decision_frame"]) for d in pipeline.decisions] == [(1, 1)]
    assert pipeline.observed_tracks == 3
