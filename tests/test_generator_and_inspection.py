from __future__ import annotations

import numpy as np

from inspectflow import (
    CentroidTracker,
    PartInspector,
)
from inspectflow.evaluation import evaluate_run
from inspectflow.generator import FrameGroundTruth, SyntheticConveyor


DEFECTS = {
    "normal",
    "incorrect_dimensions",
    "wrong_color",
    "missing_feature",
}


def _iou(first: list[float], second: list[float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def test_generation_is_byte_for_byte_deterministic() -> None:
    generator = SyntheticConveyor(seed=41, frame_count=24, width=400, height=240)

    first = generator.generate()
    second = generator.generate()

    assert len(first) == len(second) == 24
    for left, right in zip(first, second):
        assert left.index == right.index
        assert left.ground_truth == right.ground_truth
        assert np.array_equal(left.image, right.image)


def test_streaming_frames_match_eager_generation_and_offer_image_free_truth() -> None:
    generator = SyntheticConveyor(seed=19, frame_count=8, width=400, height=240)

    eager = generator.generate()
    streamed = list(generator.iter_frames())

    for left, right in zip(eager, streamed):
        assert left.index == right.index
        assert left.ground_truth == right.ground_truth
        assert np.array_equal(left.image, right.image)

    metadata = streamed[0].truth_metadata()
    assert isinstance(metadata, FrameGroundTruth)
    assert not hasattr(metadata, "image")
    assert metadata.index == streamed[0].index
    assert metadata.ground_truth == streamed[0].ground_truth
    if metadata.ground_truth:
        metadata.ground_truth[0]["bbox"][0] += 1
        assert metadata.ground_truth != streamed[0].ground_truth


def test_seed_changes_the_generated_scene() -> None:
    first = SyntheticConveyor(seed=11, frame_count=12).generate()
    second = SyntheticConveyor(seed=12, frame_count=12).generate()

    assert any(
        left.ground_truth != right.ground_truth
        or not np.array_equal(left.image, right.image)
        for left, right in zip(first, second)
    )


def test_opencv_pipeline_classifies_normal_and_every_defect() -> None:
    frames = SyntheticConveyor(seed=7, frame_count=80, scenario="nominal").generate()
    inspector = PartInspector()
    correctly_classified: set[str] = set()

    for frame in frames:
        detections = inspector.inspect(frame.image)
        for truth in frame.ground_truth:
            candidates = [
                detection
                for detection in detections
                if _iou(truth["bbox"], detection["bbox"]) >= 0.5
            ]
            if not candidates:
                continue
            match = max(candidates, key=lambda item: _iou(truth["bbox"], item["bbox"]))
            if match["predicted_defect"] == truth["defect"]:
                correctly_classified.add(truth["defect"])

    assert correctly_classified == DEFECTS


def test_poor_lighting_is_a_measurably_worse_failure_example() -> None:
    def run(scenario: str) -> dict:
        frames = SyntheticConveyor(seed=7, frame_count=72, scenario=scenario).generate()
        inspector = PartInspector()
        tracker = CentroidTracker()
        detections = [
            tracker.update(inspector.inspect(frame.image), frame.index)
            for frame in frames
        ]
        return evaluate_run(frames, detections)

    nominal = run("nominal")
    poor = run("poor_lighting")

    assert poor["detection"]["recall"] < nominal["detection"]["recall"]
    assert poor["classification"]["f1"] < nominal["classification"]["f1"]
