import math

import numpy as np

from inspectflow.detectors import INPUT_SIZE, OUTPUT_CHANNELS, STRIDE, decode, model_record, model_sha256
from inspectflow.evaluation import evaluate_gate, evaluate_run
from inspectflow.generator import SyntheticConveyor
from inspectflow.pipeline import InspectionPipeline, make_detector


def test_decode_turns_a_heatmap_peak_into_a_box():
    output = np.zeros((OUTPUT_CHANNELS, INPUT_SIZE[1] // STRIDE, INPUT_SIZE[0] // STRIDE), np.float32)
    output[2, 15, 30] = 0.9  # wrong_color
    output[4:6, 15, 30] = 0.5  # centre of the cell
    output[6:8, 15, 30] = math.log(80 / 64), math.log(50 / 64)
    output[0, 15, 2] = 0.9  # a part touching the left edge is dropped

    (found,) = decode(output, INPUT_SIZE)

    assert found["bbox"] == [204, 99, 80, 50]
    assert found["predicted_defect"] == "wrong_color"


def test_bundled_model_matches_its_training_record():
    assert model_record()["sha256"] == model_sha256()


def test_cnn_gets_every_part_right_in_poor_light():
    conveyor = SyntheticConveyor(seed=7, frame_count=96, scenario="poor_lighting")
    pipeline = InspectionPipeline(make_detector("cnn"), gate_x=160)
    truth, detections = [], []
    for frame in conveyor.iter_frames():
        detections.append(pipeline.process(frame.image, frame.index)[0])
        truth.append(frame.ground_truth)

    assert evaluate_run(truth, detections)["detection"]["recall"] > 0.99
    gate = evaluate_gate(pipeline.decisions, truth, gate_x=160)
    assert gate["parts"] == 5
    assert gate["verdict_accuracy"] == 1.0
