from inspectflow.evaluation import evaluate_gate, evaluate_run


def truth(part_id, x, defect, y=0):
    return {"part_id": part_id, "bbox": [x, y, 20, 10], "centroid": [x + 10, y + 5], "defect": defect}


def detection(x, defect, y=0, frame=1):
    return {"bbox": [x, y, 20, 10], "predicted_defect": defect, "decision_frame": frame,
            "verdict": "pass" if defect == "normal" else "fail"}


def test_frame_metrics_count_misses_false_alarms_and_wrong_labels():
    parts = [[truth("a", 0, "normal"), truth("b", 50, "wrong_color"), truth("c", 100, "missing_feature")]]
    found = [[detection(1, "normal"), detection(51, "missing_feature"), detection(300, "normal")]]

    report = evaluate_run(parts, found)

    assert report["detection"] == {
        "true_positives": 2, "false_positives": 1, "false_negatives": 1,
        "precision": 0.666667, "recall": 0.666667,
    }
    assert report["defect_accuracy"] == 0.5


def test_gate_outcomes():
    # Five parts on separate rows; every centre crosses x=100 on frame 1.
    defects = ["normal", "normal", "wrong_color", "missing_feature", "wrong_color"]
    frames = [[truth(f"p{row}", 85 + 10 * index, d, y=20 * row) for row, d in enumerate(defects)] for index in range(2)]
    decisions = [
        detection(95, "normal", y=0),  # good part passed
        detection(95, "missing_feature", y=20),  # good part rejected
        detection(95, "normal", y=40),  # bad part passed
        detection(95, "wrong_color", y=60),  # rejected, wrong reason
        detection(95, "wrong_color", y=60),  # same part again
    ]  # p4 never gets a decision

    report = evaluate_gate(decisions, frames, gate_x=100)

    assert [o["outcome"] for o in report["outcomes"]] == [
        "correct_pass", "false_reject", "escape", "wrong_defect", "spurious",
    ]
    assert report["counts"]["missed"] == 1
    assert (report["parts"], report["good_parts"], report["defective_parts"]) == (5, 2, 3)
    assert report["verdict_accuracy"] == 0.4  # correct pass + wrong-defect reject
