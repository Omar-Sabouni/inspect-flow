"""Score detections and gate decisions against the generator's ground truth.

``truth`` is always a list with one entry per frame, in frame order, each a
list of labelled parts as ``SyntheticConveyor`` produces them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .generator import DEFECTS

Parts = list[dict[str, Any]]


def box_iou(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    overlap = width * height
    union = a[2] * a[3] + b[2] * b[3] - overlap
    return overlap / union if union > 0 else 0.0


def match(truth: Parts, detections: Parts, iou_threshold: float = 0.3) -> list[tuple[int, int]]:
    """Pair parts with detections greedily, highest IoU first."""

    pairs = sorted(
        ((box_iou(t["bbox"], d["bbox"]), i, j) for i, t in enumerate(truth) for j, d in enumerate(detections)),
        reverse=True,
    )
    matched: list[tuple[int, int]] = []
    for overlap, i, j in pairs:
        if overlap < iou_threshold:
            break
        if all(i != a and j != b for a, b in matched):
            matched.append((i, j))
    return matched


def evaluate_run(truth: list[Parts], detections: list[Parts], iou_threshold: float = 0.3) -> dict[str, Any]:
    """Per-frame detection counts and a confusion matrix over matched parts."""

    hits = misses = false_alarms = 0
    confusion = np.zeros((len(DEFECTS), len(DEFECTS)), int)
    for frame_truth, frame_detections in zip(truth, detections, strict=True):
        pairs = match(frame_truth, frame_detections, iou_threshold)
        hits += len(pairs)
        misses += len(frame_truth) - len(pairs)
        false_alarms += len(frame_detections) - len(pairs)
        for i, j in pairs:
            actual = DEFECTS.index(frame_truth[i]["defect"])
            predicted = DEFECTS.index(frame_detections[j]["predicted_defect"])
            confusion[actual, predicted] += 1
    return {
        "detection": {
            "true_positives": hits,
            "false_positives": false_alarms,
            "false_negatives": misses,
            "precision": _ratio(hits, hits + false_alarms),
            "recall": _ratio(hits, hits + misses),
        },
        "defect_accuracy": _ratio(int(np.trace(confusion)), int(confusion.sum())),
        "confusion_matrix": {"labels": list(DEFECTS), "matrix": confusion.tolist()},
    }


def evaluate_gate(
    decisions: Parts, truth: list[Parts], gate_x: float, iou_threshold: float = 0.3
) -> dict[str, Any]:
    """Match each gate decision to the part under it and count the outcomes.

    Outcomes: correct pass or reject; wrong defect (rejected for the wrong
    reason); escape (defective part passed); false reject (good part
    rejected); spurious (nothing matched, or the part was already decided).
    A part whose true center crosses the gate with no decision is missed.
    """

    crossed: dict[str, dict[str, Any]] = {}
    last_x: dict[str, float] = {}
    for parts in truth:
        for part in parts:
            x = part["centroid"][0]
            if last_x.get(part["part_id"], gate_x) < gate_x <= x:
                crossed.setdefault(part["part_id"], part)
            last_x[part["part_id"]] = x

    counts = dict.fromkeys(
        ("correct_pass", "correct_reject", "wrong_defect", "escape", "false_reject", "spurious"), 0
    )
    outcomes = []
    decided: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        candidates = truth[decision["decision_frame"]]
        overlaps = [box_iou(part["bbox"], decision["bbox"]) for part in candidates]
        best = int(np.argmax(overlaps)) if overlaps else -1
        target = candidates[best] if best >= 0 and overlaps[best] >= iou_threshold else None

        if target is None or target["part_id"] in decided:
            outcome = "spurious"
        else:
            decided[target["part_id"]] = target
            actual, predicted = target["defect"], decision["predicted_defect"]
            if actual == "normal":
                outcome = "correct_pass" if predicted == "normal" else "false_reject"
            elif predicted == "normal":
                outcome = "escape"
            else:
                outcome = "correct_reject" if predicted == actual else "wrong_defect"
        counts[outcome] += 1
        outcomes.append(
            {
                "truth_part_id": target["part_id"] if target else "",
                "truth_defect": target["defect"] if target else "",
                "outcome": outcome,
            }
        )

    at_gate = {**crossed, **decided}
    good = sum(part["defect"] == "normal" for part in at_gate.values())
    missed = sum(part_id not in decided for part_id in crossed)
    right = counts["correct_pass"] + counts["correct_reject"] + counts["wrong_defect"]
    return {
        "parts": len(at_gate),
        "good_parts": good,
        "defective_parts": len(at_gate) - good,
        "counts": {**counts, "missed": missed},
        "verdict_accuracy": _ratio(right, len(at_gate)),
        "escape_rate": _ratio(counts["escape"], len(at_gate) - good),
        "false_reject_rate": _ratio(counts["false_reject"], good),
        "miss_rate": _ratio(missed, len(at_gate)),
        "outcomes": outcomes,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
