"""Ground-truth matching and detection, classification, and tracking metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from .generator import FrameGroundTruth, GeneratedFrame


DEFECT_LABELS = (
    "normal",
    "incorrect_dimensions",
    "wrong_color",
    "missing_feature",
)

EvaluationFrame = GeneratedFrame | FrameGroundTruth


def evaluate_run(
    generated_frames: Sequence[EvaluationFrame],
    detections_by_frame: (
        Sequence[Sequence[Mapping[str, Any]]]
        | Mapping[int, Sequence[Mapping[str, Any]]]
    ),
    *,
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Evaluate a run against generated truth and return JSON-safe metrics.

    ``precision``, ``recall`` and ``f1`` in the top-level ``classification``
    section describe binary defective-vs-normal decisions over matched objects.
    Unmatched ground-truth objects count as false negatives for detection, but do
    not invent a defect-class prediction in the multiclass confusion matrix.
    Image-free :class:`FrameGroundTruth` values let streaming callers evaluate a
    run without retaining its raw frame arrays.
    """

    threshold = _validate_iou_threshold(iou_threshold)
    frames, frame_indices = _validate_frames(generated_frames)
    indexed_detections = _normalize_detections(detections_by_frame, frame_indices)
    _validate_objects(frames, indexed_detections)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    binary = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    matrix = {
        actual: {predicted: 0 for predicted in DEFECT_LABELS}
        for actual in DEFECT_LABELS
    }
    ground_truth_counts = {label: 0 for label in DEFECT_LABELS}
    matched_ground_truth_counts = {label: 0 for label in DEFECT_LABELS}
    matched_pairs = 0
    ground_truth_objects = 0
    predicted_objects = 0
    assignments_by_part: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for position, generated in enumerate(frames):
        truth = generated.ground_truth
        detections = indexed_detections[position]
        ground_truth_objects += len(truth)
        predicted_objects += len(detections)

        for item in truth:
            ground_truth_counts[_truth_label(item)] += 1

        matches = _match_frame(truth, detections, threshold)
        true_positives += len(matches)
        false_negatives += len(truth) - len(matches)
        false_positives += len(detections) - len(matches)
        matched_pairs += len(matches)

        for truth_index, detection_index, _ in matches:
            actual = _truth_label(truth[truth_index])
            predicted = _prediction_label(detections[detection_index])
            matrix[actual][predicted] += 1
            matched_ground_truth_counts[actual] += 1

            actual_defective = actual != "normal"
            predicted_defective = predicted != "normal"
            if actual_defective and predicted_defective:
                binary["tp"] += 1
            elif not actual_defective and predicted_defective:
                binary["fp"] += 1
            elif actual_defective and not predicted_defective:
                binary["fn"] += 1
            else:
                binary["tn"] += 1

            part_id = str(
                truth[truth_index].get(
                    "part_id", f"frame-{position}-gt-{truth_index}"
                )
            )
            track_id = detections[detection_index].get("track_id")
            if isinstance(track_id, Integral) and not isinstance(track_id, bool):
                assignments_by_part[part_id].append(
                    (generated.index, int(track_id))
                )

    detection_precision = _safe_ratio(
        true_positives, true_positives + false_positives
    )
    detection_recall = _safe_ratio(
        true_positives, true_positives + false_negatives
    )
    binary_precision = _safe_ratio(binary["tp"], binary["tp"] + binary["fp"])
    binary_recall = _safe_ratio(binary["tp"], binary["tp"] + binary["fn"])
    binary_f1 = _safe_ratio(
        2 * binary_precision * binary_recall,
        binary_precision + binary_recall,
    )
    accuracy = _safe_ratio(binary["tp"] + binary["tn"], sum(binary.values()))

    id_switches = 0
    tracked_transitions = 0
    tracked_parts = 0
    for observations in assignments_by_part.values():
        observations.sort()
        if observations:
            tracked_parts += 1
        for previous, current in zip(observations, observations[1:]):
            tracked_transitions += 1
            if previous[1] != current[1]:
                id_switches += 1
    continuity = (
        None
        if tracked_transitions == 0
        else _rounded(1.0 - id_switches / tracked_transitions)
    )

    return {
        "detection": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": _rounded(detection_precision),
            "recall": _rounded(detection_recall),
            "f1": _rounded(
                _safe_ratio(
                    2 * detection_precision * detection_recall,
                    detection_precision + detection_recall,
                )
            ),
        },
        "classification": {
            "task": "binary_defect",
            "positive_class": "defective",
            "true_positives": binary["tp"],
            "false_positives": binary["fp"],
            "false_negatives": binary["fn"],
            "true_negatives": binary["tn"],
            "precision": _rounded(binary_precision),
            "recall": _rounded(binary_recall),
            "f1": _rounded(binary_f1),
            "accuracy": _rounded(accuracy),
            "confusion_matrix": {
                "labels": ["normal", "defective"],
                "matrix": [
                    [binary["tn"], binary["fp"]],
                    [binary["fn"], binary["tp"]],
                ],
            },
        },
        "defect_classification": {
            "labels": list(DEFECT_LABELS),
            "confusion_matrix": [
                [matrix[actual][predicted] for predicted in DEFECT_LABELS]
                for actual in DEFECT_LABELS
            ],
            "ground_truth_distribution": ground_truth_counts,
            "matched_ground_truth_distribution": matched_ground_truth_counts,
            "matched_objects": matched_pairs,
        },
        "tracking": {
            "id_switches": id_switches,
            "tracked_transitions": tracked_transitions,
            "tracked_parts": tracked_parts,
            "continuity": continuity,
        },
        "totals": {
            "frames": len(frames),
            "ground_truth_objects": ground_truth_objects,
            "predicted_objects": predicted_objects,
            "matched_objects": matched_pairs,
        },
    }


def _validate_iou_threshold(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or not 0.0 < float(value) <= 1.0
    ):
        raise ValueError("iou_threshold must be in (0, 1]")
    return float(value)


def _validate_frames(
    values: Sequence[EvaluationFrame],
) -> tuple[list[EvaluationFrame], list[int]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("generated_frames must be a sequence")
    frames = list(values)
    if any(
        not isinstance(frame, (GeneratedFrame, FrameGroundTruth))
        for frame in frames
    ):
        raise ValueError(
            "generated_frames must contain GeneratedFrame or FrameGroundTruth objects"
        )

    frame_indices: list[int] = []
    for frame in frames:
        if (
            isinstance(frame.index, bool)
            or not isinstance(frame.index, Integral)
            or frame.index < 0
        ):
            raise ValueError("generated frame indices must be non-negative integers")
        if (
            not isinstance(frame.ground_truth, Sequence)
            or isinstance(frame.ground_truth, (str, bytes))
        ):
            raise ValueError("every frame's ground_truth must be a sequence")
        frame_indices.append(int(frame.index))
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError("generated frame indices must be unique")
    return frames, frame_indices


def _normalize_detections(
    values: (
        Sequence[Sequence[Mapping[str, Any]]]
        | Mapping[int, Sequence[Mapping[str, Any]]]
    ),
    frame_indices: Sequence[int],
) -> list[list[Mapping[str, Any]]]:
    if isinstance(values, Mapping):
        unexpected = set(values).difference(frame_indices)
        if unexpected:
            names = ", ".join(str(key) for key in sorted(unexpected, key=str))
            raise ValueError(f"detections contain unknown frame index(es): {names}")
        raw_frames = [values.get(index, []) for index in frame_indices]
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != len(frame_indices):
            raise ValueError(
                "detections_by_frame must have one entry per generated frame"
            )
        raw_frames = list(values)
    else:
        raise ValueError(
            "detections_by_frame must be a sequence or frame-index mapping"
        )

    normalized: list[list[Mapping[str, Any]]] = []
    for frame in raw_frames:
        if isinstance(frame, (str, bytes, Mapping)):
            raise ValueError("every frame's detections must be a sequence")
        try:
            detections = list(frame)
        except TypeError as exc:
            raise ValueError("every frame's detections must be a sequence") from exc
        if any(not isinstance(detection, Mapping) for detection in detections):
            raise ValueError("each detection must be an object")
        normalized.append(detections)
    return normalized


def _validate_objects(
    frames: Sequence[EvaluationFrame],
    detections_by_frame: Sequence[Sequence[Mapping[str, Any]]],
) -> None:
    """Validate all objects before metric accumulation begins."""

    for frame in frames:
        for item in frame.ground_truth:
            if not isinstance(item, Mapping):
                raise ValueError("each ground-truth object must be an object")
            _bbox(item)
            _truth_label(item)

    for detections in detections_by_frame:
        for detection in detections:
            _bbox(detection)
            _prediction_label(detection)
            track_id = detection.get("track_id")
            if track_id is not None and (
                isinstance(track_id, bool) or not isinstance(track_id, Integral)
            ):
                raise ValueError("track_id must be an integer when provided")


@dataclass(slots=True)
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: float


def _add_flow_edge(
    graph: list[list[_FlowEdge]], source: int, target: int, cost: float
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), 1, cost)
    backward = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def _match_frame(
    truth: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Return a max-cardinality, then maximum-total-IoU assignment."""

    if not truth or not detections:
        return []

    truth_count = len(truth)
    detection_count = len(detections)
    source = 0
    truth_offset = 1
    detection_offset = truth_offset + truth_count
    sink = detection_offset + detection_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    for truth_index in range(truth_count):
        _add_flow_edge(graph, source, truth_offset + truth_index, 0.0)
    for detection_index in range(detection_count):
        _add_flow_edge(graph, detection_offset + detection_index, sink, 0.0)

    candidate_edges: list[tuple[int, int, float, _FlowEdge]] = []
    for truth_index, item in enumerate(truth):
        truth_bbox = _bbox(item)
        for detection_index, detection in enumerate(detections):
            score = _iou(truth_bbox, _bbox(detection))
            if score >= threshold:
                edge = _add_flow_edge(
                    graph,
                    truth_offset + truth_index,
                    detection_offset + detection_index,
                    -score,
                )
                candidate_edges.append(
                    (truth_index, detection_index, score, edge)
                )

    # Successive shortest augmenting paths produce the minimum-cost matching for
    # each flow cardinality. Continuing until no path exists therefore makes
    # cardinality primary and summed IoU (the negated cost) secondary.
    node_count = len(graph)
    while True:
        distances = [float("inf")] * node_count
        predecessors: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0.0

        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distances[node] == float("inf"):
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity == 0:
                        continue
                    candidate = distances[node] + edge.cost
                    if candidate < distances[edge.to]:
                        distances[edge.to] = candidate
                        predecessors[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break

        if predecessors[sink] is None:
            break

        node = sink
        while node != source:
            predecessor = predecessors[node]
            if predecessor is None:  # pragma: no cover - path invariant
                raise RuntimeError("incomplete augmenting path")
            previous, edge_index = predecessor
            edge = graph[previous][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = previous

    matches = [
        (truth_index, detection_index, score)
        for truth_index, detection_index, score, edge in candidate_edges
        if edge.capacity == 0
    ]
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches


def _bbox(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
    bbox = item.get("bbox")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
            for value in bbox
        )
    ):
        raise ValueError(
            "every ground-truth object and detection requires a finite numeric bbox"
        )
    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")
    return x, y, width, height


def _iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return _safe_ratio(intersection, union)


def _truth_label(item: Mapping[str, Any]) -> str:
    label = item.get("defect")
    if label not in DEFECT_LABELS:
        raise ValueError(f"unknown ground-truth defect label: {label}")
    return str(label)


def _prediction_label(item: Mapping[str, Any]) -> str:
    label = item.get("predicted_defect", item.get("defect"))
    if label not in DEFECT_LABELS:
        raise ValueError(f"unknown predicted defect label: {label}")
    return str(label)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _rounded(value: float) -> float:
    return round(float(value), 6)
