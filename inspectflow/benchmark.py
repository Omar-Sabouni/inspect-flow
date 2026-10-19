"""Repeatable accuracy and latency benchmark over seeds, scenarios and detectors."""

from __future__ import annotations

import platform
from collections.abc import Callable, Sequence
from statistics import mean, pstdev
from typing import Any

import cv2
import numpy as np

from .detectors import DETECTOR_NAMES, DetectorName, model_sha256
from .evaluation import evaluate_gate, evaluate_run
from .generator import FrameGroundTruth, SyntheticConveyor
from .pipeline import INSPECTION_GATE_FRACTION, InspectionPipeline, make_detector

SCENARIOS = ("nominal", "poor_lighting")
# Test seeds. Validation (100-104) and training (>= 1000) never overlap them.
TEST_SEEDS = tuple(range(7, 27))
# Columns reported for every detector/scenario pair: (key, heading, kind).
# A count's denominator is named by COUNT_DENOMINATORS.
METRICS: tuple[tuple[str, str, str], ...] = (
    ("gate_verdict_accuracy", "Gate verdict accuracy", "ratio"),
    ("escapes", "Escapes", "count"),
    ("false_rejects", "False rejects", "count"),
    ("missed", "Missed at gate", "count"),
    ("detection_recall", "Detection recall", "ratio"),
    ("detection_precision", "Detection precision", "ratio"),
    ("defect_accuracy", "Defect class accuracy", "ratio"),
    ("detect_p50_ms", "Detect p50 (ms)", "ms"),
    ("detect_p95_ms", "Detect p95 (ms)", "ms"),
)
COUNT_DENOMINATORS = {
    "escapes": "defective_parts",
    "false_rejects": "good_parts",
    "missed": "parts_at_gate",
}


def run_case(
    detector: DetectorName, scenario: str, seed: int, frame_count: int
) -> dict[str, Any]:
    """Run one seeded scene and return flat metrics for it."""

    conveyor = SyntheticConveyor(seed=seed, frame_count=frame_count, scenario=scenario)
    gate_x = round(conveyor.width * INSPECTION_GATE_FRACTION)
    pipeline = InspectionPipeline(make_detector(detector), gate_x=gate_x)
    truth: list[FrameGroundTruth] = []
    detections: list[list[dict[str, Any]]] = []
    for frame in conveyor.iter_frames():
        tracked, _ = pipeline.process(frame.image, frame.index)
        detections.append(tracked)
        truth.append(frame.truth_metadata())

    frames = evaluate_run(truth, detections)
    gate = evaluate_gate(pipeline.decisions, truth, gate_x)
    matrix = np.asarray(frames["defect_classification"]["confusion_matrix"])
    matched = int(matrix.sum())
    timing = pipeline.timer.summary()["detect"]
    return {
        "detector": detector,
        "scenario": scenario,
        "seed": seed,
        "frames": frame_count,
        "parts_at_gate": gate["parts"],
        "good_parts": gate["good_parts"],
        "defective_parts": gate["defective_parts"],
        "detection_recall": frames["detection"]["recall"],
        "detection_precision": frames["detection"]["precision"],
        "defect_accuracy": round(float(np.trace(matrix)) / matched, 6) if matched else 0.0,
        "gate_verdict_accuracy": gate["verdict_accuracy"],
        "escapes": gate["counts"]["escape"],
        "false_rejects": gate["counts"]["false_reject"],
        "missed": gate["counts"]["missed"],
        "detect_p50_ms": timing["p50_ms"],
        "detect_p95_ms": timing["p95_ms"],
    }


def run_benchmark(
    *,
    detectors: Sequence[DetectorName] = DETECTOR_NAMES,
    scenarios: Sequence[str] = SCENARIOS,
    seeds: Sequence[int] = TEST_SEEDS,
    frame_count: int = 240,
    threads: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run every combination and aggregate the results per detector/scenario."""

    if threads is not None:
        cv2.setNumThreads(threads)
    cases: list[dict[str, Any]] = []
    for detector in detectors:
        for scenario in scenarios:
            for seed in seeds:
                case = run_case(detector, scenario, seed, frame_count)
                cases.append(case)
                if progress is not None:
                    progress(case)

    groups: list[dict[str, Any]] = []
    for detector in detectors:
        for scenario in scenarios:
            rows = [
                case
                for case in cases
                if case["detector"] == detector and case["scenario"] == scenario
            ]
            summary: dict[str, Any] = {
                "detector": detector,
                "scenario": scenario,
            }
            for key in ("parts_at_gate", "good_parts", "defective_parts"):
                summary[key] = sum(row[key] for row in rows)
            for key, _, kind in METRICS:
                values = [float(row[key]) for row in rows]
                summary[key] = (
                    {"total": int(sum(values))}
                    if kind == "count"
                    else {
                        "mean": round(mean(values), 4),
                        "std": round(pstdev(values), 4),
                    }
                )
            groups.append(summary)

    return {
        "settings": {
            "seeds": list(seeds),
            "frames_per_seed": frame_count,
            "opencv_threads": cv2.getNumThreads(),
        },
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "machine": platform.machine(),
            "processor": _processor_name(),
            "cnn_model_sha256": model_sha256(),
        },
        "summary": groups,
        "cases": cases,
    }


def format_markdown(report: dict[str, Any]) -> str:
    """Render the aggregated benchmark as a GitHub-flavoured Markdown table."""

    headings = ["Detector", "Scenario", *(heading for _, heading, _ in METRICS)]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join(["---", "---"] + ["---:"] * len(METRICS)) + " |",
    ]
    for group in report["summary"]:
        cells = [group["detector"], group["scenario"].replace("_", " ")]
        for key, _, kind in METRICS:
            value = group[key]
            if kind == "count":
                denominator = group[COUNT_DENOMINATORS[key]]
                cells.append(f"{value['total']} of {denominator}")
            elif kind == "ratio":
                cells.append(f"{value['mean'] * 100:.1f}% ± {value['std'] * 100:.1f}")
            else:
                cells.append(f"{value['mean']:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    settings = report["settings"]
    environment = report["environment"]
    lines += [
        "",
        f"{_describe_seeds(settings['seeds'])}, {settings['frames_per_seed']} frames "
        "each. Ratios are mean ± standard deviation across seeds, in percentage "
        "points. Escapes are out of defective parts at the gate, false rejects out "
        "of good parts, and misses out of all parts.",
        "",
        f"{environment['processor']}, OpenCV {environment['opencv']} with "
        f"{settings['opencv_threads']} thread(s), Python {environment['python']}. "
        f"CNN model SHA-256 `{environment['cnn_model_sha256'][:12]}`.",
    ]
    return "\n".join(lines) + "\n"


def _describe_seeds(seeds: Sequence[int]) -> str:
    ordered = sorted(seeds)
    if len(ordered) > 2 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{len(ordered)} seeds ({ordered[0]}–{ordered[-1]})"
    return f"Seeds {', '.join(str(seed) for seed in ordered)}"


def _processor_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()
