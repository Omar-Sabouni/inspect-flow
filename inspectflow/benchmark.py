"""Accuracy and latency of both detectors over many seeded scenes."""

from __future__ import annotations

import platform
from collections.abc import Callable, Sequence
from statistics import mean, pstdev
from typing import Any

import cv2

from .detectors import DETECTOR_NAMES, DetectorName, model_sha256
from .evaluation import evaluate_gate, evaluate_run
from .generator import SCENARIOS, SyntheticConveyor
from .pipeline import GATE_FRACTION, InspectionPipeline, make_detector

# Validation (100-104) and training (1000+) seeds never overlap these.
TEST_SEEDS = tuple(range(7, 27))


def run_case(detector: DetectorName, scenario: str, seed: int, frame_count: int) -> dict[str, Any]:
    conveyor = SyntheticConveyor(seed, frame_count, scenario)
    gate_x = round(conveyor.width * GATE_FRACTION)
    pipeline = InspectionPipeline(make_detector(detector), gate_x)
    truth, detections = [], []
    for frame in conveyor.iter_frames():
        detections.append(pipeline.process(frame.image, frame.index)[0])
        truth.append(frame.ground_truth)

    frames = evaluate_run(truth, detections)
    gate = evaluate_gate(pipeline.decisions, truth, gate_x)
    timing = pipeline.timing_summary()["detect"]
    return {
        "detector": detector,
        "scenario": scenario,
        "seed": seed,
        "parts": gate["parts"],
        "good_parts": gate["good_parts"],
        "defective_parts": gate["defective_parts"],
        "right_call": gate["verdict_accuracy"],
        "escapes": gate["counts"]["escape"],
        "false_rejects": gate["counts"]["false_reject"],
        "missed": gate["counts"]["missed"],
        "recall": frames["detection"]["recall"],
        "precision": frames["detection"]["precision"],
        "defect_accuracy": frames["defect_accuracy"],
        "p50_ms": timing["p50_ms"],
        "p95_ms": timing["p95_ms"],
    }


def run_benchmark(
    detectors: Sequence[DetectorName] = DETECTOR_NAMES,
    scenarios: Sequence[str] = SCENARIOS,
    seeds: Sequence[int] = TEST_SEEDS,
    frame_count: int = 240,
    threads: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if threads is not None:
        cv2.setNumThreads(threads)
    cases = []
    for detector in detectors:
        for scenario in scenarios:
            for seed in seeds:
                cases.append(run_case(detector, scenario, seed, frame_count))
                if progress:
                    progress(cases[-1])
    return {
        "seeds": list(seeds),
        "frames": frame_count,
        "threads": cv2.getNumThreads(),
        "machine": _cpu_name(),
        "opencv": cv2.__version__,
        "model_sha256": model_sha256(),
        "cases": cases,
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "| Detector | Lighting | Right call | Defects passed | Good parts rejected | Missed "
        "| Recall | Precision | Defect accuracy | p50 ms | p95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in report["cases"]:
        groups.setdefault((case["detector"], case["scenario"]), []).append(case)
    for (detector, scenario), cases in groups.items():
        counts = ("escapes", "false_rejects", "missed", "parts", "good_parts", "defective_parts")
        total = {key: sum(case[key] for case in cases) for key in counts}
        lines.append(
            f"| {detector} | {scenario.replace('_', ' ')} | {_percent(cases, 'right_call')} "
            f"| {total['escapes']} of {total['defective_parts']} "
            f"| {total['false_rejects']} of {total['good_parts']} "
            f"| {total['missed']} of {total['parts']} "
            f"| {_percent(cases, 'recall')} | {_percent(cases, 'precision')} | {_percent(cases, 'defect_accuracy')} "
            f"| {mean(c['p50_ms'] for c in cases):.2f} | {mean(c['p95_ms'] for c in cases):.2f} |"
        )
    seeds = report["seeds"]
    lines += [
        "",
        f"{len(seeds)} seeds ({min(seeds)}–{max(seeds)}), {report['frames']} frames each. "
        "Percentages are the mean ± standard deviation across seeds.",
        "",
        f"{report['machine']}, OpenCV {report['opencv']}, {report['threads']} thread(s). "
        f"Model SHA-256 `{report['model_sha256'][:12]}`.",
    ]
    return "\n".join(lines) + "\n"


def _percent(cases: list[dict[str, Any]], key: str) -> str:
    values = [case[key] * 100 for case in cases]
    return f"{mean(values):.1f}% ± {pstdev(values):.1f}"


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()
