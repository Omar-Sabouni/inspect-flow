"""Run the pipeline on a video file, camera or stream, or write a scene to video."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from .detectors import DetectorName
from .generator import SyntheticConveyor
from .pipeline import GATE_FRACTION, InspectionPipeline, annotate_frame, decision_csv, make_detector

CODECS = {".mp4": "mp4v", ".avi": "MJPG"}


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    if path.suffix.lower() not in CODECS:
        raise ValueError("output video must end in .mp4 or .avi")
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*CODECS[path.suffix.lower()]), fps, size)


def inspect_video(
    source: str,
    detector: DetectorName = "classical",
    output: Path | None = None,
    log: Path | None = None,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """``source`` is a file path, a camera index such as ``"0"``, or a stream URL.

    Parts are expected to move left to right, as on the synthetic belt.
    """

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise ValueError(f"could not open video source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 12.0
    pipeline = writer = None
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if pipeline is None:
                pipeline = InspectionPipeline(make_detector(detector), round(image.shape[1] * GATE_FRACTION))
            tracked, decided = pipeline.process(image, index)
            for decision in decided:
                if on_decision:
                    on_decision(decision)
            if output is not None:
                writer = writer or _writer(output, fps, (image.shape[1], image.shape[0]))
                gate_x = round(pipeline.gate_x)
                writer.write(annotate_frame(image, tracked, frame_index=index, label=detector, gate_x=gate_x))
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if pipeline is None:
        raise ValueError(f"no frames could be read from {source}")

    decisions = pipeline.decisions
    if log is not None:
        log.write_text(decision_csv(decisions))
    return {
        "frames": index,
        "passed": sum(d["verdict"] == "pass" for d in decisions),
        "rejected": sum(d["verdict"] == "fail" for d in decisions),
        "undecided_tracks": pipeline.observed_tracks - len(decisions),
        "timing": pipeline.timing_summary(),
    }


def render_scene(output: Path, scenario: str = "nominal", seed: int = 7, frame_count: int = 240) -> None:
    conveyor = SyntheticConveyor(seed, frame_count, scenario)
    writer = _writer(output, 12.0, (conveyor.width, conveyor.height))
    try:
        for frame in conveyor.iter_frames():
            writer.write(frame.image)
    finally:
        writer.release()
