"""Run the inspection pipeline on a video file, camera or stream."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from .detectors import DetectorName
from .generator import SyntheticConveyor
from .pipeline import (
    INSPECTION_GATE_FRACTION,
    InspectionPipeline,
    annotate_frame,
    decision_counts,
    decision_rows,
    make_detector,
    rows_to_csv,
)

DEFAULT_FPS = 12.0
FOURCC_BY_SUFFIX = {".mp4": "mp4v", ".avi": "MJPG"}


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a file path, camera index (``"0"``) or stream URL."""

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise ValueError(f"could not open video source: {source}")
    return capture


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open an MP4 (mp4v) or AVI (MJPG) writer chosen by file extension."""

    suffix = path.suffix.lower()
    if suffix not in FOURCC_BY_SUFFIX:
        raise ValueError("output video must end in .mp4 or .avi")
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*FOURCC_BY_SUFFIX[suffix])
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise ValueError(f"could not open video writer for {path}")
    return writer


def inspect_video(
    source: str,
    *,
    detector: DetectorName = "classical",
    output: Path | None = None,
    log: Path | None = None,
    gate_fraction: float = INSPECTION_GATE_FRACTION,
    max_frames: int | None = None,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Inspect every frame of ``source`` and return run totals and timing.

    Parts must move left to right across the gate, as on the synthetic
    conveyor. Classical thresholds are tuned for the synthetic scenes and need
    recalibration for real footage.
    """

    if not 0.0 < gate_fraction < 1.0:
        raise ValueError("gate_fraction must be between 0 and 1")
    capture = open_capture(source)
    writer: cv2.VideoWriter | None = None
    pipeline: InspectionPipeline | None = None
    frame_index = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        while max_frames is None or frame_index < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            height, width = image.shape[:2]
            if pipeline is None:
                pipeline = InspectionPipeline(
                    make_detector(detector),
                    gate_x=round(width * gate_fraction),
                )
            tracked, decisions = pipeline.process(image, frame_index)
            if on_decision is not None:
                for decision in decisions:
                    on_decision(decision)
            if output is not None:
                if writer is None:
                    writer = open_writer(output, fps, (width, height))
                writer.write(
                    annotate_frame(
                        image,
                        tracked,
                        frame_index=frame_index,
                        label=detector,
                        gate_x=round(pipeline.gate_x),
                    )
                )
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if pipeline is None:
        raise ValueError(f"no frames could be read from {source}")
    decisions = pipeline.decisions
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(rows_to_csv(decision_rows(decisions)), encoding="utf-8")
    counts = decision_counts(decisions)
    return {
        "frames": frame_index,
        "detector": detector,
        "observed_tracks": pipeline.observed_tracks,
        "undecided_tracks": pipeline.observed_tracks - len(decisions),
        **counts,
        "timing": pipeline.timer.summary(),
    }


def render_scenario(
    output: Path,
    *,
    scenario: str = "nominal",
    seed: int = 7,
    frame_count: int = 240,
    fps: float = DEFAULT_FPS,
) -> int:
    """Write a synthetic conveyor scene to a video file and return its length."""

    conveyor = SyntheticConveyor(seed=seed, frame_count=frame_count, scenario=scenario)
    writer = open_writer(output, fps, (conveyor.width, conveyor.height))
    try:
        for frame in conveyor.iter_frames():
            writer.write(frame.image)
    finally:
        writer.release()
    return frame_count
