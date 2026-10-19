"""FastAPI app: run an inspection on a synthetic scene and serve the replay."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import InspectionConfig
from .detectors import DetectorName
from .evaluation import evaluate_gate, evaluate_run
from .generator import SyntheticConveyor
from .pipeline import GATE_FRACTION, InspectionPipeline, annotate_frame, decision_csv, make_detector

STATIC_DIR = Path(__file__).parent / "static"
KEEP_RUNS = 6
FPS = 12


class RunRequest(BaseModel):
    scenario: Literal["nominal", "poor_lighting"] = "nominal"
    detector: DetectorName = "classical"
    frame_count: int = Field(96, ge=8, le=240)
    config: InspectionConfig = InspectionConfig()


@dataclass
class Run:
    payload: dict[str, Any]
    frames: list[bytes]
    csv: str


app = FastAPI(title="InspectFlow")
_runs: OrderedDict[str, Run] = OrderedDict()
_runs_lock = threading.Lock()


def build_run(request: RunRequest, seed: int = 7) -> Run:
    conveyor = SyntheticConveyor(seed, request.frame_count, request.scenario)
    gate_x = round(conveyor.width * GATE_FRACTION)
    pipeline = InspectionPipeline(make_detector(request.detector, request.config), gate_x)
    run_id = uuid.uuid4().hex[:12]

    truth, detections, jpegs = [], [], []
    for frame in conveyor.iter_frames():
        tracked, _ = pipeline.process(frame.image, frame.index)
        truth.append(frame.ground_truth)
        detections.append(tracked)
        with pipeline.timed("annotate"):
            image = annotate_frame(
                frame.image, tracked, frame_index=frame.index, label="", gate_x=gate_x, captions=False
            )
            jpegs.append(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())

    decisions = pipeline.decisions
    gate = evaluate_gate(decisions, truth, gate_x)
    outcomes = gate.pop("outcomes")
    frames = []
    for index, tracked in enumerate(detections):
        so_far = [d for d in decisions if d["decision_frame"] <= index]
        frames.append(
            {
                "detections": tracked,
                "accepted": sum(d["verdict"] == "pass" for d in so_far),
                "rejected": sum(d["verdict"] == "fail" for d in so_far),
            }
        )
    payload = {
        "run_id": run_id,
        "scenario": request.scenario,
        "detector": request.detector,
        "fps": FPS,
        "frames": frames,
        "decisions": len(decisions),
        "undecided_tracks": pipeline.observed_tracks - len(decisions),
        "evaluation": {"frames": evaluate_run(truth, detections), "gate": gate},
        "timing": pipeline.timing_summary(),
    }
    return Run(payload, jpegs, decision_csv(decisions, outcomes))


@app.post("/api/runs")
def create_run(request: RunRequest) -> dict[str, Any]:
    run = build_run(request)
    with _runs_lock:
        _runs[run.payload["run_id"]] = run
        while len(_runs) > KEEP_RUNS:
            _runs.popitem(last=False)
    return run.payload


def _get_run(run_id: str) -> Run:
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found or expired")
    return run


@app.get("/api/runs/{run_id}/frames/{index}.jpg")
def frame_image(run_id: str, index: int) -> Response:
    frames = _get_run(run_id).frames
    if not 0 <= index < len(frames):
        raise HTTPException(404, "frame not found")
    return Response(frames[index], media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/runs/{run_id}/decisions.csv")
def decisions_csv(run_id: str) -> Response:
    headers = {"Content-Disposition": f'attachment; filename="inspectflow-{run_id}.csv"'}
    return Response(_get_run(run_id).csv, media_type="text/csv", headers=headers)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
