"""FastAPI application and in-memory runtime for InspectFlow."""

from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from inspectflow import INSPECTION_CONFIG_FIELDS, InspectionConfig
from inspectflow.api_models import (
    ApiMetaResponse,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    RunRequest,
    RunResponse,
)
from inspectflow.detectors import (
    DETECTOR_NAMES,
    DetectorName,
    default_model_path,
    load_model_card,
)
from inspectflow.evaluation import evaluate_gate, evaluate_run
from inspectflow.generator import SyntheticConveyor
from inspectflow.pipeline import (
    DECISION_LOG_FIELDS,
    INSPECTION_GATE_FRACTION,
    InspectionPipeline,
    annotate_frame,
    decision_counts,
    decision_rows,
    decision_timeline,
    make_detector,
    rows_to_csv,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
REQUIRED_STATIC_FILES = (
    "index.html",
    "styles.css",
    "app.js",
    "api.js",
    "charts.js",
    "model.js",
    "playback.js",
)
MAX_CACHED_RUNS = 6
MAX_CACHE_BYTES = 64 * 1024 * 1024
DEFAULT_FPS = 12
EVALUATION_LOG_FIELDS = (
    *DECISION_LOG_FIELDS,
    "truth_part_id",
    "truth_defect",
    "outcome",
)
DETECTOR_OPTIONS = (
    {
        "value": "classical",
        "label": "Classical rules",
        "description": "HSV segmentation and measured thresholds",
    },
    {
        "value": "cnn",
        "label": "CNN",
        "description": "Learned detector (ONNX, OpenCV DNN)",
    },
)


@dataclass(slots=True)
class RunArtifact:
    """All generated output for one bounded, in-memory run."""

    payload: dict[str, Any]
    jpeg_frames: list[bytes]
    csv_text: str
    size_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        """Cache a stable conservative estimate used by the in-memory LRU."""

        self.size_bytes = (
            sum(len(frame) for frame in self.jpeg_frames)
            + len(self.csv_text.encode("utf-8"))
            + len(json.dumps(self.payload, separators=(",", ":")).encode("utf-8"))
        )


class ArtifactTooLargeError(ValueError):
    """Raised when one run cannot fit in the configured cache budget."""


class RunStore:
    """Thread-safe LRU constrained by both entry count and approximate bytes."""

    def __init__(
        self,
        *,
        max_runs: int = MAX_CACHED_RUNS,
        max_bytes: int = MAX_CACHE_BYTES,
    ) -> None:
        if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 1:
            raise ValueError("max_runs must be a positive integer")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.max_runs = max_runs
        self.max_bytes = max_bytes
        self._items: "OrderedDict[str, RunArtifact]" = OrderedDict()
        self._size_bytes = 0
        self._lock = threading.RLock()

    def put(self, run_id: str, artifact: RunArtifact) -> None:
        if artifact.size_bytes > self.max_bytes:
            raise ArtifactTooLargeError(
                f"run artifact needs {artifact.size_bytes} bytes but the cache "
                f"budget is {self.max_bytes} bytes"
            )
        with self._lock:
            replaced = self._items.pop(run_id, None)
            if replaced is not None:
                self._size_bytes -= replaced.size_bytes
            self._items[run_id] = artifact
            self._size_bytes += artifact.size_bytes
            self._items.move_to_end(run_id)
            while self._items and (
                len(self._items) > self.max_runs or self._size_bytes > self.max_bytes
            ):
                _, evicted = self._items.popitem(last=False)
                self._size_bytes -= evicted.size_bytes

    def get(self, run_id: str) -> RunArtifact | None:
        with self._lock:
            artifact = self._items.get(run_id)
            if artifact is not None:
                self._items.move_to_end(run_id)
            return artifact

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._size_bytes = 0

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"runs": len(self._items), "bytes": self._size_bytes}


def dashboard() -> FileResponse:
    """Serve the zero-build dashboard."""

    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=503, detail="dashboard assets are unavailable")
    return FileResponse(index)


async def health() -> dict[str, str]:
    return {"status": "ok", "service": "inspect-flow"}


async def readiness(response: Response) -> dict[str, str]:
    dashboard = all((STATIC_DIR / name).is_file() for name in REQUIRED_STATIC_FILES)
    model = default_model_path().is_file()
    ready = dashboard and model
    response.status_code = 200 if ready else 503
    return {
        "status": "ready" if ready else "not_ready",
        "dashboard": "available" if dashboard else "unavailable",
        "cnn_model": "available" if model else "unavailable",
    }


async def api_meta() -> dict[str, Any]:
    config = InspectionConfig()
    return {
        "defaults": {
            "scenario": "nominal",
            "detector": "classical",
            "frame_count": 96,
            "config": config.to_dict(),
        },
        "limits": {
            "frame_count": {"min": 8, "max": 240, "step": 1},
        },
        "scenarios": [
            {"value": "nominal", "label": "Nominal"},
            {"value": "poor_lighting", "label": "Poor lighting"},
        ],
        "detectors": list(DETECTOR_OPTIONS),
        "config_fields": INSPECTION_CONFIG_FIELDS,
        "model": _model_summary(),
    }


def _model_summary() -> dict[str, Any] | None:
    try:
        card = load_model_card()
    except (OSError, ValueError):
        return None
    return {
        "name": card.get("model"),
        "sha256": card.get("sha256"),
        "parameters": card.get("parameters"),
        "input": card.get("input"),
        "labels": card.get("labels"),
    }


def create_run(body: RunRequest, request: Request) -> dict[str, Any]:
    """Run the inspection pipeline and retain its replay in memory."""

    build_slot: threading.BoundedSemaphore = request.app.state.build_slot
    if not build_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="another inspection run is already in progress",
            headers={"Retry-After": "1"},
        )
    try:
        try:
            config = InspectionConfig.from_dict(
                body.config.model_dump(exclude_none=True)
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        artifact = build_run_artifact(
            scenario=body.scenario,
            frame_count=body.frame_count,
            config=config,
            detector=body.detector,
        )
        run_id = artifact.payload["run_id"]
        try:
            request.app.state.run_store.put(run_id, artifact)
        except ArtifactTooLargeError as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from exc
        return artifact.payload
    finally:
        build_slot.release()


def frame_image(run_id: str, frame_index: int, request: Request) -> Response:
    artifact = _get_run(request.app.state.run_store, run_id)
    if frame_index < 0 or frame_index >= len(artifact.jpeg_frames):
        raise HTTPException(status_code=404, detail="frame not found")
    return Response(
        artifact.jpeg_frames[frame_index],
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


def inspection_log(run_id: str, request: Request) -> Response:
    artifact = _get_run(request.app.state.run_store, run_id)
    return Response(
        artifact.csv_text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="inspect-flow-{run_id}-log.csv"'
        },
    )


def build_run_artifact(
    *,
    scenario: str = "nominal",
    seed: int = 7,
    frame_count: int = 96,
    config: InspectionConfig | None = None,
    detector: DetectorName = "classical",
) -> RunArtifact:
    """Build a complete, ground-truth-scored run without filesystem writes."""

    resolved_config = InspectionConfig() if config is None else config
    if not isinstance(resolved_config, InspectionConfig):
        raise ValueError("config must be an InspectionConfig")
    if detector not in DETECTOR_NAMES:
        raise ValueError(f"detector must be one of: {', '.join(DETECTOR_NAMES)}")
    conveyor = SyntheticConveyor(
        seed=seed,
        frame_count=frame_count,
        scenario=scenario,
    )
    run_id = uuid.uuid4().hex[:12]
    image_width, image_height = conveyor.width, conveyor.height
    gate_x = round(image_width * INSPECTION_GATE_FRACTION)
    pipeline = InspectionPipeline(
        make_detector(detector, resolved_config), gate_x=gate_x
    )

    detections_by_frame: list[list[dict[str, Any]]] = []
    truth_frames = []
    jpeg_frames: list[bytes] = []
    for generated in conveyor.iter_frames():
        tracked, _ = pipeline.process(generated.image, generated.index)
        detections_by_frame.append(tracked)
        truth_frames.append(generated.truth_metadata())

        with pipeline.timer.measure("annotate"):
            annotated = annotate_frame(
                generated.image,
                tracked,
                frame_index=generated.index,
                label=f"{scenario} {detector}",
                gate_x=gate_x,
                captions=False,
            )
        with pipeline.timer.measure("encode"):
            encoded_ok, encoded = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
            )
        if not encoded_ok:
            raise RuntimeError(f"could not encode frame {generated.index}")
        jpeg_frames.append(encoded.tobytes())

    decisions = pipeline.decisions
    frame_report = evaluate_run(truth_frames, detections_by_frame)
    gate_report = evaluate_gate(decisions, truth_frames, gate_x)
    outcomes = gate_report.pop("outcomes")
    timeline = decision_timeline(decisions, frame_count)
    final_counts = decision_counts(decisions)
    frame_payloads: list[dict[str, Any]] = []
    for index, detections in enumerate(detections_by_frame):
        cumulative = timeline[index]
        frame_payloads.append(
            {
                "index": index,
                "image_url": f"/api/runs/{run_id}/frames/{index}.jpg",
                "detections": detections,
                "accepted": cumulative["accepted"],
                "rejected": cumulative["rejected"],
                "defect_counts": dict(cumulative["defect_counts"]),
            }
        )

    summary = {
        "accepted": final_counts["accepted"],
        "rejected": final_counts["rejected"],
        "total_inspected": final_counts["accepted"] + final_counts["rejected"],
        "observed_tracks": pipeline.observed_tracks,
        "undecided_tracks": pipeline.observed_tracks - len(decisions),
        "defect_counts": dict(final_counts["defect_counts"]),
    }
    stages = pipeline.timer.summary()
    inspect_mean = stages["detect"]["mean_ms"] + stages["track"]["mean_ms"]
    payload = {
        "run_id": run_id,
        "scenario": scenario,
        "detector": detector,
        "frame_count": frame_count,
        "fps": DEFAULT_FPS,
        "image_width": image_width,
        "image_height": image_height,
        "config": resolved_config.to_dict(),
        "frames": frame_payloads,
        "summary": summary,
        "evaluation": {"frames": frame_report, "gate": gate_report},
        "timing": {
            "stages": stages,
            "inspection_fps": round(1000.0 / inspect_mean, 1) if inspect_mean else None,
        },
        "exports": {
            "csv_url": f"/api/runs/{run_id}/inspection-log.csv",
        },
    }
    rows = decision_rows(decisions, outcomes)
    return RunArtifact(
        payload=payload,
        jpeg_frames=jpeg_frames,
        csv_text=rows_to_csv(rows, EVALUATION_LOG_FIELDS),
    )


def _get_run(store: RunStore, run_id: str) -> RunArtifact:
    artifact = store.get(run_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="run not found or expired")
    return artifact


async def _security_headers(request: Request, call_next: Any) -> Response:
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; base-uri 'none'; object-src 'none'; "
        "frame-ancestors 'none'; form-action 'self'; img-src 'self' data:; "
        "script-src 'self'; style-src 'self'; connect-src 'self'",
    )
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
    )
    return response


def create_app(*, run_store: RunStore | None = None) -> FastAPI:
    """Create an isolated application instance for serving or contract tests."""

    application = FastAPI(
        title="InspectFlow",
        version="current",
        description="Computer-vision inspection replay station",
    )
    application.state.run_store = run_store or RunStore()
    application.state.build_slot = threading.BoundedSemaphore(value=1)
    application.middleware("http")(_security_headers)
    application.mount(
        "/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static"
    )

    error_responses = {
        404: {"model": ErrorResponse, "description": "Run or frame not found"},
        422: {"model": ErrorResponse, "description": "Invalid request or configuration"},
        429: {"model": ErrorResponse, "description": "Run generator is busy"},
        507: {
            "model": ErrorResponse,
            "description": "Run exceeds the in-memory cache budget",
        },
    }
    application.add_api_route(
        "/", dashboard, methods=["GET"], include_in_schema=False
    )
    application.add_api_route(
        "/api/health",
        health,
        methods=["GET"],
        response_model=HealthResponse,
        tags=["service"],
    )
    application.add_api_route(
        "/api/ready",
        readiness,
        methods=["GET"],
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
        tags=["service"],
    )
    application.add_api_route(
        "/api/meta",
        api_meta,
        methods=["GET"],
        response_model=ApiMetaResponse,
        tags=["configuration"],
    )
    application.add_api_route(
        "/api/runs",
        create_run,
        methods=["POST"],
        status_code=201,
        response_model=RunResponse,
        responses=error_responses,
        tags=["runs"],
    )
    application.add_api_route(
        "/api/runs/{run_id}/frames/{frame_index}.jpg",
        frame_image,
        methods=["GET"],
        response_class=Response,
        responses={
            200: {"content": {"image/jpeg": {}}, "description": "Annotated frame"},
            404: error_responses[404],
        },
        tags=["runs"],
    )
    application.add_api_route(
        "/api/runs/{run_id}/inspection-log.csv",
        inspection_log,
        methods=["GET"],
        response_class=Response,
        responses={
            200: {
                "content": {"text/csv": {}},
                "description": "Gate-crossing inspection decisions",
            },
            404: error_responses[404],
        },
        tags=["runs"],
    )
    return application


app = create_app()
