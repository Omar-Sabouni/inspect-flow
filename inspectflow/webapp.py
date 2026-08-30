"""FastAPI application and in-memory runtime for InspectFlow."""

from __future__ import annotations

import csv
import io
import json
import threading
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from inspectflow import (
    CentroidTracker,
    INSPECTION_CONFIG_FIELDS,
    InspectionConfig,
    PartInspector,
)
from inspectflow.api_models import (
    ApiMetaResponse,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    RunRequest,
    RunResponse,
)
from inspectflow.generator import SyntheticConveyor


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
INSPECTION_GATE_FRACTION = 0.25


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
    ready = all((STATIC_DIR / name).is_file() for name in REQUIRED_STATIC_FILES)
    response.status_code = 200 if ready else 503
    return {
        "status": "ready" if ready else "not_ready",
        "dashboard": "available" if ready else "unavailable",
    }


async def api_meta() -> dict[str, Any]:
    config = InspectionConfig()
    return {
        "defaults": {
            "scenario": "nominal",
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
        "config_fields": INSPECTION_CONFIG_FIELDS,
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
) -> RunArtifact:
    """Build a complete run without filesystem writes."""

    resolved_config = InspectionConfig() if config is None else config
    if not isinstance(resolved_config, InspectionConfig):
        raise ValueError("config must be an InspectionConfig")
    conveyor = SyntheticConveyor(
        seed=seed,
        frame_count=frame_count,
        scenario=scenario,
    )
    inspector = PartInspector(resolved_config)
    tracker = CentroidTracker(max_distance=70.0, max_missed=4)
    run_id = uuid.uuid4().hex[:12]
    image_width, image_height = conveyor.width, conveyor.height
    gate_x = round(image_width * INSPECTION_GATE_FRACTION)

    detections_by_frame: list[list[dict[str, Any]]] = []
    jpeg_frames: list[bytes] = []
    observations_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for generated in conveyor.iter_frames():
        detections = inspector.inspect(generated.image)
        tracked = tracker.update(detections, generated.index)

        detections_by_frame.append(tracked)
        for detection in tracked:
            observation = {
                "frame_index": generated.index,
                **detection,
            }
            observations_by_track[int(detection["track_id"])].append(observation)

        annotated = _annotate_frame(
            generated.image,
            tracked,
            frame_index=generated.index,
            scenario=scenario,
            gate_x=gate_x,
        )
        encoded_ok, encoded = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
        )
        if not encoded_ok:
            raise RuntimeError(f"could not encode frame {generated.index}")
        jpeg_frames.append(encoded.tobytes())

    decisions = _gate_decisions(observations_by_track, gate_x)
    log_rows = _build_inspection_rows(decisions)
    decision_timeline = _decision_timeline(decisions, frame_count)
    final_counts = decision_timeline[-1]
    frame_payloads: list[dict[str, Any]] = []
    for index, detections in enumerate(detections_by_frame):
        cumulative = decision_timeline[index]
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
        "observed_tracks": len(observations_by_track),
        "undecided_tracks": len(observations_by_track) - len(decisions),
        "defect_counts": dict(final_counts["defect_counts"]),
    }
    payload = {
        "run_id": run_id,
        "scenario": scenario,
        "frame_count": frame_count,
        "fps": DEFAULT_FPS,
        "image_width": image_width,
        "image_height": image_height,
        "config": resolved_config.to_dict(),
        "frames": frame_payloads,
        "summary": summary,
        "exports": {
            "csv_url": f"/api/runs/{run_id}/inspection-log.csv",
        },
    }
    return RunArtifact(
        payload=payload,
        jpeg_frames=jpeg_frames,
        csv_text=_rows_to_csv(log_rows),
    )


def _annotate_frame(
    image: np.ndarray,
    detections: Sequence[Mapping[str, Any]],
    *,
    frame_index: int,
    scenario: str,
    gate_x: int,
) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    cv2.line(canvas, (gate_x, 53), (gate_x, height - 43), (130, 187, 255), 1)
    cv2.putText(
        canvas,
        "INSPECTION GATE",
        (gate_x - 66, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (185, 216, 255),
        1,
        cv2.LINE_AA,
    )
    for detection in detections:
        x, y, box_width, box_height = (int(value) for value in detection["bbox"])
        failed = detection["verdict"] == "fail"
        color = (72, 92, 255) if failed else (87, 214, 143)
        cv2.rectangle(canvas, (x, y), (x + box_width, y + box_height), color, 2)
        defect = str(detection["predicted_defect"]).replace("_", " ")
        verdict = "FAIL " + defect if failed else "PASS"
        label = f"ID {detection['track_id']}  {verdict}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1
        )
        label_y = max(20, y - 5)
        cv2.rectangle(
            canvas,
            (x, label_y - text_height - baseline - 4),
            (min(width - 1, x + text_width + 8), label_y + baseline),
            (19, 24, 31),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x + 4, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"FRAME {frame_index + 1:03d}  {scenario.upper().replace('_', ' ')}",
        (18, height - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (211, 221, 231),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _gate_decisions(
    observations: Mapping[int, list[dict[str, Any]]], gate_x: int
) -> list[dict[str, Any]]:
    """Return exactly one decision for each confirmed left-to-right crossing."""

    decisions: list[dict[str, Any]] = []
    for track_id, items in sorted(observations.items()):
        previous: dict[str, Any] | None = None
        for current in items:
            if (
                previous is not None
                and float(previous["centroid"][0]) < gate_x
                and float(current["centroid"][0]) >= gate_x
            ):
                copied = dict(current)
                copied["track_id"] = track_id
                copied["decision_frame"] = int(current["frame_index"])
                decisions.append(copied)
                break
            previous = current
    return decisions


def _decision_timeline(
    representatives: list[dict[str, Any]], frame_count: int
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        decided = [
            item for item in representatives if int(item["decision_frame"]) <= frame_index
        ]
        defect_counts = {
            "incorrect_dimensions": 0,
            "wrong_color": 0,
            "missing_feature": 0,
        }
        for item in decided:
            label = str(item["predicted_defect"])
            if label in defect_counts:
                defect_counts[label] += 1
        timeline.append(
            {
                "accepted": sum(item["verdict"] == "pass" for item in decided),
                "rejected": sum(item["verdict"] == "fail" for item in decided),
                "defect_counts": defect_counts,
            }
        )
    return timeline


def _build_inspection_rows(
    representatives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in representatives:
        frame_index = int(item["frame_index"])
        measurements = item.get("measurements", {})
        centroid = item["centroid"]
        bbox = item["bbox"]
        predicted = str(item["predicted_defect"])
        rows.append(
            {
                "track_id": int(item["track_id"]),
                "decision_frame": int(item["decision_frame"]),
                "representative_frame": frame_index,
                "verdict": item["verdict"],
                "predicted_defect": predicted,
                "centroid_x_px": round(float(centroid[0]), 3),
                "centroid_y_px": round(float(centroid[1]), 3),
                "bbox_x": int(bbox[0]),
                "bbox_y": int(bbox[1]),
                "bbox_width": int(bbox[2]),
                "bbox_height": int(bbox[3]),
                "measured_width_px": measurements.get("width_px", ""),
                "measured_height_px": measurements.get("height_px", ""),
                "color_distance": measurements.get("color_distance", ""),
                "feature_radius_px": measurements.get("feature_radius_px", ""),
            }
        )
    return rows


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    fieldnames = [
        "track_id",
        "decision_frame",
        "representative_frame",
        "verdict",
        "predicted_defect",
        "centroid_x_px",
        "centroid_y_px",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "measured_width_px",
        "measured_height_px",
        "color_distance",
        "feature_radius_px",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


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
