"""Typed HTTP contracts for the InspectFlow demo API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Scenario = Literal["nominal", "poor_lighting"]
DetectorChoice = Literal["classical", "cnn"]
Verdict = Literal["pass", "fail"]
DefectLabel = Literal[
    "normal", "incorrect_dimensions", "wrong_color", "missing_feature"
]
ColorChannel = Annotated[int, Field(ge=0, le=255, strict=True)]


class StrictModel(BaseModel):
    """Base model that rejects misspelled or undeclared API fields."""

    model_config = ConfigDict(extra="forbid")


class InspectionConfigOverrides(StrictModel):
    """Partial inspection configuration accepted by run requests."""

    expected_width_px: int | None = Field(default=None, ge=1, le=4096, strict=True)
    expected_height_px: int | None = Field(default=None, ge=1, le=4096, strict=True)
    dimension_tolerance: float | None = Field(
        default=None, ge=0.0, le=1.0, strict=True
    )
    expected_color_bgr: tuple[ColorChannel, ColorChannel, ColorChannel] | None = None
    color_distance_threshold: float | None = Field(
        default=None, ge=0.0, le=442.0, strict=True
    )
    feature_min_radius: float | None = Field(
        default=None, ge=0.0, le=2048.0, strict=True
    )
    feature_max_radius: float | None = Field(
        default=None, gt=0.0, le=2048.0, strict=True
    )
    min_part_area: float | None = Field(
        default=None, gt=0.0, le=16_777_216.0, strict=True
    )
    max_part_area: float | None = Field(
        default=None, gt=0.0, le=16_777_216.0, strict=True
    )
    segmentation_saturation_min: int | None = Field(
        default=None, ge=0, le=255, strict=True
    )
    segmentation_value_min: int | None = Field(default=None, ge=0, le=255, strict=True)
    morph_kernel_size: int | None = Field(default=None, ge=1, le=31, strict=True)
    border_margin_px: int | None = Field(default=None, ge=0, le=2048, strict=True)


class RunRequest(StrictModel):
    """Settings for one bounded inspection replay."""

    scenario: Scenario = "nominal"
    detector: DetectorChoice = "classical"
    frame_count: int = Field(default=96, ge=8, le=240, strict=True)
    config: InspectionConfigOverrides = Field(default_factory=InspectionConfigOverrides)


class InspectionConfigPayload(StrictModel):
    expected_width_px: int
    expected_height_px: int
    dimension_tolerance: float
    expected_color_bgr: list[int]
    color_distance_threshold: float
    feature_min_radius: float
    feature_max_radius: float
    min_part_area: float
    max_part_area: float
    segmentation_saturation_min: int
    segmentation_value_min: int
    morph_kernel_size: int
    border_margin_px: int


class Measurements(StrictModel):
    """Measured part properties. The CNN detector reports only box size."""

    width_px: int
    height_px: int
    contour_area_px: float | None = None
    color_bgr: list[float] | None = None
    color_distance: float | None = None
    feature_radius_px: float | None = None
    feature_center: list[float] | None = None


class Detection(StrictModel):
    bbox: list[int]
    centroid: list[float]
    predicted_defect: DefectLabel
    verdict: Verdict
    measurements: Measurements
    track_id: int
    score: float | None = None


class FramePayload(StrictModel):
    index: int
    image_url: str
    detections: list[Detection]
    accepted: int
    rejected: int
    defect_counts: dict[str, int]


class RunSummary(StrictModel):
    accepted: int
    rejected: int
    total_inspected: int
    observed_tracks: int = 0
    undecided_tracks: int = 0
    defect_counts: dict[str, int]


class RunExports(StrictModel):
    csv_url: str


class DetectionMetrics(StrictModel):
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


class LabelledMatrix(StrictModel):
    labels: list[str]
    matrix: list[list[int]]


class BinaryClassificationMetrics(StrictModel):
    task: Literal["binary_defect"]
    positive_class: Literal["defective"]
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion_matrix: LabelledMatrix


class DefectClassificationMetrics(StrictModel):
    labels: list[DefectLabel]
    confusion_matrix: list[list[int]]
    ground_truth_distribution: dict[str, int]
    matched_ground_truth_distribution: dict[str, int]
    matched_objects: int


class TrackingMetrics(StrictModel):
    id_switches: int
    tracked_transitions: int
    tracked_parts: int
    continuity: float | None


class EvaluationTotals(StrictModel):
    frames: int
    ground_truth_objects: int
    predicted_objects: int
    matched_objects: int


class FrameEvaluation(StrictModel):
    """Per-frame detection and classification scored against ground truth."""

    detection: DetectionMetrics
    classification: BinaryClassificationMetrics
    defect_classification: DefectClassificationMetrics
    tracking: TrackingMetrics
    totals: EvaluationTotals


class GateOutcomeCounts(StrictModel):
    correct_pass: int
    correct_reject: int
    wrong_defect: int
    escape: int
    false_reject: int
    spurious: int
    missed: int


class GateEvaluation(StrictModel):
    """One decision per part at the gate, scored against the true part."""

    gate_x: float
    iou_threshold: float
    parts: int
    good_parts: int
    defective_parts: int
    decisions: int
    counts: GateOutcomeCounts
    verdict_accuracy: float
    escape_rate: float
    false_reject_rate: float
    miss_rate: float


class RunEvaluation(StrictModel):
    frames: FrameEvaluation
    gate: GateEvaluation


class StageTiming(StrictModel):
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class RunTiming(StrictModel):
    stages: dict[str, StageTiming]
    inspection_fps: float | None


class RunResponse(StrictModel):
    run_id: str
    scenario: Scenario
    detector: DetectorChoice
    frame_count: int
    fps: int
    image_width: int
    image_height: int
    config: InspectionConfigPayload
    frames: list[FramePayload]
    summary: RunSummary
    evaluation: RunEvaluation
    timing: RunTiming
    exports: RunExports


class ErrorResponse(BaseModel):
    detail: str | list[dict[str, Any]]


class HealthResponse(StrictModel):
    status: Literal["ok"]
    service: Literal["inspect-flow"]


class ReadinessResponse(StrictModel):
    status: Literal["ready", "not_ready"]
    dashboard: Literal["available", "unavailable"]
    cnn_model: Literal["available", "unavailable"]


class ApiMetaResponse(StrictModel):
    defaults: dict[str, Any]
    limits: dict[str, dict[str, int]]
    scenarios: list[dict[str, str]]
    detectors: list[dict[str, str]]
    config_fields: dict[str, dict[str, Any]]
    model: dict[str, Any] | None
