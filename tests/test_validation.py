from __future__ import annotations

import json

import numpy as np
import pytest

from inspectflow import (
    CentroidTracker,
    INSPECTION_CONFIG_FIELDS,
    InspectionConfig,
    PartInspector,
)
from inspectflow.evaluation import evaluate_run
from inspectflow.generator import SyntheticConveyor


@pytest.mark.parametrize(
    "override",
    [
        {"expected_width_px": 0},
        {"dimension_tolerance": 1.01},
        {"expected_color_bgr": [20, 30]},
        {"color_distance_threshold": -1},
        {"feature_min_radius": 12, "feature_max_radius": 12},
        {"min_part_area": 100, "max_part_area": 99},
        {"segmentation_value_min": 256},
        {"morph_kernel_size": 4},
        {"morph_kernel_size": 33},
        {"border_margin_px": -1},
        {"not_a_threshold": 2},
    ],
)
def test_invalid_inspection_config_is_rejected(override: dict) -> None:
    with pytest.raises(ValueError):
        InspectionConfig.from_dict(override)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_count": 0},
        {"scenario": "rainstorm"},
        {"seed": True},
        {"width": 120},
    ],
)
def test_invalid_generator_input_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SyntheticConveyor(**kwargs)


def test_invalid_image_and_tracking_input_are_rejected() -> None:
    inspector = PartInspector()
    with pytest.raises(ValueError, match="uint8"):
        inspector.inspect(np.zeros((20, 20, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        inspector.inspect(np.zeros((20, 20), dtype=np.uint8))

    tracker = CentroidTracker()
    tracker.update([], frame_index=2)
    with pytest.raises(ValueError, match="monotonically"):
        tracker.update([], frame_index=2)
    with pytest.raises(ValueError, match="centroid"):
        CentroidTracker().update([{"bbox": [0, 0, 1, 1]}], frame_index=0)


def test_config_defaults_and_field_metadata_share_one_json_safe_source() -> None:
    config = InspectionConfig.from_dict(None)

    assert InspectionConfig.from_dict({"morph_kernel_size": 31}).morph_kernel_size == 31
    assert INSPECTION_CONFIG_FIELDS["morph_kernel_size"]["max"] == 31
    assert set(INSPECTION_CONFIG_FIELDS) == set(config.to_dict())
    assert {
        name: specification["default"]
        for name, specification in INSPECTION_CONFIG_FIELDS.items()
    } == config.to_dict()
    json.dumps(INSPECTION_CONFIG_FIELDS)


def test_only_none_selects_the_default_inspector_config() -> None:
    assert PartInspector(None).config == InspectionConfig()
    with pytest.raises(ValueError, match="InspectionConfig"):
        PartInspector({})  # type: ignore[arg-type]


def test_invalid_evaluation_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_run([], [], iou_threshold=0)
    with pytest.raises(ValueError, match="one entry"):
        evaluate_run(SyntheticConveyor(frame_count=2).generate(), [])
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_run([], [], iou_threshold="0.3")  # type: ignore[arg-type]
