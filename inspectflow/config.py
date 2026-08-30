"""Validated configuration for the computer-vision inspection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any, Mapping


MAX_MORPH_KERNEL_SIZE = 31


@dataclass(frozen=True, slots=True)
class InspectionConfig:
    """Thresholds used to segment and classify conveyor parts.

    Distances and dimensions are expressed in pixels. ``dimension_tolerance`` is
    a fractional tolerance around the expected width and height (for example,
    ``0.18`` means +/-18%). Colors use OpenCV's BGR channel order.
    """

    expected_width_px: int = 80
    expected_height_px: int = 50
    dimension_tolerance: float = 0.18
    expected_color_bgr: tuple[int, int, int] = (70, 170, 225)
    color_distance_threshold: float = 60.0
    feature_min_radius: float = 6.0
    feature_max_radius: float = 14.0
    min_part_area: float = 700.0
    max_part_area: float = 9000.0
    segmentation_saturation_min: int = 50
    segmentation_value_min: int = 65
    morph_kernel_size: int = 5
    border_margin_px: int = 2

    def __post_init__(self) -> None:
        # JSON sends colors as lists. Normalize them once so callers see a
        # consistent, immutable value from this frozen dataclass.
        if isinstance(self.expected_color_bgr, list):
            object.__setattr__(
                self, "expected_color_bgr", tuple(self.expected_color_bgr)
            )
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` with a useful message for invalid thresholds."""

        integer_fields = {
            "expected_width_px": self.expected_width_px,
            "expected_height_px": self.expected_height_px,
            "segmentation_saturation_min": self.segmentation_saturation_min,
            "segmentation_value_min": self.segmentation_value_min,
            "morph_kernel_size": self.morph_kernel_size,
            "border_margin_px": self.border_margin_px,
        }
        for name, integer_value in integer_fields.items():
            if isinstance(integer_value, bool) or not isinstance(
                integer_value, Integral
            ):
                raise ValueError(f"{name} must be an integer")

        numeric_fields = {
            "dimension_tolerance": self.dimension_tolerance,
            "color_distance_threshold": self.color_distance_threshold,
            "feature_min_radius": self.feature_min_radius,
            "feature_max_radius": self.feature_max_radius,
            "min_part_area": self.min_part_area,
            "max_part_area": self.max_part_area,
        }
        for name, numeric_value in numeric_fields.items():
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, Real)
                or not isfinite(float(numeric_value))
            ):
                raise ValueError(f"{name} must be a finite number")

        if self.expected_width_px <= 0 or self.expected_height_px <= 0:
            raise ValueError("expected dimensions must be positive")
        if not 0.0 <= self.dimension_tolerance <= 1.0:
            raise ValueError("dimension_tolerance must be between 0 and 1")
        if (
            not isinstance(self.expected_color_bgr, (tuple, list))
            or len(self.expected_color_bgr) != 3
            or any(
                isinstance(channel, bool)
                or not isinstance(channel, Real)
                or not isfinite(float(channel))
                or not 0 <= channel <= 255
                for channel in self.expected_color_bgr
            )
        ):
            raise ValueError(
                "expected_color_bgr must contain three values from 0 to 255"
            )
        if self.color_distance_threshold < 0:
            raise ValueError("color_distance_threshold must be non-negative")
        if self.feature_min_radius < 0:
            raise ValueError("feature_min_radius must be non-negative")
        if self.feature_max_radius <= self.feature_min_radius:
            raise ValueError(
                "feature_max_radius must be greater than feature_min_radius"
            )
        if self.min_part_area <= 0 or self.max_part_area <= self.min_part_area:
            raise ValueError("part area bounds must be positive and ordered")
        if not 0 <= self.segmentation_saturation_min <= 255:
            raise ValueError("segmentation_saturation_min must be between 0 and 255")
        if not 0 <= self.segmentation_value_min <= 255:
            raise ValueError("segmentation_value_min must be between 0 and 255")
        if (
            self.morph_kernel_size < 1
            or self.morph_kernel_size > MAX_MORPH_KERNEL_SIZE
            or self.morph_kernel_size % 2 == 0
        ):
            raise ValueError(
                "morph_kernel_size must be an odd integer between 1 and "
                f"{MAX_MORPH_KERNEL_SIZE}"
            )
        if self.border_margin_px < 0:
            raise ValueError("border_margin_px must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the configuration."""

        result = asdict(self)
        result["expected_color_bgr"] = list(self.expected_color_bgr)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None = None) -> "InspectionConfig":
        """Build a validated config from partial JSON-style overrides."""

        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("configuration must be an object")
        unknown = set(values).difference(cls.__dataclass_fields__)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown configuration field(s): {names}")
        try:
            return cls(**dict(values))
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


# JSON-safe field descriptions are the single source of truth for clients that
# render configuration controls.  The ranges are deliberately useful UI ranges;
# validation remains responsible for cross-field rules such as ordered radii and
# part-area bounds.
INSPECTION_CONFIG_FIELDS: dict[str, dict[str, Any]] = {
    "expected_width_px": {
        "type": "integer",
        "default": 80,
        "min": 20,
        "max": 200,
        "step": 1,
    },
    "expected_height_px": {
        "type": "integer",
        "default": 50,
        "min": 20,
        "max": 150,
        "step": 1,
    },
    "dimension_tolerance": {
        "type": "number",
        "default": 0.18,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "expected_color_bgr": {
        "type": "bgr",
        "default": [70, 170, 225],
        "min": 0,
        "max": 255,
        "step": 1,
    },
    "color_distance_threshold": {
        "type": "number",
        "default": 60.0,
        "min": 0.0,
        "max": 442.0,
        "step": 1.0,
    },
    "feature_min_radius": {
        "type": "number",
        "default": 6.0,
        "min": 0.0,
        "max": 40.0,
        "step": 1.0,
    },
    "feature_max_radius": {
        "type": "number",
        "default": 14.0,
        "min": 0.0,
        "max": 40.0,
        "step": 1.0,
    },
    "min_part_area": {
        "type": "number",
        "default": 700.0,
        "min": 1.0,
        "max": 30_000.0,
        "step": 1.0,
    },
    "max_part_area": {
        "type": "number",
        "default": 9_000.0,
        "min": 1.0,
        "max": 30_000.0,
        "step": 1.0,
    },
    "segmentation_saturation_min": {
        "type": "integer",
        "default": 50,
        "min": 0,
        "max": 255,
        "step": 1,
    },
    "segmentation_value_min": {
        "type": "integer",
        "default": 65,
        "min": 0,
        "max": 255,
        "step": 1,
    },
    "morph_kernel_size": {
        "type": "integer",
        "default": 5,
        "min": 1,
        "max": MAX_MORPH_KERNEL_SIZE,
        "step": 2,
    },
    "border_margin_px": {
        "type": "integer",
        "default": 2,
        "min": 0,
        "max": 30,
        "step": 1,
    },
}
