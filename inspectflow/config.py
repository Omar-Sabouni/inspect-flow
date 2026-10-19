"""Thresholds for the classical inspection rules."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InspectionConfig(BaseModel):
    """Sizes are in pixels of the 640 x 320 synthetic frames; colors are BGR.

    The API accepts this model directly, so the bounds here are also the
    request validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_width_px: int = Field(80, ge=1)
    expected_height_px: int = Field(50, ge=1)
    dimension_tolerance: float = Field(0.18, ge=0, le=1)
    expected_color_bgr: tuple[int, int, int] = (70, 170, 225)
    color_distance_threshold: float = Field(60.0, ge=0)
    feature_min_radius: float = Field(6.0, ge=0)
    feature_max_radius: float = Field(14.0, gt=0)
    min_part_area: float = Field(700.0, gt=0)
    max_part_area: float = Field(9000.0, gt=0)
    segmentation_saturation_min: int = Field(50, ge=0, le=255)
    segmentation_value_min: int = Field(65, ge=0, le=255)
    morph_kernel_size: int = Field(5, ge=1, le=31)
    border_margin_px: int = Field(2, ge=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> InspectionConfig:
        if self.feature_min_radius >= self.feature_max_radius:
            raise ValueError("feature_min_radius must be below feature_max_radius")
        if self.min_part_area >= self.max_part_area:
            raise ValueError("min_part_area must be below max_part_area")
        if self.morph_kernel_size % 2 == 0:
            raise ValueError("morph_kernel_size must be odd")
        return self
