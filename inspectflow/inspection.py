"""OpenCV segmentation and interpretable, threshold-based classification."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import InspectionConfig


class PartInspector:
    """Find colored parts and classify their first failing inspection rule."""

    def __init__(self, config: InspectionConfig | None = None) -> None:
        self.config = InspectionConfig() if config is None else config
        if not isinstance(self.config, InspectionConfig):
            raise ValueError("config must be an InspectionConfig")

    def inspect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Inspect one BGR image and return JSON-serializable detections.

        Classification precedence is dimensions, then color, then circular
        feature. This gives every rejected part one stable primary defect label.
        """

        self._validate_image(image)
        mask = self._segment(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_part_area or area > self.config.max_part_area:
                continue
            x, y, width, height = (int(value) for value in cv2.boundingRect(contour))
            margin = self.config.border_margin_px
            image_height, image_width = image.shape[:2]
            if (
                x <= margin
                or y <= margin
                or x + width >= image_width - margin
                or y + height >= image_height - margin
            ):
                continue

            contour_mask = np.zeros((height, width), dtype=np.uint8)
            shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(contour_mask, [shifted], -1, 255, -1)

            roi = image[y : y + height, x : x + width]
            feature_radius, feature_center = self._measure_feature(roi, contour_mask)
            color_bgr = self._measure_color(
                roi, contour_mask, feature_center, feature_radius
            )
            color_distance = float(
                np.linalg.norm(
                    np.asarray(color_bgr, dtype=np.float32)
                    - np.asarray(self.config.expected_color_bgr, dtype=np.float32)
                )
            )
            predicted_defect = self._classify(
                width=width,
                height=height,
                color_distance=color_distance,
                feature_radius=feature_radius,
            )

            moments = cv2.moments(contour)
            if moments["m00"]:
                centroid_x = moments["m10"] / moments["m00"]
                centroid_y = moments["m01"] / moments["m00"]
            else:
                centroid_x = x + width / 2.0
                centroid_y = y + height / 2.0

            measurements: dict[str, Any] = {
                "width_px": width,
                "height_px": height,
                "contour_area_px": round(area, 3),
                "color_bgr": [round(float(value), 3) for value in color_bgr],
                "color_distance": round(color_distance, 3),
                "feature_radius_px": round(float(feature_radius), 3),
            }
            if feature_center is not None:
                measurements["feature_center"] = [
                    round(x + feature_center[0], 3),
                    round(y + feature_center[1], 3),
                ]

            detections.append(
                {
                    "bbox": [x, y, width, height],
                    "centroid": [round(centroid_x, 3), round(centroid_y, 3)],
                    "predicted_defect": predicted_defect,
                    "verdict": "pass" if predicted_defect == "normal" else "fail",
                    "measurements": measurements,
                }
            )

        # Stable ordering keeps API payloads and exported reports deterministic.
        detections.sort(
            key=lambda detection: (
                detection["centroid"][0],
                detection["centroid"][1],
            )
        )
        return detections

    def _segment(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [
                0,
                self.config.segmentation_saturation_min,
                self.config.segmentation_value_min,
            ],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv, lower, np.array([179, 255, 255], dtype=np.uint8))
        size = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _measure_feature(
        self, roi: np.ndarray, part_mask: np.ndarray
    ) -> tuple[float, tuple[float, float] | None]:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # The hole is dark and comparatively unsaturated. Ignore a generous
        # perimeter band so rounded part edges cannot masquerade as a feature.
        dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 110, 100]))
        interior = cv2.erode(
            part_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        dark = cv2.bitwise_and(dark, interior)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_radius = 0.0
        best_center: tuple[float, float] | None = None
        roi_center = np.asarray([roi.shape[1] / 2.0, roi.shape[0] / 2.0])
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            distance_from_center = float(
                np.linalg.norm(np.asarray([cx, cy]) - roi_center)
            )
            if circularity < 0.55 or distance_from_center > min(roi.shape[:2]) * 0.3:
                continue
            if radius > best_radius:
                best_radius = float(radius)
                best_center = (float(cx), float(cy))
        return best_radius, best_center

    @staticmethod
    def _measure_color(
        roi: np.ndarray,
        part_mask: np.ndarray,
        feature_center: tuple[float, float] | None,
        feature_radius: float,
    ) -> tuple[float, float, float]:
        sample_mask = part_mask.copy()
        if feature_center is not None:
            cv2.circle(
                sample_mask,
                (round(feature_center[0]), round(feature_center[1])),
                max(1, round(feature_radius + 3)),
                0,
                -1,
            )
        # Erosion removes anti-aliased border pixels and conveyor bleed.
        sample_mask = cv2.erode(
            sample_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        mean = cv2.mean(roi, mask=sample_mask)[:3]
        return float(mean[0]), float(mean[1]), float(mean[2])

    def _classify(
        self, *, width: int, height: int, color_distance: float, feature_radius: float
    ) -> str:
        width_error = (
            abs(width - self.config.expected_width_px)
            / self.config.expected_width_px
        )
        height_error = (
            abs(height - self.config.expected_height_px)
            / self.config.expected_height_px
        )
        if (
            width_error > self.config.dimension_tolerance
            or height_error > self.config.dimension_tolerance
        ):
            return "incorrect_dimensions"
        if color_distance > self.config.color_distance_threshold:
            return "wrong_color"
        if not (
            self.config.feature_min_radius
            <= feature_radius
            <= self.config.feature_max_radius
        ):
            return "missing_feature"
        return "normal"

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a NumPy array")
        if image.dtype != np.uint8:
            raise ValueError("image must have uint8 pixels")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (height, width, 3) in BGR order")
        if image.shape[0] < 3 or image.shape[1] < 3:
            raise ValueError("image is too small to inspect")
