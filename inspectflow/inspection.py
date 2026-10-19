"""Classical inspection: HSV segmentation, then measured tolerances."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import InspectionConfig


class PartInspector:
    """Find colored parts and label each with its first failed check.

    Checks run in order: size, then color, then the dark center hole.
    """

    name = "classical"

    def __init__(self, config: InspectionConfig | None = None) -> None:
        self.config = config if config is not None else InspectionConfig()

    def inspect(self, image: np.ndarray) -> list[dict[str, Any]]:
        cfg = self.config
        height_px, width_px = image.shape[:2]
        contours, _ = cv2.findContours(self._mask(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not cfg.min_part_area <= area <= cfg.max_part_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            m = cfg.border_margin_px
            if x <= m or y <= m or x + w >= width_px - m or y + h >= height_px - m:
                continue

            part_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(part_mask, [contour - np.array([x, y], contour.dtype)], -1, 255, -1)
            roi = image[y : y + h, x : x + w]
            hole_radius, hole_center = self._find_hole(roi, part_mask)
            color = self._mean_color(roi, part_mask, hole_center, hole_radius)
            color_distance = float(np.linalg.norm(np.subtract(color, cfg.expected_color_bgr)))
            defect = self._classify(w, h, color_distance, hole_radius)

            moments = cv2.moments(contour)
            centroid = (
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
                if moments["m00"]
                else [x + w / 2, y + h / 2]
            )
            detections.append(
                {
                    "bbox": [x, y, w, h],
                    "centroid": [round(c, 3) for c in centroid],
                    "predicted_defect": defect,
                    "verdict": "pass" if defect == "normal" else "fail",
                    "measurements": {
                        "width_px": w,
                        "height_px": h,
                        "contour_area_px": round(area, 3),
                        "color_distance": round(color_distance, 3),
                        "feature_radius_px": round(hole_radius, 3),
                    },
                }
            )
        detections.sort(key=lambda d: d["centroid"])
        return detections

    def _mask(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config
        hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
        lower = np.array([0, cfg.segmentation_saturation_min, cfg.segmentation_value_min], np.uint8)
        mask = cv2.inRange(hsv, lower, np.array([179, 255, 255], np.uint8))
        size = cfg.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    @staticmethod
    def _find_hole(roi: np.ndarray, part_mask: np.ndarray) -> tuple[float, tuple[float, float] | None]:
        """Radius and center of the largest dark, round blob near the middle."""

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 110, 100]))
        # Ignore a band around the edge so the rounded corners never count.
        interior = cv2.erode(part_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        dark = cv2.bitwise_and(dark, interior)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_radius, best_center = 0.0, None
        middle = np.array([roi.shape[1] / 2, roi.shape[0] / 2])
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < 30 or perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / perimeter**2
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            off_center = np.linalg.norm(np.array([cx, cy]) - middle)
            if circularity >= 0.55 and off_center <= min(roi.shape[:2]) * 0.3 and radius > best_radius:
                best_radius, best_center = float(radius), (float(cx), float(cy))
        return best_radius, best_center

    @staticmethod
    def _mean_color(
        roi: np.ndarray, part_mask: np.ndarray, hole_center: tuple[float, float] | None, hole_radius: float
    ) -> tuple[float, float, float]:
        sample = part_mask.copy()
        if hole_center is not None:
            center = (round(hole_center[0]), round(hole_center[1]))
            cv2.circle(sample, center, max(1, round(hole_radius + 3)), 0, -1)
        # Erode away anti-aliased edges and belt bleed before averaging.
        sample = cv2.erode(sample, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        b, g, r, _ = cv2.mean(roi, mask=sample)
        return b, g, r

    def _classify(self, width: int, height: int, color_distance: float, hole_radius: float) -> str:
        cfg = self.config
        width_error = abs(width - cfg.expected_width_px) / cfg.expected_width_px
        height_error = abs(height - cfg.expected_height_px) / cfg.expected_height_px
        if max(width_error, height_error) > cfg.dimension_tolerance:
            return "incorrect_dimensions"
        if color_distance > cfg.color_distance_threshold:
            return "wrong_color"
        if not cfg.feature_min_radius <= hole_radius <= cfg.feature_max_radius:
            return "missing_feature"
        return "normal"
