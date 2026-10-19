"""Synthetic conveyor footage with known labels."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

DEFECTS = ("normal", "incorrect_dimensions", "wrong_color", "missing_feature")
SCENARIOS = ("nominal", "poor_lighting")

GOOD_COLOR = (70, 170, 225)
GOOD_SIZE = (80, 50)
WRONG_COLORS = ((205, 80, 75), (80, 170, 60), (170, 70, 160))
# Each wrong size is at least 25% off on one axis, well past the 18% tolerance.
WRONG_SIZES = ((104, 36), (60, 50), (80, 66), (100, 60))
# Parts closer than this to the frame edge are drawn but not labelled.
EDGE_CLEARANCE_PX = 6


@dataclass
class Frame:
    index: int
    image: np.ndarray
    ground_truth: list[dict[str, Any]]
    # Parts cut off by the frame edge. Not labelled; CNN training ignores them.
    partial_boxes: list[list[int]] = field(default_factory=list)


@dataclass
class _Part:
    part_id: str
    defect: str
    spawn_frame: int
    y: int
    width: int
    height: int
    color: tuple[int, ...]
    speed: int


class SyntheticConveyor:
    """Parts moving left to right on a belt, reproducible from a seed.

    Labels are dealt in shuffled blocks of four, so classes stay balanced and
    no class is tied to a position. Both scenarios share the same parts; poor
    lighting adds a moving spotlight, flicker, sensor noise and blur.
    """

    def __init__(
        self,
        seed: int = 7,
        frame_count: int = 96,
        scenario: str = "nominal",
        width: int = 640,
        height: int = 320,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        self.seed = seed
        self.frame_count = frame_count
        self.scenario = scenario
        self.width = width
        self.height = height

    def iter_frames(self) -> Iterator[Frame]:
        rng = np.random.default_rng(self.seed)
        parts = self._schedule(rng)
        for index in range(self.frame_count):
            image = self._background(index)
            truth: list[dict[str, Any]] = []
            partial: list[list[int]] = []
            for part in parts:
                x = -part.width + part.speed * (index - part.spawn_frame)
                if x + part.width <= 0 or x >= self.width:
                    continue
                self._draw_part(image, x, part)
                bbox = [x, part.y, part.width, part.height]
                if x < EDGE_CLEARANCE_PX or x + part.width > self.width - EDGE_CLEARANCE_PX:
                    partial.append(bbox)
                    continue
                truth.append(
                    {
                        "part_id": part.part_id,
                        "bbox": bbox,
                        "centroid": [x + part.width / 2, part.y + part.height / 2],
                        "defect": part.defect,
                    }
                )
            if self.scenario == "poor_lighting":
                image = self._poor_light(image, index, rng)
            yield Frame(index, image, truth, partial)

    def _schedule(self, rng: np.random.Generator) -> list[_Part]:
        speed = max(4, round(self.width / 92))
        spacing = max(17, round(145 / speed))
        lead = round(125 / speed)
        count = max(8, self.frame_count // spacing + 5)
        belt_top, belt_bottom = round(self.height * 0.19), round(self.height * 0.84)

        labels: list[str] = []
        while len(labels) < count:
            labels.extend(DEFECTS[int(i)] for i in rng.permutation(len(DEFECTS)))

        parts = []
        for number, defect in enumerate(labels[:count]):
            if defect == "incorrect_dimensions":
                width, height = WRONG_SIZES[int(rng.integers(len(WRONG_SIZES)))]
            else:
                width = GOOD_SIZE[0] + int(rng.integers(-3, 4))
                height = GOOD_SIZE[1] + int(rng.integers(-2, 3))
            color: tuple[int, ...]
            if defect == "wrong_color":
                color = WRONG_COLORS[int(rng.integers(len(WRONG_COLORS)))]
            else:
                shift = rng.integers(-6, 7, size=3)
                color = tuple(int(c + s) for c, s in zip(GOOD_COLOR, shift, strict=True))
            jitter = int(rng.integers(-1, 2))
            center_y = int(
                rng.integers(belt_top + height // 2 + 8, belt_bottom - height // 2 - 8)
            )
            parts.append(
                _Part(
                    part_id=f"part-{number + 1:03d}",
                    defect=defect,
                    spawn_frame=number * spacing - lead + jitter,
                    y=center_y - height // 2,
                    width=width,
                    height=height,
                    color=color,
                    speed=speed,
                )
            )
        return parts

    def _background(self, index: int) -> np.ndarray:
        image = np.full((self.height, self.width, 3), (22, 27, 33), dtype=np.uint8)
        top, bottom = round(self.height * 0.16), round(self.height * 0.87)
        cv2.rectangle(image, (0, top), (self.width - 1, bottom), (48, 54, 60), -1)
        # Faint slats show the belt moving; they stay below the mask thresholds.
        phase = (index * 7) % 42
        for x in range(-phase, self.width, 42):
            cv2.line(image, (x, top + 3), (x, bottom - 3), (58, 63, 69), 2)
        cv2.line(image, (0, top), (self.width - 1, top), (112, 119, 124), 4)
        cv2.line(image, (0, bottom), (self.width - 1, bottom), (15, 18, 22), 5)
        return image

    @staticmethod
    def _draw_part(image: np.ndarray, x: int, part: _Part) -> None:
        x1, y1 = x, part.y
        x2, y2 = x1 + part.width - 1, y1 + part.height - 1
        r = min(8, part.height // 5)
        # Rounded rectangle whose straight edges keep the exact width and height.
        cv2.rectangle(image, (x1 + r, y1), (x2 - r, y2), part.color, -1)
        cv2.rectangle(image, (x1, y1 + r), (x2, y2 - r), part.color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(image, (cx, cy), r, part.color, -1, lineType=cv2.LINE_AA)
        highlight = tuple(min(255, c + 22) for c in part.color)
        cv2.line(image, (x1 + r + 2, y1 + 5), (x2 - r - 2, y1 + 5), highlight, 2, lineType=cv2.LINE_AA)
        if part.defect != "missing_feature":
            center = (x1 + part.width // 2, y1 + part.height // 2)
            cv2.circle(image, center, 10, (20, 25, 30), -1, lineType=cv2.LINE_AA)
            cv2.circle(image, center, 7, (42, 48, 53), -1, lineType=cv2.LINE_AA)

    def _poor_light(self, image: np.ndarray, index: int, rng: np.random.Generator) -> np.ndarray:
        x = np.linspace(0.0, 1.0, self.width, dtype=np.float32)
        center = 0.50 + 0.28 * np.sin(index * 0.13)
        spotlight = 0.24 + 0.64 * np.exp(-((x - center) ** 2) / 0.045)
        flicker = 0.88 + 0.10 * np.sin(index * 0.43)
        lit = image.astype(np.float32) * (spotlight * flicker)[None, :, None]
        lit += rng.normal(0.0, 15.0, image.shape).astype(np.float32)
        lit = cv2.GaussianBlur(lit, (5, 5), 1.15)
        return np.clip(lit, 0, 255).astype(np.uint8)
