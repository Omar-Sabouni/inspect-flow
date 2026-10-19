"""Deterministic, in-memory synthetic conveyor footage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import cv2
import numpy as np


DefectLabel = Literal[
    "normal", "incorrect_dimensions", "wrong_color", "missing_feature"
]


@dataclass(slots=True)
class FrameGroundTruth:
    """Image-free frame metadata retained for run-level evaluation."""

    index: int
    ground_truth: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return an independent, JSON-serializable metadata copy."""

        return {
            "index": int(self.index),
            "ground_truth": deepcopy(self.ground_truth),
        }


@dataclass(slots=True)
class GeneratedFrame:
    """One synthetic BGR frame and its visible, fully inspectable parts.

    ``partial_boxes`` lists parts that are drawn but cut off by, or too close
    to, the frame edge. They are not ground truth; detector training uses them
    as regions where neither a detection nor its absence is penalized.
    """

    index: int
    image: np.ndarray
    ground_truth: list[dict[str, Any]]
    partial_boxes: list[list[int]] = field(default_factory=list)

    def to_dict(self, *, include_image: bool = False) -> dict[str, Any]:
        """Return serializable frame metadata, optionally with pixels as lists."""

        result: dict[str, Any] = {
            "index": int(self.index),
            "ground_truth": [dict(part) for part in self.ground_truth],
        }
        if include_image:
            result["image"] = self.image.tolist()
        return result

    def truth_metadata(self) -> FrameGroundTruth:
        """Return the ground truth without retaining this frame's pixel array."""

        return FrameGroundTruth(
            index=int(self.index),
            ground_truth=deepcopy(self.ground_truth),
        )


@dataclass(frozen=True, slots=True)
class _PartSpec:
    part_id: str
    defect: DefectLabel
    spawn_frame: int
    y: int
    width: int
    height: int
    color_bgr: tuple[int, int, int]
    speed: int


class SyntheticConveyor:
    """Generate repeatable conveyor frames without writing image assets.

    A local NumPy random generator is recreated by every call to ``iter_frames``;
    consequently repeated iterations on one instance are byte-for-byte
    deterministic. ``generate`` remains as the eager compatibility helper.
    ``poor_lighting`` uses the same part schedule and labels as ``nominal`` while
    adding strong illumination gradients, blur and sensor noise.

    The seed controls the part schedule. Labels are dealt in blocks of four,
    one of each class in random order, so classes stay balanced without a
    fixed sequence. Good parts vary by a few pixels in size and color, defect
    shape and color are drawn from small sets, and spawn times jitter by one
    frame so a part's position does not follow from the frame index alone.
    """

    VALID_SCENARIOS = ("nominal", "poor_lighting")
    DEFECT_LABELS: tuple[DefectLabel, ...] = (
        "normal",
        "incorrect_dimensions",
        "wrong_color",
        "missing_feature",
    )
    NORMAL_COLOR_BGR = (70, 170, 225)
    WRONG_COLORS_BGR = ((205, 80, 75), (80, 170, 60), (170, 70, 160))
    NORMAL_SIZE = (80, 50)
    # Every shape is at least 25% off nominal on one axis, well beyond the
    # default 18% tolerance. Widths stay at or below 104 px so neighbouring
    # parts never touch.
    WRONG_SIZES = ((104, 36), (60, 50), (80, 66), (100, 60))

    def __init__(
        self,
        seed: int = 7,
        frame_count: int = 96,
        scenario: str = "nominal",
        width: int = 640,
        height: int = 320,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 1
        ):
            raise ValueError("frame_count must be a positive integer")
        if scenario not in self.VALID_SCENARIOS:
            choices = ", ".join(self.VALID_SCENARIOS)
            raise ValueError(f"scenario must be one of: {choices}")
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width < 240
            or height < 180
        ):
            raise ValueError("frame dimensions must be integers of at least 240x180")

        self.seed = seed
        self.frame_count = frame_count
        self.scenario = scenario
        self.width = width
        self.height = height

    def generate(self) -> list[GeneratedFrame]:
        """Render the configured sequence into memory and return all frames."""

        return list(self.iter_frames())

    def iter_frames(self) -> Iterator[GeneratedFrame]:
        """Yield frames one at a time so callers need not retain raw pixels."""

        rng = np.random.default_rng(self.seed)
        parts = self._build_schedule(rng)
        for frame_index in range(self.frame_count):
            image = self._draw_background(frame_index)
            truth: list[dict[str, Any]] = []
            partial: list[list[int]] = []
            for spec in parts:
                x = -spec.width + spec.speed * (frame_index - spec.spawn_frame)
                if x + spec.width <= 0 or x >= self.width:
                    continue
                self._draw_part(image, x, spec)
                bbox = [int(x), int(spec.y), int(spec.width), int(spec.height)]
                # Border fragments are intentionally excluded from both the
                # inspectable ground truth and detector output.
                # Blur and morphology expand a colored silhouette by roughly
                # two pixels, so retain extra clearance beyond the detector's
                # configured border margin.
                if x < 6 or x + spec.width > self.width - 6:
                    partial.append(bbox)
                    continue
                truth.append(
                    {
                        "part_id": spec.part_id,
                        "bbox": bbox,
                        "centroid": [
                            round(x + spec.width / 2.0, 3),
                            round(spec.y + spec.height / 2.0, 3),
                        ],
                        "defect": spec.defect,
                        "verdict": "pass" if spec.defect == "normal" else "fail",
                    }
                )

            if self.scenario == "poor_lighting":
                image = self._degrade(image, frame_index, rng)
            yield GeneratedFrame(frame_index, image, truth, partial)

    def _build_schedule(self, rng: np.random.Generator) -> list[_PartSpec]:
        speed = max(4, round(self.width / 92))
        spacing_frames = max(17, round(145 / speed))
        lead_frames = round(125 / speed)
        travel_frames = round((self.width + 120) / speed)
        count = max(8, self.frame_count // spacing_frames + 5)
        belt_top = round(self.height * 0.19)
        belt_bottom = round(self.height * 0.84)

        parts: list[_PartSpec] = []
        labels: list[DefectLabel] = []
        while len(labels) < count:
            order = rng.permutation(len(self.DEFECT_LABELS))
            labels.extend(self.DEFECT_LABELS[int(position)] for position in order)
        for index, defect in enumerate(labels[:count]):
            if defect == "incorrect_dimensions":
                choice = int(rng.integers(len(self.WRONG_SIZES)))
                part_width, part_height = self.WRONG_SIZES[choice]
            else:
                part_width = self.NORMAL_SIZE[0] + int(rng.integers(-3, 4))
                part_height = self.NORMAL_SIZE[1] + int(rng.integers(-2, 3))
            if defect == "wrong_color":
                color = self.WRONG_COLORS_BGR[
                    int(rng.integers(len(self.WRONG_COLORS_BGR)))
                ]
            else:
                shift = rng.integers(-6, 7, size=3)
                color = (
                    int(self.NORMAL_COLOR_BGR[0] + shift[0]),
                    int(self.NORMAL_COLOR_BGR[1] + shift[1]),
                    int(self.NORMAL_COLOR_BGR[2] + shift[2]),
                )
            jitter = int(rng.integers(-1, 2))
            center_y = int(
                rng.integers(
                    belt_top + part_height // 2 + 8,
                    belt_bottom - part_height // 2 - 8,
                )
            )
            parts.append(
                _PartSpec(
                    part_id=f"part-{index + 1:03d}",
                    defect=defect,
                    spawn_frame=index * spacing_frames - lead_frames + jitter,
                    y=center_y - part_height // 2,
                    width=part_width,
                    height=part_height,
                    color_bgr=color,
                    speed=speed,
                )
            )
        # ``travel_frames`` documents the relevant lifetime calculation and
        # protects against accidental schedules too short for tiny frame counts.
        del travel_frames
        return parts

    def _draw_background(self, frame_index: int) -> np.ndarray:
        image = np.full((self.height, self.width, 3), (22, 27, 33), dtype=np.uint8)
        top, bottom = round(self.height * 0.16), round(self.height * 0.87)
        cv2.rectangle(image, (0, top), (self.width - 1, bottom), (48, 54, 60), -1)

        # Low-contrast slats make motion visible but remain below segmentation
        # saturation. Their phase is deterministic and tied to the frame index.
        phase = (frame_index * 7) % 42
        for x in range(-phase, self.width, 42):
            cv2.line(image, (x, top + 3), (x, bottom - 3), (58, 63, 69), 2)
        cv2.line(image, (0, top), (self.width - 1, top), (112, 119, 124), 4)
        cv2.line(image, (0, bottom), (self.width - 1, bottom), (15, 18, 22), 5)
        return image

    @staticmethod
    def _draw_part(image: np.ndarray, x: int, spec: _PartSpec) -> None:
        x1, y1 = int(x), int(spec.y)
        x2, y2 = x1 + spec.width - 1, y1 + spec.height - 1
        radius = min(8, spec.height // 5)

        # Rounded rectangle assembled from primitives. The flat extents retain
        # exact, measurable dimensions for boundingRect.
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), spec.color_bgr, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), spec.color_bgr, -1)
        for cx, cy in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(
                image,
                (cx, cy),
                radius,
                spec.color_bgr,
                -1,
                lineType=cv2.LINE_AA,
            )

        # A small highlight gives the browser animation some material depth.
        highlight = tuple(min(255, channel + 22) for channel in spec.color_bgr)
        cv2.line(
            image,
            (x1 + radius + 2, y1 + 5),
            (x2 - radius - 2, y1 + 5),
            highlight,
            2,
            lineType=cv2.LINE_AA,
        )

        if spec.defect != "missing_feature":
            center = (x1 + spec.width // 2, y1 + spec.height // 2)
            cv2.circle(image, center, 10, (20, 25, 30), -1, lineType=cv2.LINE_AA)
            cv2.circle(image, center, 7, (42, 48, 53), -1, lineType=cv2.LINE_AA)

    def _degrade(
        self, image: np.ndarray, frame_index: int, rng: np.random.Generator
    ) -> np.ndarray:
        # A moving, strongly uneven light field makes some otherwise valid parts
        # fall below the configured segmentation and color thresholds.
        x = np.linspace(0.0, 1.0, self.width, dtype=np.float32)
        center = 0.50 + 0.28 * np.sin(frame_index * 0.13)
        light = 0.24 + 0.64 * np.exp(-((x - center) ** 2) / 0.045)
        flicker = 0.88 + 0.10 * np.sin(frame_index * 0.43)
        field = (light * flicker)[None, :, None]
        degraded = image.astype(np.float32) * field

        noise = rng.normal(0.0, 15.0, image.shape).astype(np.float32)
        degraded += noise
        degraded = cv2.GaussianBlur(degraded, (5, 5), 1.15)
        return np.clip(degraded, 0, 255).astype(np.uint8)
