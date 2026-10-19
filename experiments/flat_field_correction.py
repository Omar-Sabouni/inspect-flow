#!/usr/bin/env python3
"""Check whether flat-field correction rescues the classical rules in poor light.

The light field is estimated with a large median filter on a quarter-size
brightness image. Parts are small next to the kernel, so the median follows the
belt. Each pixel is then scaled so the belt returns to its nominal brightness,
and the unchanged classical pipeline runs on the corrected frame. A third
variant applies a bilateral filter first to see whether denoising helps.

Runs on validation seeds 100-102 only, so the test seeds stay untouched.

    python experiments/flat_field_correction.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspectflow.evaluation import evaluate_run  # noqa: E402
from inspectflow.generator import SyntheticConveyor  # noqa: E402
from inspectflow.inspection import PartInspector  # noqa: E402

NOMINAL_BELT_BRIGHTNESS = 60.0
SEEDS = (100, 101, 102)


def flat_field(
    image: np.ndarray, kernel_px: int = 101, bilateral_sigma: float = 0.0
) -> np.ndarray:
    if bilateral_sigma > 0:
        image = cv2.bilateralFilter(image, 7, bilateral_sigma, 7)
    small = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    brightness = small.max(axis=2)
    background = cv2.medianBlur(brightness, (kernel_px // 4) | 1).astype(np.float32)
    background = cv2.resize(
        background, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    gain = NOMINAL_BELT_BRIGHTNESS / np.maximum(background, 4.0)
    corrected = image.astype(np.float32) * gain[:, :, None]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def score(scenario: str, mode: str) -> tuple[float, float, float]:
    recalls, accuracies, milliseconds = [], [], []
    for seed in SEEDS:
        inspector = PartInspector()
        truth, detections = [], []
        elapsed = 0.0
        for frame in SyntheticConveyor(seed, 240, scenario).iter_frames():
            started = time.perf_counter()
            if mode == "none":
                image = frame.image
            else:
                image = flat_field(
                    frame.image, bilateral_sigma=30.0 if mode == "bilateral" else 0.0
                )
            found = inspector.inspect(image)
            elapsed += time.perf_counter() - started
            detections.append(found)
            truth.append(frame.ground_truth)
        report = evaluate_run(truth, detections)
        recalls.append(report["detection"]["recall"])
        accuracies.append(report["defect_accuracy"])
        milliseconds.append(1000.0 * elapsed / len(truth))
    return float(np.mean(recalls)), float(np.mean(accuracies)), float(np.mean(milliseconds))


MODES = {
    "none": "None",
    "flat_field": "Flat-field",
    "bilateral": "Bilateral filter, then flat-field",
}


def belt_noise(scenario: str) -> tuple[float, float]:
    """Noise std on an empty belt patch, before and after correction."""

    frames = SyntheticConveyor(SEEDS[0], 40, scenario)
    before, after = [], []
    for frame in frames.iter_frames():
        # Rows 56-66 are empty belt above every part. The light field and the
        # belt slats both vary only along x, so the spread down one column is
        # sensor noise. Average it over the 40 darkest columns, where the
        # correction gain is largest.
        patch = frame.image[56:66, :, 2].astype(np.float32)
        corrected = flat_field(frame.image)[56:66, :, 2].astype(np.float32)
        darkest = np.argsort(patch.mean(axis=0))[:40]
        before.append(float(patch[:, darkest].std(axis=0).mean()))
        after.append(float(corrected[:, darkest].std(axis=0).mean()))
    return float(np.mean(before)), float(np.mean(after))


def main() -> None:
    print("| Scenario | Correction | Detection recall | Defect class accuracy | ms/frame |")
    print("| --- | --- | ---: | ---: | ---: |")
    for scenario in ("nominal", "poor_lighting"):
        for mode, label in MODES.items():
            recall, accuracy, ms = score(scenario, mode)
            print(
                f"| {scenario.replace('_', ' ')} | {label} | "
                f"{recall * 100:.1f}% | {accuracy * 100:.1f}% | {ms:.2f} |"
            )
    before, after = belt_noise("poor_lighting")
    print(
        f"\nPoor lighting, darkest belt patch: noise std {before:.1f} before "
        f"correction, {after:.1f} after."
    )


if __name__ == "__main__":
    main()
