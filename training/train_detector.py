#!/usr/bin/env python3
"""Train the CNN part detector on synthetic frames and export it to ONNX.

Frames come from nominal scenes (seeds 1000 and up) with random gain, light
gradients, vignetting, noise and blur. None of that models the moving
spotlight of the poor-light test scene. Validation uses seeds 100-104; the
README's numbers use seeds 7-26.

    pip install -e '.[train]'
    python training/train_detector.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspectflow.detectors import (  # noqa: E402
    INPUT_SIZE,
    MODEL_PATH,
    SIZE_SCALE_PX,
    STRIDE,
    CnnPartDetector,
    model_sha256,
)
from inspectflow.evaluation import evaluate_run  # noqa: E402
from inspectflow.generator import DEFECTS, SCENARIOS, SyntheticConveyor  # noqa: E402

TRAIN_SEED_START = 1000
VALIDATION_SEEDS = (100, 101, 102, 103, 104)
SCENE_FRAMES = 240
COLUMNS, ROWS = INPUT_SIZE[0] // STRIDE, INPUT_SIZE[1] // STRIDE


def conv(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def head(width: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(width, outputs, 1))


class PartNet(nn.Module):
    """Stride-8 detector. Dilations 1-8 give each cell a ~280 px view."""

    def __init__(self, width: int = 48) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            conv(3, 16, stride=2),
            conv(16, 24, stride=2),
            conv(24, 24),
            conv(24, width, stride=2),
            conv(width, width),
            conv(width, width, dilation=2),
            conv(width, width, dilation=4),
            conv(width, width, dilation=8),
        )
        self.heatmap = head(width, len(DEFECTS))
        self.box = head(width, 4)
        # CenterNet's low initial prior keeps the focal loss stable early on.
        nn.init.constant_(self.heatmap[-1].bias, -2.19)  # type: ignore[arg-type]

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image)
        return self.heatmap(features), self.box(features)


class Exported(nn.Module):
    """One output tensor with activations applied, the layout ``decode`` reads."""

    def __init__(self, model: PartNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        heatmap, box = self.model(image)
        return torch.cat([torch.sigmoid(heatmap), torch.sigmoid(box[:, :2]), box[:, 2:]], dim=1)


def damage(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random lighting and sensor damage; deliberately no spotlight."""

    if rng.random() < 0.25:
        return image
    height, width = image.shape[:2]
    frame = image.astype(np.float32)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    xs /= width
    ys /= height
    light = np.full((height, width), rng.uniform(0.35, 1.2), dtype=np.float32)
    choice = rng.random()
    if choice < 0.45:  # linear gradient in a random direction
        angle = rng.uniform(0.0, 2.0 * math.pi)
        ramp = (xs - 0.5) * math.cos(angle) + (ys - 0.5) * math.sin(angle)
        start, end = rng.uniform(0.25, 1.15, size=2)
        light *= start + (end - start) * (ramp + 0.71) / 1.42
    elif choice < 0.75:  # vignette
        light *= 1.0 - rng.uniform(0.2, 0.8) * ((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / 0.5
    frame *= np.clip(light, 0.1, 1.5)[:, :, None]
    # Only small white-balance shifts: big ones would make a good part look
    # like the wrong color.
    frame *= rng.uniform(0.92, 1.08, size=3).astype(np.float32)
    sigma = rng.uniform(0.0, 20.0)
    if sigma > 0.5:
        frame += rng.normal(0.0, sigma, frame.shape).astype(np.float32)
    blur = rng.uniform(0.0, 1.6)
    if blur > 0.3:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    return np.clip(frame, 0, 255).astype(np.uint8)


def targets(parts: list[dict[str, Any]], partial_boxes: list[list[int]]) -> dict[str, np.ndarray]:
    """Gaussian center heatmaps per class, box targets at each center cell,
    and a mask that zeroes the loss around parts cut off by the frame edge."""

    heatmap = np.zeros((len(DEFECTS), ROWS, COLUMNS), np.float32)
    ignore = np.ones((ROWS, COLUMNS), np.float32)
    box = np.zeros((4, ROWS, COLUMNS), np.float32)
    box_mask = np.zeros((ROWS, COLUMNS), np.float32)
    grid_y, grid_x = np.mgrid[0:ROWS, 0:COLUMNS].astype(np.float32)

    for x, y, w, h in partial_boxes:
        ignore[max(0, y // STRIDE - 1) : (y + h) // STRIDE + 2, max(0, x // STRIDE - 1) : (x + w) // STRIDE + 2] = 0
    for part in parts:
        x, y, w, h = (float(v) for v in part["bbox"])
        cx, cy = (x + w / 2) / STRIDE, (y + h / 2) / STRIDE
        col, row = min(COLUMNS - 1, int(cx)), min(ROWS - 1, int(cy))
        label = DEFECTS.index(part["defect"])
        heatmap[label] = np.maximum(heatmap[label], np.exp(-((grid_x - col) ** 2 + (grid_y - row) ** 2) / 2))
        box[:, row, col] = (cx - col, cy - row, math.log(w / SIZE_SCALE_PX), math.log(h / SIZE_SCALE_PX))
        box_mask[row, col] = 1
    return {"heatmap": heatmap, "ignore": ignore, "box": box, "box_mask": box_mask}


class Frames(IterableDataset):
    """Endless stream of damaged frames. Each worker shuffles through a buffer
    of uint8 frames so a batch mixes scenes."""

    def __init__(self, seed: int, buffer_size: int = 128) -> None:
        self.seed = seed
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng(self.seed * 7919 + worker)
        scene = TRAIN_SEED_START + worker
        buffer: list[dict[str, np.ndarray]] = []
        while True:
            for frame in SyntheticConveyor(scene, SCENE_FRAMES, "nominal").iter_frames():
                sample = {"image": damage(frame.image, rng), **targets(frame.ground_truth, frame.partial_boxes)}
                if len(buffer) < self.buffer_size:
                    buffer.append(sample)
                    continue
                slot = int(rng.integers(len(buffer)))
                yield buffer[slot]
                buffer[slot] = sample
            scene += workers


def focal_loss(logits: torch.Tensor, target: torch.Tensor, ignore: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float()
    loss = torch.log(p) * (1 - p) ** 2 * positive + torch.log(1 - p) * p**2 * (1 - target) ** 4 * (1 - positive)
    return -(loss * ignore.unsqueeze(1)).sum() / positive.sum().clamp(min=1)


def box_losses(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight, count = mask.unsqueeze(1), mask.sum().clamp(min=1)
    offset = ((torch.sigmoid(prediction[:, :2]) - target[:, :2]).abs() * weight).sum() / count
    size = ((prediction[:, 2:] - target[:, 2:]).abs() * weight).sum() / count
    return offset, size


def validate(model_path: Path, frame_count: int) -> dict[str, dict[str, float]]:
    results = {}
    for scenario in SCENARIOS:
        detector = CnnPartDetector(model_path)
        truth, detections = [], []
        for seed in VALIDATION_SEEDS:
            for frame in SyntheticConveyor(seed, frame_count, scenario).iter_frames():
                truth.append(frame.ground_truth)
                detections.append(detector.inspect(frame.image))
        report = evaluate_run(truth, detections)
        results[scenario] = {**report["detection"], "defect_accuracy": report["defect_accuracy"]}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=6, help="each holds ~80 MB of frames")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = PartNet().to(device)
    loader = DataLoader(
        Frames(args.seed), batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, total_steps=args.steps, pct_start=0.1)

    started = time.monotonic()
    model.train()
    for step, batch in enumerate(loader, start=1):
        image = batch["image"].to(device).permute(0, 3, 1, 2).float() / 255
        heatmap, box = model(image)
        heatmap_loss = focal_loss(heatmap, batch["heatmap"].to(device), batch["ignore"].to(device))
        offset_loss, size_loss = box_losses(box, batch["box"].to(device), batch["box_mask"].to(device))
        loss = heatmap_loss + offset_loss + 0.5 * size_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 250 == 0:
            print(f"step {step}/{args.steps}  loss {loss.item():.4f}  {time.monotonic() - started:.0f}s", flush=True)
        if step == args.steps:
            break

    model.eval().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Exported(model),
        (torch.zeros(1, 3, INPUT_SIZE[1], INPUT_SIZE[0]),),
        str(args.output),
        input_names=["image"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,  # the TorchScript exporter at opset 17 is what OpenCV's DNN reads
    )
    record = {
        "model": "PartNet",
        "sha256": model_sha256(args.output),
        "parameters": sum(p.numel() for p in model.parameters()),
        "input": {"width": INPUT_SIZE[0], "height": INPUT_SIZE[1], "channels": "BGR", "scale": "1/255"},
        "output_stride": STRIDE,
        "labels": list(DEFECTS),
        "training": {
            **{k: v for k, v in vars(args).items() if k != "output"},
            "duration_s": round(time.monotonic() - started),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "validation": validate(args.output, SCENE_FRAMES),
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["validation"], indent=2))


if __name__ == "__main__":
    main()
