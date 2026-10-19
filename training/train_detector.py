#!/usr/bin/env python3
"""Train the CNN part detector on synthetic frames and export it to ONNX.

Training frames come from ``SyntheticConveyor`` in its nominal scenario and are
then degraded with random global gain, linear light gradients, vignetting,
sensor noise and blur. The poor-lighting scenario's moving spotlight is not one
of those degradations, so the poor-lighting score measures transfer to a light
pattern the network never saw.

Seeds are split three ways and the ranges must not overlap:

* training: 1000 and above
* validation (threshold tuning, model selection): 100-104
* test (reported in the README and benchmark): 7-11

Example::

    python -m pip install -e .[train]
    python training/train_detector.py --steps 6000 --device cuda

The script writes ``inspectflow/models/part_detector.onnx`` and a JSON training
record next to it. ``--evaluate-only`` skips training and re-scores the existing
model with the current decode defaults, which keeps the record in step after a
threshold or margin change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inspectflow.detectors import (  # noqa: E402
    DEFECT_LABELS,
    MODEL_INPUT_SIZE,
    OUTPUT_CHANNELS,
    OUTPUT_STRIDE,
    SIZE_SCALE_PX,
    CnnPartDetector,
)
from inspectflow.evaluation import evaluate_run  # noqa: E402
from inspectflow.generator import SyntheticConveyor  # noqa: E402
from inspectflow.tracking import CentroidTracker  # noqa: E402

TRAIN_SEED_START = 1000
VALIDATION_SEEDS = (100, 101, 102, 103, 104)
TEST_SEEDS = (7, 8, 9, 10, 11)
SEQUENCE_FRAMES = 240
HEATMAP_SIGMA_CELLS = 1.0
DEFAULT_OUTPUT = REPO_ROOT / "inspectflow" / "models" / "part_detector.onnx"


def conv_block(
    in_channels: int, out_channels: int, *, stride: int = 1, dilation: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class PartNet(nn.Module):
    """Stride-8 fully convolutional detector.

    Three stride-2 convolutions reduce resolution, then dilated convolutions
    widen the receptive field to about 280 px so a cell can see a whole part
    and judge its size. Parameter count is about 150k.
    """

    def __init__(self, width: int = 48) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            conv_block(3, 16, stride=2),
            conv_block(16, 24, stride=2),
            conv_block(24, 24),
            conv_block(24, width, stride=2),
            conv_block(width, width),
            conv_block(width, width, dilation=2),
            conv_block(width, width, dilation=4),
            conv_block(width, width, dilation=8),
        )
        self.heatmap = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, len(DEFECT_LABELS), 1),
        )
        self.box = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 4, 1),
        )
        # Start with a low prior on every heatmap cell, as in CenterNet. This
        # keeps the focal loss stable in the first iterations.
        final = self.heatmap[-1]
        assert isinstance(final, nn.Conv2d) and final.bias is not None
        nn.init.constant_(final.bias, -2.19)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image)
        return self.heatmap(features), self.box(features)


class ExportWrapper(nn.Module):
    """Single-output graph with activations applied, as ``decode_output`` expects."""

    def __init__(self, model: PartNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        heatmap_logits, box = self.model(image)
        return torch.cat(
            [
                torch.sigmoid(heatmap_logits),
                torch.sigmoid(box[:, :2]),
                box[:, 2:],
            ],
            dim=1,
        )


def degrade_for_training(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random photometric damage that excludes the test spotlight model."""

    if rng.random() < 0.25:
        return image
    height, width = image.shape[:2]
    frame = image.astype(np.float32)

    gain = rng.uniform(0.35, 1.2)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    xs /= width
    ys /= height
    field = np.full((height, width), gain, dtype=np.float32)

    choice = rng.random()
    if choice < 0.45:
        angle = rng.uniform(0.0, 2.0 * math.pi)
        ramp = (xs - 0.5) * math.cos(angle) + (ys - 0.5) * math.sin(angle)
        start, end = rng.uniform(0.25, 1.15, size=2)
        field *= start + (end - start) * (ramp + 0.71) / 1.42
    elif choice < 0.75:
        strength = rng.uniform(0.2, 0.8)
        radius = ((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / 0.5
        field *= 1.0 - strength * radius
    frame *= np.clip(field, 0.1, 1.5)[:, :, None]

    # Small white-balance shifts only. Large ones would turn the orange part
    # blue and contradict the wrong_color label.
    frame *= rng.uniform(0.92, 1.08, size=3).astype(np.float32)

    sigma = rng.uniform(0.0, 20.0)
    if sigma > 0.5:
        frame += rng.normal(0.0, sigma, frame.shape).astype(np.float32)
    blur = rng.uniform(0.0, 1.6)
    if blur > 0.3:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    return np.clip(frame, 0, 255).astype(np.uint8)


def build_targets(
    ground_truth: list[dict[str, Any]], partial_boxes: list[list[int]]
) -> dict[str, np.ndarray]:
    """Rasterize ground truth into heatmap and box-regression targets."""

    columns = MODEL_INPUT_SIZE[0] // OUTPUT_STRIDE
    rows = MODEL_INPUT_SIZE[1] // OUTPUT_STRIDE
    heatmap = np.zeros((len(DEFECT_LABELS), rows, columns), dtype=np.float32)
    ignore = np.ones((rows, columns), dtype=np.float32)
    box = np.zeros((4, rows, columns), dtype=np.float32)
    box_mask = np.zeros((rows, columns), dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:rows, 0:columns].astype(np.float32)

    for x, y, width, height in partial_boxes:
        left = max(0, int(x) // OUTPUT_STRIDE - 1)
        right = min(columns, (int(x) + int(width)) // OUTPUT_STRIDE + 2)
        top = max(0, int(y) // OUTPUT_STRIDE - 1)
        bottom = min(rows, (int(y) + int(height)) // OUTPUT_STRIDE + 2)
        ignore[top:bottom, left:right] = 0.0

    for item in ground_truth:
        x, y, width, height = (float(value) for value in item["bbox"])
        center_x = (x + width / 2.0) / OUTPUT_STRIDE
        center_y = (y + height / 2.0) / OUTPUT_STRIDE
        cell_x = min(columns - 1, int(center_x))
        cell_y = min(rows - 1, int(center_y))
        label = DEFECT_LABELS.index(item["defect"])
        gaussian = np.exp(
            -((grid_x - cell_x) ** 2 + (grid_y - cell_y) ** 2)
            / (2.0 * HEATMAP_SIGMA_CELLS**2)
        )
        heatmap[label] = np.maximum(heatmap[label], gaussian)
        box[:, cell_y, cell_x] = (
            center_x - cell_x,
            center_y - cell_y,
            math.log(width / SIZE_SCALE_PX),
            math.log(height / SIZE_SCALE_PX),
        )
        box_mask[cell_y, cell_x] = 1.0

    return {
        "heatmap": heatmap,
        "ignore": ignore,
        "box": box,
        "box_mask": box_mask,
    }


class SyntheticStream(IterableDataset):
    """Endless shuffled stream of degraded frames from unique training seeds.

    Each worker keeps a shuffle buffer so a batch mixes frames from several
    scenes. Frames stay uint8 in the buffer (0.6 MB each); converting them to
    float32 here would cost four times the memory per worker.
    """

    def __init__(self, base_seed: int, buffer_size: int = 128) -> None:
        super().__init__()
        self.base_seed = base_seed
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        info = get_worker_info()
        worker = 0 if info is None else info.id
        workers = 1 if info is None else info.num_workers
        rng = np.random.default_rng(self.base_seed * 7919 + worker)
        scene_seed = TRAIN_SEED_START + worker
        buffer: list[dict[str, np.ndarray]] = []
        while True:
            conveyor = SyntheticConveyor(
                seed=scene_seed,
                frame_count=SEQUENCE_FRAMES,
                scenario="nominal",
                width=MODEL_INPUT_SIZE[0],
                height=MODEL_INPUT_SIZE[1],
            )
            scene_seed += workers
            for frame in conveyor.iter_frames():
                image = degrade_for_training(frame.image, rng)
                sample = build_targets(frame.ground_truth, frame.partial_boxes)
                sample["image"] = image
                if len(buffer) < self.buffer_size:
                    buffer.append(sample)
                    continue
                slot = int(rng.integers(len(buffer)))
                yield buffer[slot]
                buffer[slot] = sample


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, ignore: torch.Tensor
) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss, masked by ``ignore``."""

    probability = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    positive = target.eq(1.0).float()
    negative_weight = torch.pow(1.0 - target, 4)
    mask = ignore.unsqueeze(1)
    positive_loss = torch.log(probability) * torch.pow(1.0 - probability, 2) * positive
    negative_loss = (
        torch.log(1.0 - probability)
        * torch.pow(probability, 2)
        * negative_weight
        * (1.0 - positive)
    )
    count = positive.sum().clamp(min=1.0)
    return -((positive_loss + negative_loss) * mask).sum() / count


def box_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = mask.unsqueeze(1)
    count = mask.sum().clamp(min=1.0)
    offset = (
        torch.abs(torch.sigmoid(prediction[:, :2]) - target[:, :2]) * weight
    ).sum() / count
    size = (torch.abs(prediction[:, 2:] - target[:, 2:]) * weight).sum() / count
    return offset, size


def evaluate_onnx(
    model_path: Path, seeds: tuple[int, ...], frame_count: int
) -> dict[str, dict[str, float]]:
    """Score an exported model with the production decode and tracker."""

    summary: dict[str, dict[str, float]] = {}
    for scenario in ("nominal", "poor_lighting"):
        recalls, precisions, accuracies = [], [], []
        for seed in seeds:
            detector = CnnPartDetector(model_path)
            tracker = CentroidTracker(max_distance=70.0, max_missed=4)
            truth, detections = [], []
            for frame in SyntheticConveyor(
                seed=seed, frame_count=frame_count, scenario=scenario
            ).iter_frames():
                detections.append(
                    tracker.update(detector.inspect(frame.image), frame.index)
                )
                truth.append(frame.truth_metadata())
            report = evaluate_run(truth, detections)
            recalls.append(report["detection"]["recall"])
            precisions.append(report["detection"]["precision"])
            accuracies.append(report["classification"]["accuracy"])
        summary[scenario] = {
            "detection_precision": round(float(np.mean(precisions)), 4),
            "detection_recall": round(float(np.mean(recalls)), 4),
            "verdict_accuracy": round(float(np.mean(accuracies)), 4),
        }
    return summary


def decode_settings() -> dict[str, Any]:
    detector = CnnPartDetector()
    return {
        "score_threshold": detector.score_threshold,
        "nms_iou": detector.nms_iou,
        "border_margin_px": detector.border_margin_px,
    }


def update_validation(model_path: Path, frame_count: int) -> None:
    card_path = model_path.with_suffix(".json")
    record = json.loads(card_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if record.get("sha256") != digest:
        raise SystemExit("model file does not match the hash in its training record")
    validation = evaluate_onnx(model_path, VALIDATION_SEEDS, frame_count)
    record["decode"] = decode_settings()
    record["validation"] = {"seeds": list(VALIDATION_SEEDS), **validation}
    card_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Validation (seeds 100-104):", json.dumps(validation, indent=2))
    print(f"Updated {card_path}")


def export_onnx(model: PartNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model).eval().cpu()
    example = torch.zeros(1, 3, MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0])
    # The TorchScript exporter at opset 17 produces a graph that OpenCV's DNN
    # importer reads. The newer dynamo exporter emits opset 18+ operators that
    # OpenCV 4.x does not fully support.
    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        input_names=["image"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    with torch.no_grad():
        reference = wrapper(example).numpy()
    if reference.shape[1] != OUTPUT_CHANNELS:
        raise RuntimeError("export wrapper produced the wrong channel count")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-frames", type=int, default=240)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="re-score the existing model and update its record without training",
    )
    args = parser.parse_args()

    if args.evaluate_only:
        update_validation(args.output, args.eval_frames)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = PartNet().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"PartNet parameters: {parameters:,}  device: {device}")

    loader = DataLoader(
        SyntheticStream(args.seed),
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=args.steps, pct_start=0.1
    )

    started = time.monotonic()
    running = {"heatmap": 0.0, "offset": 0.0, "size": 0.0}
    final_losses: dict[str, float] = {}
    model.train()
    for step, batch in enumerate(loader, start=1):
        image = (
            batch["image"].to(device, non_blocking=True).permute(0, 3, 1, 2).float()
            / 255.0
        )
        heatmap_logits, box = model(image)
        heatmap = focal_loss(
            heatmap_logits,
            batch["heatmap"].to(device),
            batch["ignore"].to(device),
        )
        offset, size = box_loss(
            box, batch["box"].to(device), batch["box_mask"].to(device)
        )
        loss = heatmap + offset + 0.5 * size
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()

        running["heatmap"] += heatmap.item()
        running["offset"] += offset.item()
        running["size"] += size.item()
        if step % args.log_every == 0 or step == args.steps:
            count = args.log_every if step % args.log_every == 0 else step % args.log_every
            final_losses = {key: round(value / count, 4) for key, value in running.items()}
            elapsed = time.monotonic() - started
            print(
                f"step {step:5d}/{args.steps}  "
                + "  ".join(f"{key} {value:.4f}" for key, value in final_losses.items())
                + f"  {elapsed:6.0f}s",
                flush=True,
            )
            running = {key: 0.0 for key in running}
        if step >= args.steps:
            break

    model.eval()
    export_onnx(model, args.output)
    print(f"Exported {args.output} ({args.output.stat().st_size / 1024:.0f} KiB)")

    validation = evaluate_onnx(args.output, VALIDATION_SEEDS, args.eval_frames)
    print("Validation (seeds 100-104):", json.dumps(validation, indent=2))

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    record = {
        "model": "PartNet",
        "format": "onnx",
        "opset": 17,
        "sha256": digest,
        "parameters": parameters,
        "input": {
            "width": MODEL_INPUT_SIZE[0],
            "height": MODEL_INPUT_SIZE[1],
            "channels": "BGR",
            "scale": "1/255",
        },
        "output_stride": OUTPUT_STRIDE,
        "labels": list(DEFECT_LABELS),
        "decode": decode_settings(),
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "scene_seeds": f">= {TRAIN_SEED_START}",
            "scenario": "nominal + random gain, gradients, vignetting, noise, blur",
            "final_losses": final_losses,
            "duration_s": round(time.monotonic() - started, 1),
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "validation": {"seeds": list(VALIDATION_SEEDS), **validation},
    }
    card = args.output.with_suffix(".json")
    card.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {card}")


if __name__ == "__main__":
    main()
