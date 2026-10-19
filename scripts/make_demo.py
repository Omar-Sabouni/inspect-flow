#!/usr/bin/env python3
"""Render the README demo: both detectors on the same poor-light scene.

The two runs are stacked so each part lines up vertically. Each panel header
keeps a running count of gate decisions scored against ground truth, and a
badge appears at the gate line when a part is decided or slips past.

Needs Pillow (for text) and the ffmpeg command-line tool (for encoding).
The output format follows the file extension: .gif, .webp or .mp4.

    python scripts/make_demo.py docs/images/poor-light-demo.gif
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspectflow.detectors import DetectorName  # noqa: E402
from inspectflow.evaluation import evaluate_gate  # noqa: E402
from inspectflow.generator import SyntheticConveyor  # noqa: E402
from inspectflow.pipeline import GATE_FRACTION, InspectionPipeline, annotate_frame, make_detector  # noqa: E402

HEADER_PX = 38
BADGE_FRAMES = 9
BACKGROUND = (17, 20, 24)
TEXT = (236, 239, 242)
MUTED = (150, 158, 166)
GOOD = (88, 214, 141)
BAD = (255, 107, 107)
MISSED = (255, 196, 87)


@dataclass
class Panel:
    title: str
    frames: list[np.ndarray] = field(default_factory=list)
    # frame index -> (badge text, color) for decisions and misses
    events: dict[int, tuple[str, tuple[int, int, int]]] = field(default_factory=dict)
    # running totals per frame: (correct, wrong, missed)
    totals: list[tuple[int, int, int]] = field(default_factory=list)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for folder in ("/usr/share/fonts/truetype/dejavu", "/Library/Fonts", "C:/Windows/Fonts"):
        path = Path(folder) / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def run_panel(detector: DetectorName, title: str, seed: int, frame_count: int) -> Panel:
    conveyor = SyntheticConveyor(seed, frame_count, "poor_lighting")
    gate_x = round(conveyor.width * GATE_FRACTION)
    pipeline = InspectionPipeline(make_detector(detector), gate_x)
    panel = Panel(title)
    truth: list[list[dict]] = []
    for frame in conveyor.iter_frames():
        tracked, _ = pipeline.process(frame.image, frame.index)
        truth.append(frame.ground_truth)
        panel.frames.append(
            annotate_frame(
                frame.image, tracked, frame_index=frame.index, label="poor lighting", gate_x=gate_x
            )
        )

    report = evaluate_gate(pipeline.decisions, truth, gate_x)
    correct_at: list[int] = []
    wrong_at: list[int] = []
    matched: set[str] = set()
    for decision, outcome in zip(pipeline.decisions, report["outcomes"], strict=True):
        frame_index = int(decision["decision_frame"])
        matched.add(outcome["truth_part_id"])
        if outcome["outcome"] in ("correct_pass", "correct_reject"):
            correct_at.append(frame_index)
            panel.events[frame_index] = ("\u2713 correct", GOOD)
        else:
            wrong_at.append(frame_index)
            panel.events[frame_index] = ("\u2717 " + outcome["outcome"].replace("_", " "), BAD)

    # A part is missed on the frame its true centroid crosses the gate.
    missed_at: list[int] = []
    last_x: dict[str, float] = {}
    for index, parts in enumerate(truth):
        for item in parts:
            part, x = item["part_id"], float(item["centroid"][0])
            previous = last_x.get(part)
            if previous is not None and previous < gate_x <= x and part not in matched:
                missed_at.append(index)
                panel.events.setdefault(index, ("missed", MISSED))
            last_x[part] = x

    for index in range(frame_count):
        panel.totals.append(
            (
                sum(at <= index for at in correct_at),
                sum(at <= index for at in wrong_at),
                sum(at <= index for at in missed_at),
            )
        )
    return panel


def compose(panels: list[Panel], index: int, gate_x: int) -> Image.Image:
    width = panels[0].frames[0].shape[1]
    height = panels[0].frames[0].shape[0]
    canvas = Image.new("RGB", (width, len(panels) * (HEADER_PX + height) + 2), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font, count_font, badge_font = font(17, bold=True), font(15), font(14, bold=True)

    for row, panel in enumerate(panels):
        top = row * (HEADER_PX + height + 2)
        draw.text((14, top + 9), panel.title, font=title_font, fill=TEXT)
        correct, wrong, missed = panel.totals[index]
        x = width - 14
        for label, value, color in (
            ("missed", missed, MISSED),
            ("wrong", wrong, BAD),
            ("correct", correct, GOOD),
        ):
            text = f"{value} {label}"
            x -= draw.textlength(text, font=count_font)
            draw.text((x, top + 11), text, font=count_font, fill=color if value else MUTED)
            x -= 18

        frame = cv2.cvtColor(panel.frames[index], cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(frame), (0, top + HEADER_PX))

        for start, (text, color) in panel.events.items():
            if start <= index < start + BADGE_FRAMES:
                box_width = draw.textlength(text, font=badge_font) + 16
                left, upper = gate_x + 8, top + HEADER_PX + 60
                draw.rounded_rectangle(
                    (left, upper, left + box_width, upper + 24), radius=5, fill=color
                )
                draw.text((left + 8, upper + 4), text, font=badge_font, fill=BACKGROUND)
    return canvas


def encode(frame_dir: Path, output: Path, fps: int) -> None:
    pattern = str(frame_dir / "%04d.png")
    suffix = output.suffix.lower()
    if suffix == ".gif":
        filters = (
            f"fps={fps},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
        )
        command = ["-framerate", str(fps), "-i", pattern, "-filter_complex", filters, "-loop", "0"]
    elif suffix == ".webp":
        command = ["-framerate", str(fps), "-i", pattern, "-c:v", "libwebp_anim", "-quality", "70", "-loop", "0"]
    elif suffix == ".mp4":
        command = [
            "-framerate", str(fps), "-i", pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
        ]
    else:
        raise SystemExit("output must end in .gif, .webp or .mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *command, str(output)], check=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", nargs="+", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=144)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")

    panels = [
        run_panel("classical", "Classical rules", args.seed, args.frames),
        run_panel("cnn", "CNN", args.seed, args.frames),
    ]
    gate_x = round(panels[0].frames[0].shape[1] * GATE_FRACTION)
    with tempfile.TemporaryDirectory() as temp:
        for index in range(args.frames):
            compose(panels, index, gate_x).save(Path(temp) / f"{index:04d}.png")
        for output in args.outputs:
            encode(Path(temp), output, args.fps)
            print(f"{output}: {output.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
