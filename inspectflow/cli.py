"""Command-line entry point for InspectFlow.

``inspect-flow`` with no subcommand (or ``inspect-flow serve``) starts the
local dashboard. ``bench``, ``video`` and ``render`` run without the server.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .benchmark import TEST_SEEDS
from .detectors import DETECTOR_NAMES

SUBCOMMANDS = ("serve", "bench", "video", "render")


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8000)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect-flow",
        description="Conveyor inspection with classical and CNN detectors.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    serve = commands.add_parser("serve", help="run the local dashboard (default)")
    _add_serve_arguments(serve)

    bench = commands.add_parser(
        "bench", help="score detectors against synthetic ground truth"
    )
    bench.add_argument("--detectors", nargs="+", choices=DETECTOR_NAMES, default=list(DETECTOR_NAMES))
    bench.add_argument(
        "--scenarios",
        nargs="+",
        choices=("nominal", "poor_lighting"),
        default=["nominal", "poor_lighting"],
    )
    bench.add_argument(
        "--seeds", nargs="+", type=int, default=list(TEST_SEEDS), help="default: 7-26"
    )
    bench.add_argument("--frames", type=_positive_int, default=240)
    bench.add_argument(
        "--threads", type=_positive_int, help="OpenCV worker threads (default: all)"
    )
    bench.add_argument("--json", type=Path, help="also write the full report here")

    video = commands.add_parser(
        "video", help="inspect a video file, camera index or stream URL"
    )
    video.add_argument("source", help="file path, camera index such as 0, or URL")
    video.add_argument("--detector", choices=DETECTOR_NAMES, default="classical")
    video.add_argument("--output", type=Path, help="annotated .mp4 or .avi to write")
    video.add_argument("--log", type=Path, help="decision CSV to write")
    video.add_argument("--max-frames", type=_positive_int)
    video.add_argument(
        "--gate", type=float, default=0.25, help="gate position as a fraction of width"
    )

    render = commands.add_parser(
        "render", help="write a synthetic conveyor scene to a video file"
    )
    render.add_argument("output", type=Path, help=".mp4 or .avi path")
    render.add_argument("--scenario", choices=("nominal", "poor_lighting"), default="nominal")
    render.add_argument("--seed", type=int, default=7)
    render.add_argument("--frames", type=_positive_int, default=240)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Keep ``inspect-flow --port 8000`` working: no subcommand means serve.
    if not arguments or arguments[0] not in (*SUBCOMMANDS, "-h", "--help"):
        arguments.insert(0, "serve")
    args = build_parser().parse_args(arguments)

    if args.command == "serve":
        _serve(args)
    elif args.command == "bench":
        _bench(args)
    elif args.command == "video":
        _video(args)
    else:
        _render(args)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    print(f"InspectFlow is starting at http://{args.host}:{args.port}")
    uvicorn.run(
        "inspectflow.webapp:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
        workers=1,
    )


def _bench(args: argparse.Namespace) -> None:
    from .benchmark import format_markdown, run_benchmark

    def progress(case: dict[str, Any]) -> None:
        print(
            f"{case['detector']:>9} {case['scenario']:<13} seed {case['seed']:<4} "
            f"recall {case['detection_recall']:.3f}  "
            f"gate accuracy {case['gate_verdict_accuracy']:.3f}  "
            f"detect p50 {case['detect_p50_ms']:.2f} ms",
            file=sys.stderr,
        )

    report = run_benchmark(
        detectors=args.detectors,
        scenarios=args.scenarios,
        seeds=args.seeds,
        frame_count=args.frames,
        threads=args.threads,
        progress=progress,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_markdown(report), end="")


def _video(args: argparse.Namespace) -> None:
    from .video import inspect_video

    def announce(decision: dict[str, Any]) -> None:
        print(
            f"frame {decision['decision_frame']:>5}  track {decision['track_id']:>4}  "
            f"{decision['verdict'].upper():<4} {decision['predicted_defect']}",
            file=sys.stderr,
        )

    try:
        summary = inspect_video(
            args.source,
            detector=args.detector,
            output=args.output,
            log=args.log,
            gate_fraction=args.gate,
            max_frames=args.max_frames,
            on_decision=announce,
        )
    except ValueError as exc:
        raise SystemExit(f"inspect-flow video: {exc}") from exc
    print(json.dumps(summary, indent=2))


def _render(args: argparse.Namespace) -> None:
    from .video import render_scenario

    try:
        frames = render_scenario(
            args.output, scenario=args.scenario, seed=args.seed, frame_count=args.frames
        )
    except ValueError as exc:
        raise SystemExit(f"inspect-flow render: {exc}") from exc
    print(f"Wrote {frames} frames to {args.output}")


if __name__ == "__main__":
    main()
