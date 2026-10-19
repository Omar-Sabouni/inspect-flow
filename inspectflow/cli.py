"""``inspect-flow``: the dashboard by default, plus bench, video and render."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .detectors import DETECTOR_NAMES
from .generator import SCENARIOS


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="inspect-flow")
    commands = parser.add_subparsers(dest="command")

    serve = commands.add_parser("serve", help="run the dashboard (default)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    bench = commands.add_parser("bench", help="score both detectors on the test scenes")
    bench.add_argument("--detectors", nargs="+", choices=DETECTOR_NAMES, default=list(DETECTOR_NAMES))
    bench.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    bench.add_argument("--seeds", nargs="+", type=int, help="default: 7 to 26")
    bench.add_argument("--frames", type=int, default=240)
    bench.add_argument("--threads", type=int, help="OpenCV threads (default: all cores)")
    bench.add_argument("--json", type=Path, help="also write every case to this file")

    video = commands.add_parser("video", help="inspect a video file, camera index or stream URL")
    video.add_argument("source")
    video.add_argument("--detector", choices=DETECTOR_NAMES, default="classical")
    video.add_argument("--output", type=Path, help="annotated .mp4 or .avi")
    video.add_argument("--log", type=Path, help="decision CSV")

    render = commands.add_parser("render", help="write a synthetic scene to video")
    render.add_argument("output", type=Path)
    render.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    render.add_argument("--seed", type=int, default=7)
    render.add_argument("--frames", type=int, default=240)

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in commands.choices and args[0] not in ("-h", "--help"):
        args.insert(0, "serve")
    options = parser.parse_args(args)

    if options.command == "serve":
        import uvicorn

        uvicorn.run("inspectflow.webapp:app", host=options.host, port=options.port)
    elif options.command == "bench":
        from .benchmark import TEST_SEEDS, format_markdown, run_benchmark

        report = run_benchmark(
            options.detectors,
            options.scenarios,
            options.seeds or TEST_SEEDS,
            options.frames,
            options.threads,
            progress=lambda c: _log(f"{c['detector']} {c['scenario']} seed {c['seed']}: {c['right_call']:.0%} right"),
        )
        if options.json:
            options.json.write_text(json.dumps(report, indent=1))
        print(format_markdown(report), end="")
    elif options.command == "video":
        from .video import inspect_video

        try:
            summary = inspect_video(
                options.source,
                options.detector,
                options.output,
                options.log,
                on_decision=lambda d: _log(f"frame {d['decision_frame']}: {d['predicted_defect']}"),
            )
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps(summary, indent=2))
    else:
        from .video import render_scene

        try:
            render_scene(options.output, options.scenario, options.seed, options.frames)
        except ValueError as error:
            parser.error(str(error))


def _log(message: str) -> None:
    print(message, file=sys.stderr)


if __name__ == "__main__":
    main()
