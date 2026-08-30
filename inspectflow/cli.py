"""Command-line entry point for the local InspectFlow dashboard."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="inspect-flow",
        description="Run the local InspectFlow computer-vision demonstrator.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8000)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    args = parser.parse_args(argv)

    print(f"InspectFlow is starting at http://{args.host}:{args.port}")
    uvicorn.run(
        "inspectflow.webapp:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
