#!/usr/bin/env python3
"""Enforce independent line and branch coverage thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _percentage(covered: int, total: int) -> float:
    return 100.0 if total == 0 else covered / total * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-lines", type=float, default=90.0)
    parser.add_argument("--min-branches", type=float, default=85.0)
    args = parser.parse_args()

    document: dict[str, Any] = json.loads(args.report.read_text(encoding="utf-8"))
    totals = document["totals"]
    line_coverage = _percentage(
        int(totals["covered_lines"]), int(totals["num_statements"])
    )
    branch_coverage = _percentage(
        int(totals.get("covered_branches", 0)), int(totals.get("num_branches", 0))
    )
    print(
        f"Line coverage: {line_coverage:.2f}% (minimum {args.min_lines:.2f}%)"
    )
    print(
        f"Branch coverage: {branch_coverage:.2f}% "
        f"(minimum {args.min_branches:.2f}%)"
    )

    failures = []
    if line_coverage < args.min_lines:
        failures.append("line")
    if branch_coverage < args.min_branches:
        failures.append("branch")
    if failures:
        parser.error(f"coverage below threshold: {', '.join(failures)}")


if __name__ == "__main__":
    main()
