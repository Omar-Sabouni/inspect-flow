#!/usr/bin/env python3
"""Install a wheel in isolation and smoke-test its CLI and bundled dashboard."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def _free_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_ready(url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 45
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.communicate()[0]
            raise RuntimeError(
                f"installed server exited with {process.returncode}:\n{output}"
            )
        try:
            with urlopen(url, timeout=1) as response:  # noqa: S310 - loopback only
                payload = json.load(response)
            if payload == {
                "status": "ready",
                "dashboard": "available",
                "cnn_model": "available",
            }:
                return
        except (OSError, URLError, ValueError) as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"installed server did not become ready: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"not a wheel: {wheel}")

    with tempfile.TemporaryDirectory(prefix="inspectflow-wheel-") as temp_name:
        temp = Path(temp_name)
        environment = temp / "venv"
        # ``with_pip=False`` also works on minimal Debian images without the
        # optional python3-venv/ensurepip payload. The invoking environment's
        # pip can target the isolated interpreter directly.
        venv.EnvBuilder(with_pip=False, clear=True).create(environment)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        command = scripts / ("inspect-flow.exe" if os.name == "nt" else "inspect-flow")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--python",
                str(python),
                "install",
                "--disable-pip-version-check",
                str(wheel),
            ],
            cwd=temp,
            check=True,
        )

        clean_environment = os.environ.copy()
        clean_environment.pop("PYTHONPATH", None)
        subprocess.run(
            [str(command), "--help"],
            cwd=temp,
            env=clean_environment,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from importlib.resources import files; "
                    "root = files('inspectflow') / 'static'; "
                    "assert (root / 'index.html').is_file(); "
                    "assert (root / 'styles.css').is_file(); "
                    "assert (root / 'app.js').is_file(); "
                    "model = files('inspectflow') / 'models'; "
                    "assert (model / 'part_detector.onnx').is_file(); "
                    "assert (model / 'part_detector.json').is_file()"
                ),
            ],
            cwd=temp,
            env=clean_environment,
            check=True,
        )

        port = _free_loopback_port()
        process = subprocess.Popen(
            [
                str(command),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=temp,
            env={**clean_environment, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_ready(f"http://127.0.0.1:{port}/api/ready", process)
            with urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:  # noqa: S310
                html = response.read().decode("utf-8")
            if "InspectFlow" not in html:
                raise RuntimeError("installed dashboard did not contain its title")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    print(f"Wheel smoke test passed: {wheel.name}")


if __name__ == "__main__":
    main()
