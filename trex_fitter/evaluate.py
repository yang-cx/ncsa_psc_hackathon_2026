#!/usr/bin/env python3
"""Evaluate one completed TRExFitter config and emit a JSON result."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

from trex_fitter import runner
from trex_fitter.config_verify import verify_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed TRExFitter .config file.")
    parser.add_argument("config", type=Path, help="Project-local .config file to evaluate.")
    parser.add_argument("--mock", action="store_true", help="Use the fast deterministic mock runner.")
    parser.add_argument("--actions", nargs="+", default=["n", "w", "f", "s"])
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        config = runner.config_path(str(args.config))
        log_dir = (args.log_dir or PROJECT_DIR / "artifacts" / "trex_fitter" / "evaluation" / config.stem).resolve()
        runner.podman_trex.to_container_path(log_dir)
    except ValueError as error:
        parser.error(str(error))
    verification = verify_config(config, actions=args.actions)
    result = {"success": False, "mock": args.mock, "config": str(config),
              "verification": verification.model_dump(), "significance": None,
              "returncode": None, "timed_out": False}
    # Native execution is governed by analysis validity, while the report
    # always includes the separate Coffea compatibility result.
    if not verification.analysis_valid:
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.mock:
        command = [sys.executable, "-m", "trex_fitter.mock",
                   str(config), "--log-dir", str(log_dir)]
    else:
        command = [sys.executable, str(PROJECT_DIR / "trex_fitter" / "runner.py"),
                   str(config), "--actions", *args.actions, "--log-dir", str(log_dir)]
    started = time.monotonic()
    with (log_dir / "evaluation.stdout.log").open("w") as stdout, (log_dir / "evaluation.stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=PROJECT_DIR, stdout=stdout,
                                   stderr=stderr, start_new_session=True)
        try:
            result["returncode"] = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result["returncode"] = process.returncode
    result["elapsed_seconds"] = time.monotonic() - started
    result["log_dir"] = str(log_dir)
    result["success"] = result["returncode"] == 0 and not result["timed_out"]
    if result["success"]:
        # Read this invocation's stdout, not potentially stale analysis logs.
        result["significance"] = runner.podman_trex.parse_significance_from_text(
            (log_dir / "evaluation.stdout.log").read_text())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
