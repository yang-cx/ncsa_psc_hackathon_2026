#!/usr/bin/env python3
"""Timestamp VERL metric writes and summarize steady-state training throughput.

VERL records the number of sequence tokens processed by every optimizer step but
does not include wall-clock timestamps in its JSONL file.  This sidecar watches
that file before training starts.  Consecutive completion timestamps therefore
measure each later step without modifying the VERL source tree.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train-steps", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument(
        "--warmup-steps", type=int, default=1,
        help="Completed training steps excluded before calculating throughput",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    return parser.parse_args()


def summarize(observations: list[dict], warmup_steps: int) -> dict:
    intervals = []
    for previous, current in zip(observations, observations[1:]):
        if current["step"] <= warmup_steps:
            continue
        elapsed = current["monotonic_seconds"] - previous["monotonic_seconds"]
        if elapsed <= 0:
            continue
        intervals.append({
            "step": current["step"],
            "seconds": elapsed,
            "sequence_tokens": current["sequence_tokens"],
            "sequence_tokens_per_second": current["sequence_tokens"] / elapsed,
        })
    rates = [row["sequence_tokens_per_second"] for row in intervals]
    return {
        "metric_definition": "total non-padding training-sequence tokens per wall-clock second",
        "warmup_steps_excluded": warmup_steps,
        "observed_train_steps": len(observations),
        "timed_intervals": intervals,
        "mean_sequence_tokens_per_second": statistics.fmean(rates) if rates else None,
        "median_sequence_tokens_per_second": statistics.median(rates) if rates else None,
        "total_sequence_tokens_per_second": (
            sum(row["sequence_tokens"] for row in intervals)
            / sum(row["seconds"] for row in intervals)
            if intervals else None
        ),
    }


def main() -> None:
    args = parse_args()
    if args.expected_train_steps < 2 or args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        raise ValueError("invalid step count, polling interval, or timeout")
    if args.warmup_steps < 1 or args.warmup_steps >= args.expected_train_steps:
        raise ValueError("warmup steps must be between 1 and expected steps minus one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    seen_steps: set[int] = set()
    observations: list[dict] = []
    while len(observations) < args.expected_train_steps:
        if time.monotonic() - started > args.timeout_seconds:
            raise TimeoutError(f"timed out waiting for {args.metrics}")
        if args.metrics.exists():
            for line in args.metrics.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = row.get("data", {})
                step = row.get("step")
                tokens = data.get("train/global_tokens")
                if not isinstance(step, int) or not isinstance(tokens, (int, float)) or step in seen_steps:
                    continue
                observed = {
                    "step": step,
                    "sequence_tokens": tokens,
                    "monotonic_seconds": time.monotonic(),
                    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                observations.append(observed)
                seen_steps.add(step)
        if len(observations) < args.expected_train_steps:
            time.sleep(args.poll_seconds)
    payload = {
        "metrics_file": str(args.metrics),
        "poll_seconds": args.poll_seconds,
        "observations": observations,
        "summary": summarize(observations, args.warmup_steps),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
