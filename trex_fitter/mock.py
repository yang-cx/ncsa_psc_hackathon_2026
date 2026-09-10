#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def estimate_significance(config_text: str) -> float:
    regions = len(re.findall(r'^\s*Region:\s*"', config_text, flags=re.MULTILINE))
    samples = len(re.findall(r'^\s*Sample:\s*"', config_text, flags=re.MULTILINE))
    bins = [int(x) for x in re.findall(r'Variable:\s*"[^"]+"\s*,\s*(\d+)\s*,', config_text)]
    avg_bins = sum(bins) / len(bins) if bins else 0.0
    return round(0.2 * regions + 0.05 * samples + 0.01 * avg_bins, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock TRExFitter runner for RL smoke tests.")
    parser.add_argument("config")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--log-dir", default="trex_logs")
    args, _ = parser.parse_known_args()

    config_path = Path(args.config)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    significance = estimate_significance(config_path.read_text(errors="ignore"))
    (log_dir / "results.json").write_text(
        json.dumps({"significance": significance, "config": str(config_path)}, indent=2) + "\n"
    )
    print(f"SIGNIFICANCE={significance}")


if __name__ == "__main__":
    main()
