#!/usr/bin/env python3
"""Materialize canonical native-tool JSONL as Arrow-stable VERL Parquet."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.sft_schema import read_jsonl, validate_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrow_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    """Encode heterogeneous JSON leaves without rendering model tokens."""
    messages = copy.deepcopy(row["messages"])
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError(f"{row.get('uuid')}: tool arguments must be objects")
            function["arguments"] = json.dumps(
                arguments, sort_keys=True, separators=(",", ":")
            )
    return {
        "id": row["uuid"],
        "logical_task_id": row["logical_task_id"],
        "messages": messages,
        "tools": json.dumps(
            row.get("tools") or [], sort_keys=True, separators=(",", ":")
        ),
        "enable_thinking": False,
        "tool_contract": row["tool_contract"],
        "source": row["source"],
        "modality": row.get("modality", "native_agent"),
        "domain": row["domain"],
    }


def write_split(source: Path, destination: Path) -> dict[str, Any]:
    rows = validate_records(read_jsonl(source))
    prepared = [arrow_safe_row(row) for row in rows]
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prepared).to_parquet(destination, index=False)
    round_trip = pd.read_parquet(destination)
    if len(round_trip) != len(prepared):
        raise ValueError(f"{destination}: Parquet round-trip changed the row count")
    if round_trip["id"].tolist() != [row["id"] for row in prepared]:
        raise ValueError(f"{destination}: Parquet round-trip changed row ordering or IDs")
    if round_trip["enable_thinking"].tolist() != [False] * len(prepared):
        raise ValueError(f"{destination}: no-thinking column failed round-trip")
    return {
        "rows": len(prepared),
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(destination),
        "output_sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = {
        "schema_version": "trexfitter-verl-materialization/v1",
        "canonical_format": "structured messages plus optional tools",
        "framework_format": "parquet",
        "model_tokens_stored": False,
        "assistant_loss_masks_stored": False,
        "no_thinking": True,
        "splits": {
            "train": write_split(args.train, args.output_dir / "train.parquet"),
            "validation": write_split(
                args.validation, args.output_dir / "validation.parquet"
            ),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
