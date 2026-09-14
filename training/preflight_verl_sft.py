#!/usr/bin/env python3
"""Validate every VERL SFT row with the exact selected checkpoint tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from training.verl_dataset import TReXNativeToolSFTDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    config = OmegaConf.create(
        {
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": False,
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "require_native_tool_contract": True,
            "pad_mode": "no_padding",
            "max_length": args.max_length,
            "truncation": "error",
            "shuffle": False,
            "ignore_input_ids_mismatch": False,
        }
    )
    dataset = TReXNativeToolSFTDataset(
        str(args.input), tokenizer=tokenizer, config=config
    )
    lengths: list[int] = []
    assistant_lengths: list[int] = []
    decoded_examples: list[dict[str, object]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        if input_ids.shape != loss_mask.shape:
            raise ValueError(f"row {index}: input/loss mask shape mismatch")
        if not bool(loss_mask.any()):
            raise ValueError(f"row {index}: no assistant tokens are supervised")
        lengths.append(int(input_ids.numel()))
        assistant_lengths.append(int(loss_mask.sum().item()))
        if index < 2:
            supervised_ids = input_ids[loss_mask.to(bool)]
            decoded_examples.append(
                {
                    "id": dataset.dataframe.iloc[index]["id"],
                    "rendered": tokenizer.decode(input_ids, skip_special_tokens=False),
                    "supervised_assistant_span": tokenizer.decode(
                        supervised_ids, skip_special_tokens=False
                    ),
                    "tokens": lengths[-1],
                    "assistant_tokens": assistant_lengths[-1],
                }
            )

    template = tokenizer.chat_template
    resolved_revision = (
        args.revision
        or getattr(tokenizer, "_commit_hash", None)
        or tokenizer.init_kwargs.get("_commit_hash")
    )
    if not Path(args.model).exists() and not resolved_revision:
        raise ValueError("Hub tokenizer revision is unresolved")
    report = {
        "schema_version": "trexfitter-verl-tokenizer-preflight/v1",
        "passed": True,
        "trainer": "verl.trainer.sft_trainer",
        "dataset_class": "training.verl_dataset.TReXNativeToolSFTDataset",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "no_thinking": True,
        "full_render_equals_verl_turn_render": True,
        "assistant_only_loss": True,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "rows": len(dataset),
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "minimum_assistant_tokens": min(assistant_lengths),
        "maximum_assistant_tokens": max(assistant_lengths),
        "max_length": args.max_length,
        "minimum_headroom_tokens": args.max_length - max(lengths),
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("verl", "transformers", "torch", "peft")
        },
        "decoded_examples": decoded_examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "decoded_examples"}, indent=2))


if __name__ == "__main__":
    main()
