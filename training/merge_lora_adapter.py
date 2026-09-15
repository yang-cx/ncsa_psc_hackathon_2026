#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model for native-harness serving."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.output_dir)
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
