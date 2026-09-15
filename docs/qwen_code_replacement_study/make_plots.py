#!/usr/bin/env python3
"""Create the two evidence plots for the native Qwen Code replacement study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "artifacts/native-sft/qwen-code-replacement-20260915"
OUTPUT = ROOT / "artifacts/reports/qwen-code-replacement-study-20260915"
FIGURES = OUTPUT / "figures"
CHECKPOINT = (
    ROOT
    / "artifacts/checkpoints/verl-qwen35-4b-qwen-code-0.23.4-lora-r16-e1"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    metrics = read_jsonl(CHECKPOINT / "metrics.jsonl")
    train = [
        (row["step"], row["data"]["train/loss"])
        for row in metrics
        if "train/loss" in row["data"]
    ]
    validation = [
        (row["step"], row["data"]["val/loss"])
        for row in metrics
        if "val/loss" in row["data"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(*zip(*train), marker="o", markersize=3.5, linewidth=1.8, color="#0072B2")
    if validation:
        ax.scatter(*zip(*validation), marker="D", s=48, color="#D55E00", zorder=3)
        ax.annotate(
            f"held-out loss {validation[-1][1]:.3f}",
            validation[-1], xytext=(-105, 18), textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#555555"}, fontsize=9,
        )
    ax.set(xlabel="Optimizer step", ylabel="Assistant-token loss (lower is better)",
           title="One-epoch Qwen3.5-4B LoRA training")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "training-loss.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "training-loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    labels = ["Base Qwen3.5-4B", "One-epoch LoRA SFT"]
    directories = ["4b-base-validation-matched6", "4b-sft-validation-matched6"]
    passed = []
    for directory in directories:
        rows = read_jsonl(STUDY / directory / "results.jsonl")
        if len(rows) != 6:
            raise ValueError(f"matched evaluation {directory} has {len(rows)} rows, expected 6")
        passed.append(sum(bool(row["score"]["passed"]) for row in rows))
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    bars = ax.bar(labels, passed, color=["#6B7280", "#0072B2"], width=0.58)
    ax.set_ylim(0, 6.6)
    ax.set_ylabel("Strictly successful tasks (out of 6)")
    ax.set_title("Matched native Qwen Code evaluation")
    ax.grid(axis="y", alpha=0.22)
    for bar, count in zip(bars, passed):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.12,
                f"{count}/6", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "matched-strict-success.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "matched-strict-success.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "base_strict_success": passed[0],
        "sft_strict_success": passed[1],
        "tasks_per_model": 6,
        "training_steps": len(train),
        "first_training_loss": train[0][1],
        "last_training_loss": train[-1][1],
        "validation_loss": validation[-1][1],
    }
    (OUTPUT / "plot-data.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
