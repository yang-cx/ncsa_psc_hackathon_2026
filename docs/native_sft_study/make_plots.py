#!/usr/bin/env python3
"""Build concise figures and tables for the native-tool SFT study."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "reports/native-sft-study-20260914"
FIGURES = OUTPUT / "figures"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#7B61A8"
GRAY = "#6B7280"

TRAINING_RUNS = {
    "Qwen2.5-Coder 0.5B · full · LR 1e−5": "verl-qwen25-coder-0.5b-full-lr1e-5-e1",
    "Qwen3.5 0.8B · full · LR 1e−5": "verl-qwen35-0.8b-full-lr1e-5-e1",
    "Qwen3.5 0.8B · LoRA r16 · LR 1e−4": "verl-qwen35-0.8b-lora-r16-lr1e-4-e1",
    "Qwen3.5 0.8B · LoRA r16 · LR 5e−5": "verl-qwen35-0.8b-lora-r16-lr5e-5-e1",
    "Qwen3.5 0.8B · LoRA r8 · LR 1e−4": "verl-qwen35-0.8b-lora-r8-lr1e-4-e1",
    "Qwen3.5 2B · full · LR 1e−5": "verl-qwen35-2b-full-lr1e-5-e1",
    "Qwen3.5 4B · LoRA r16 · LR 1e−4": "verl-qwen35-4b-lora-r16-lr1e-4-e1",
    "Qwen3.5 9B · LoRA r16 · LR 1e−4": "verl-qwen35-9b-lora-r16-lr1e-4-e1",
    "Qwen3.5 35B-A3B · LoRA r16 · LR 1e−4": "verl-qwen35-35b-a3b-lora-r16-lr1e-4-e1",
}

EVALUATION_RUNS = {
    "Qwen2.5-Coder 0.5B · base (no SFT)": "qwen25-coder-0.5b-base",
    "Qwen2.5-Coder 0.5B · full SFT": "qwen25-coder-0.5b-full-lr1e-5-e1",
    "Qwen3.5 0.8B · base (no SFT)": "qwen35-0.8b-base",
    "Qwen3.5 0.8B · full SFT": "qwen35-0.8b-full-lr1e-5-e1",
    "Qwen3.5 0.8B · LoRA SFT": "qwen35-0.8b-lora-r16-lr1e-4-e1",
    "Qwen3.5 2B · base (no SFT)": "qwen35-2b-base",
    "Qwen3.5 2B · full SFT": "qwen35-2b-full-lr1e-5-e1",
    "Qwen3.5 4B · base (no SFT)": "qwen35-4b-base",
    "Qwen3.5 4B · LoRA SFT": "qwen35-4b-lora-r16-lr1e-4-e1",
    "Qwen3.5 9B · base (no SFT)": "qwen35-9b-base",
    "Qwen3.5 9B · LoRA SFT": "qwen35-9b-lora-r16-lr1e-4-e1",
    "Qwen3.5 35B-A3B · base (no SFT)": "qwen35-35b-a3b-base",
    "Qwen3.5 35B-A3B · LoRA SFT": "qwen35-35b-a3b-lora-r16-lr1e-4-e1",
    "GPT-6-Astra · default low effort": "codex-gpt-6-astra-default",
    "GPT-5.6-Sol · default medium effort": "codex-gpt-5.6-sol-default",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strict_summary(directory: Path) -> dict | None:
    results_path = directory / "results.jsonl"
    # A JSONL file is flushed task by task; the summary is written only after
    # the selected benchmark finishes, so exclude live partial runs from plots.
    if not results_path.exists() or not (directory / "summary.json").exists():
        return None
    rows = read_jsonl(results_path)
    if len(rows) != 36:
        return None
    if "score" not in rows[0]:
        return None
    strict = [
        bool(row.get("passed", row["score"]["passed"] and row.get("verifier_observed", False)))
        if "status" not in row
        else bool(row["score"]["passed"])
        for row in rows
    ]
    return {
        "tasks": len(rows),
        "passed": sum(strict),
        "pass_rate": sum(strict) / len(rows),
        "requested_state": sum(bool(row["score"].get("contract_ok")) for row in rows),
        "correct_final_config": sum(bool(row["score"].get("passed")) for row in rows),
        "analysis_valid": sum(bool(row["score"].get("analysis_valid")) for row in rows),
        "preservation_ok": sum(bool(row["score"].get("preservation_ok")) for row in rows),
        "verifier_observed": sum(
            bool(row.get("verifier_observed", row["score"].get("agent_validation_observed")))
            for row in rows
        ),
        "tool_errors": sum(int(row.get("tool_errors", 0)) for row in rows),
        "rows": rows,
    }


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = passed / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def finish(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def training_curves() -> list[dict]:
    rows_out = []
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    colors = [BLUE, ORANGE, GREEN, RED, PURPLE, "#56B4E9", "#CC79A7", GRAY]
    for color, (label, run) in zip(colors, TRAINING_RUNS.items()):
        path = ARTIFACTS / "checkpoints" / run / "metrics.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        train = [(row["step"], row["data"]["train/loss"]) for row in rows if "train/loss" in row["data"]]
        validation = [(row["step"], row["data"]["val/loss"]) for row in rows if "val/loss" in row["data"]]
        if train:
            ax.plot(*zip(*train), marker="o", markersize=2.8, linewidth=1.6, label=label, color=color)
        for step, loss in validation:
            rows_out.append({"experiment": label, "step": step, "validation_loss": loss})
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Assistant-token prediction loss (lower is better)")
    ax.set_title("Training loss on the native-tool SFT trajectories")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=7.5, ncol=2, frameon=False)
    finish(fig, "training-loss.png")
    return rows_out


def dataset_composition() -> None:
    task_root = ROOT / "data/datasets/hyy-trexfitter-agent-trajectories/data/native/tasks"
    by_split: dict[str, list[dict]] = {}
    for split in ("train", "validation"):
        path = task_root / f"{split}.jsonl"
        if not path.exists():
            return
        by_split[split] = read_jsonl(path)
    categories = sorted({row["category"] for rows in by_split.values() for row in rows})
    labels = [name.replace("_", " ") for name in categories]
    train = [sum(row["category"] == category for row in by_split["train"]) for category in categories]
    validation = [sum(row["category"] == category for row in by_split["validation"]) for category in categories]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.7), gridspec_kw={"width_ratios": [1.8, 1]})
    x = np.arange(len(categories))
    left.bar(x, train, color=BLUE, label="Train: 120 logical tasks")
    left.bar(x, validation, bottom=train, color=ORANGE, label="Validation: 36 logical tasks")
    left.set_xticks(x, labels, rotation=40, ha="right", fontsize=8)
    left.set_ylabel("Logical tasks")
    left.set_title("Coverage by config decision")
    left.grid(axis="y", alpha=0.22)
    left.legend(frameon=False, fontsize=8)

    levels = (1, 2, 3, 4)
    train_level = [sum(row["complexity"] == level for row in by_split["train"]) for level in levels]
    validation_level = [sum(row["complexity"] == level for row in by_split["validation"]) for level in levels]
    right.bar(levels, train_level, color=BLUE)
    right.bar(levels, validation_level, bottom=train_level, color=ORANGE)
    right.set_xticks(levels)
    right.set_xlabel("Construction complexity (1 easiest; 4 hardest)")
    right.set_ylabel("Logical tasks")
    right.set_title("Difficulty mix")
    right.grid(axis="y", alpha=0.22)
    finish(fig, "dataset-composition.png")


def endpoint_validation_loss(rows: list[dict]) -> None:
    if not rows:
        return
    latest = {}
    for row in rows:
        if row["experiment"] not in latest or row["step"] > latest[row["experiment"]]["step"]:
            latest[row["experiment"]] = row
    labels = list(latest)
    values = [latest[label]["validation_loss"] for label in labels]
    fig, ax = plt.subplots(figsize=(8.4, max(3.8, 0.42 * len(labels))))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=BLUE)
    ax.set_yticks(positions, labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Validation loss after one epoch (lower is better)")
    ax.set_title("Training-method and learning-rate ablation")
    ax.grid(axis="x", alpha=0.22)
    for y, value in zip(positions, values):
        ax.text(value, y, f"  {value:.4f}", va="center", fontsize=8)
    finish(fig, "validation-loss-ablation.png")


def evaluation_table() -> dict[str, dict]:
    summaries = {}
    for label, run in EVALUATION_RUNS.items():
        summary = strict_summary(ARTIFACTS / "evaluation" / run)
        if summary:
            summaries[label] = summary
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "evaluation-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "experiment", "strict_passed", "tasks", "strict_pass_rate",
            "requested_state", "correct_final_config", "verifier_observed", "analysis_valid",
            "preservation_ok", "tool_errors",
        ])
        for label, value in summaries.items():
            writer.writerow([
                label, value["passed"], value["tasks"], value["pass_rate"],
                value["requested_state"], value["correct_final_config"], value["verifier_observed"],
                value["analysis_valid"], value["preservation_ok"], value["tool_errors"],
            ])
    return summaries


def outcome_comparison(summaries: dict[str, dict]) -> None:
    if not summaries:
        return
    preferred = [label for label in EVALUATION_RUNS if label in summaries]
    labels = preferred
    values = [100 * summaries[label]["pass_rate"] for label in labels]
    errors = []
    for label in labels:
        row = summaries[label]
        lo, hi = wilson(row["passed"], row["tasks"])
        errors.append([100 * row["pass_rate"] - 100 * lo, 100 * hi - 100 * row["pass_rate"]])
    colors = [GRAY if "base" in label else (PURPLE if "GPT" in label else BLUE) for label in labels]
    fig, ax = plt.subplots(figsize=(8.6, max(4.2, 0.42 * len(labels))))
    positions = np.arange(len(labels))
    ax.barh(positions, values, color=colors, xerr=np.asarray(errors).T, capsize=3)
    ax.set_yticks(positions, labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Strict task success on 36 validation tasks (%)")
    ax.set_title("Untouched base checkpoints, SFT checkpoints, and hosted controls")
    ax.grid(axis="x", alpha=0.22)
    for y, label in zip(positions, labels):
        row = summaries[label]
        ax.text(min(101, 100 * row["pass_rate"] + 1.2), y, f"{row['passed']}/{row['tasks']}", va="center", fontsize=8)
    finish(fig, "strict-task-success.png")


def base_vs_sft(summaries: dict[str, dict]) -> None:
    """Show only controlled pairs that differ by project SFT."""
    pairs = [
        ("Qwen2.5-Coder 0.5B", "Qwen2.5-Coder 0.5B · base (no SFT)", "Qwen2.5-Coder 0.5B · full SFT"),
        ("Qwen3.5 0.8B full", "Qwen3.5 0.8B · base (no SFT)", "Qwen3.5 0.8B · full SFT"),
        ("Qwen3.5 0.8B LoRA", "Qwen3.5 0.8B · base (no SFT)", "Qwen3.5 0.8B · LoRA SFT"),
        ("Qwen3.5 2B", "Qwen3.5 2B · base (no SFT)", "Qwen3.5 2B · full SFT"),
        ("Qwen3.5 4B", "Qwen3.5 4B · base (no SFT)", "Qwen3.5 4B · LoRA SFT"),
        ("Qwen3.5 9B", "Qwen3.5 9B · base (no SFT)", "Qwen3.5 9B · LoRA SFT"),
        ("Qwen3.5 35B-A3B", "Qwen3.5 35B-A3B · base (no SFT)", "Qwen3.5 35B-A3B · LoRA SFT"),
    ]
    available = [(name, base, sft) for name, base, sft in pairs if base in summaries and sft in summaries]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(8.4, max(3.4, 0.62 * len(available))))
    positions = np.arange(len(available))
    base_values = [100 * summaries[base]["pass_rate"] for _, base, _ in available]
    sft_values = [100 * summaries[sft]["pass_rate"] for _, _, sft in available]
    for y, before, after in zip(positions, base_values, sft_values):
        ax.plot([before, after], [y, y], color=GRAY, linewidth=2, zorder=1)
    ax.scatter(base_values, positions, color=GRAY, s=54, label="Untouched base checkpoint", zorder=2)
    ax.scatter(sft_values, positions, color=BLUE, s=54, label="After project SFT", zorder=2)
    ax.set_yticks(positions, labels=[name for name, _, _ in available], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-2, 105)
    ax.set_xlabel("Strict task success on the same 36 validation tasks (%)")
    ax.set_title("Matched comparison: effect of native-tool SFT")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    for y, before, after in zip(positions, base_values, sft_values):
        ax.text(after + 1.2, y, f"{after - before:+.1f} pp", va="center", fontsize=8, color=BLUE)
    finish(fig, "base-vs-sft.png")

    with (OUTPUT / "base-vs-sft.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_and_method", "base_pass_rate", "sft_pass_rate", "change_percentage_points"])
        for (name, _, _), before, after in zip(available, base_values, sft_values):
            writer.writerow([name, before / 100, after / 100, after - before])


def outcome_gates(summaries: dict[str, dict]) -> None:
    labels = [label for label in (
        "Qwen3.5 0.8B · base (no SFT)",
        "Qwen3.5 0.8B · full SFT",
        "Qwen3.5 0.8B · LoRA SFT",
        "GPT-6-Astra · default low effort",
        "GPT-5.6-Sol · default medium effort",
    ) if label in summaries]
    if not labels:
        return
    measures = ["requested_state", "analysis_valid", "preservation_ok", "correct_final_config", "verifier_observed", "passed"]
    titles = ["Requested state", "Static validity", "Preservation", "Valid final config", "Verifier observed", "Strict pass"]
    x = np.arange(len(measures))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    for index, label in enumerate(labels):
        row = summaries[label]
        values = [100 * row[key] / row["tasks"] for key in measures]
        ax.bar(x - 0.4 + width / 2 + index * width, values, width, label=label)
    ax.set_xticks(x, titles)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Validation tasks satisfying gate (%)")
    ax.set_title("Why a trajectory passes or fails")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=7.3, ncol=2, frameon=False)
    finish(fig, "outcome-gates.png")


def category_heatmap(summaries: dict[str, dict]) -> None:
    labels = [label for label in summaries if label in (
        "Qwen3.5 0.8B · base (no SFT)", "Qwen3.5 0.8B · full SFT",
        "Qwen3.5 0.8B · LoRA SFT", "GPT-6-Astra · default low effort",
        "GPT-5.6-Sol · default medium effort",
    )]
    if not labels:
        return
    category_counts: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for label in labels:
        for row in summaries[label]["rows"]:
            category = row.get("category") or row.get("domain", "").removeprefix("Code_Agent-ATLAS_TRExFitter-")
            passed = row["score"]["passed"] if "status" in row else row.get(
                "passed", row["score"]["passed"] and row.get("verifier_observed", False)
            )
            category_counts[category][label].append(bool(passed))
    categories = sorted(category_counts)
    matrix = np.array([
        [
            100 * sum(category_counts[category].get(label, [])) / len(category_counts[category][label])
            if category_counts[category].get(label, []) else np.nan
            for category in categories
        ]
        for label in labels
    ])
    fig, ax = plt.subplots(figsize=(9.0, max(3.4, 0.55 * len(labels))))
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(categories)), [name.replace("_", " ") for name in categories], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            ax.text(x, y, "—" if np.isnan(value) else f"{value:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if value > 55 else "#17212B")
    fig.colorbar(image, ax=ax, label="Strict task success (%)", fraction=0.025, pad=0.02)
    ax.set_title("Strict success by validation-task category")
    finish(fig, "category-success.png")


def temperature_study() -> None:
    runs = [
        (0.0, "qwen35-0.8b-full-lr1e-5-e1"),
        (0.2, "qwen35-0.8b-full-temp0.2-seed1"),
        (0.7, "qwen35-0.8b-full-temp0.7-seed1"),
    ]
    points = []
    for temperature, run in runs:
        summary = strict_summary(ARTIFACTS / "evaluation" / run)
        if summary:
            points.append((temperature, summary))
    if len(points) < 2:
        return
    x = np.arange(len(points))
    width = 0.34
    strict = [100 * summary["passed"] / summary["tasks"] for _, summary in points]
    verifier = [100 * summary["verifier_observed"] / summary["tasks"] for _, summary in points]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x - width / 2, strict, width, color=BLUE, label="Strict task success")
    ax.bar(x + width / 2, verifier, width, color=ORANGE, label="Successful verifier observed")
    ax.set_xticks(x, [str(temperature) for temperature, _ in points])
    ax.set_xlabel("Sampling temperature (top-p = 0.95 when temperature > 0)")
    ax.set_ylabel("Validation tasks (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Qwen3.5-0.8B full SFT: exploratory decoding study")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    for positions, values in ((x - width / 2, strict), (x + width / 2, verifier)):
        for position, value in zip(positions, values):
            ax.text(position, value + 1.5, f"{value:.1f}%", ha="center", fontsize=8)
    finish(fig, "temperature-study.png")


def throughput_scaling() -> None:
    rows = []
    for gpus in (1, 2, 4, 8, 16):
        path = ARTIFACTS / "checkpoints" / f"throughput-qwen35-0.8b-a100-{gpus}" / "throughput.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rate = payload["summary"]["total_sequence_tokens_per_second"]
        rows.append((gpus, rate))
    if not rows:
        return
    base = rows[0][1] / rows[0][0]
    efficiency = [100 * rate / (base * gpus) for gpus, rate in rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    gpus = [row[0] for row in rows]
    rates = [row[1] for row in rows]
    ax.plot(gpus, rates, marker="o", linewidth=2, color=BLUE, label="Measured throughput")
    ax.plot(gpus, [base * value for value in gpus], linestyle="--", color=GRAY, label="Ideal linear scaling")
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpus, labels=[str(value) for value in gpus])
    ax.set_xlabel("A100 GPUs")
    ax.set_ylabel("Training-sequence tokens/s")
    ax.set_title("Qwen3.5-0.8B strong scaling at fixed global batch")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    for x, y, e in zip(gpus, rates, efficiency):
        ax.annotate(f"{y:,.0f}\n{e:.0f}% eff.", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    finish(fig, "a100-throughput-scaling.png")
    with (OUTPUT / "throughput-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a100_gpus", "sequence_tokens_per_second", "scaling_efficiency_percent"])
        for (gpu_count, rate), eff in zip(rows, efficiency):
            writer.writerow([gpu_count, rate, eff])


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset_composition()
    validation_rows = training_curves()
    endpoint_validation_loss(validation_rows)
    summaries = evaluation_table()
    outcome_comparison(summaries)
    base_vs_sft(summaries)
    outcome_gates(summaries)
    category_heatmap(summaries)
    temperature_study()
    throughput_scaling()
    manifest = {
        "training_runs": TRAINING_RUNS,
        "evaluation_runs": EVALUATION_RUNS,
        "generated_figures": sorted(path.name for path in FIGURES.glob("*.png")),
    }
    (OUTPUT / "plot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
