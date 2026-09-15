#!/usr/bin/env python3
"""Build concise evidence plots and appendix tables for the full native study."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "artifacts/native-sft/qwen-code-full-study-20260915"
DATASET = ROOT / "data/datasets/hyy-trexfitter-agent-trajectories"
OUTPUT = ROOT / "artifacts/reports/qwen-code-native-full-study-20260915"
FIGURES = OUTPUT / "figures"

RUNS = OrderedDict([
    ("0.5B base", "0.5b-base-validation-full36"),
    ("0.5B full SFT", "0.5b-sft-validation-full36"),
    ("0.8B base", "0.8b-base-validation-full36"),
    ("0.8B LoRA", "0.8b-sft-validation-full36"),
    ("0.8B full SFT", "0.8b-full-validation-full36"),
    ("2B base", "2b-base-validation-full36"),
    ("2B full SFT", "2b-sft-validation-full36"),
    ("4B base", "4b-base-validation-full36"),
    ("4B LoRA", "4b-sft-validation-full36"),
    ("9B base", "9b-base-validation-full36"),
    ("9B LoRA", "9b-sft-validation-full36"),
    ("27B base", "27b-base-validation-full36"),
    ("27B LoRA", "27b-sft-validation-full36"),
])

CHECKPOINTS = OrderedDict([
    ("0.5B full SFT", "verl-qwen25-coder-0.5b-qwen-code-0.23.4-full-lr1e-5-e1"),
    ("0.8B LoRA", "verl-qwen35-0.8b-qwen-code-0.23.4-lora-r16-e1"),
    ("0.8B full SFT", "verl-qwen35-0.8b-qwen-code-0.23.4-full-lr1e-5-e1"),
    ("2B full SFT", "verl-qwen35-2b-qwen-code-0.23.4-full-lr1e-5-e1"),
    ("4B LoRA", "verl-qwen35-4b-qwen-code-0.23.4-lora-r16-e1"),
    ("4B LoRA, LR 1e-5", "verl-qwen35-4b-qwen-code-0.23.4-lora-r16-lr1e-5-e1"),
    ("4B LoRA, rank 8", "verl-qwen35-4b-qwen-code-0.23.4-lora-r8-lr1e-4-e1"),
    ("9B LoRA", "verl-qwen35-9b-qwen-code-0.23.4-lora-r16-e1-2gpu"),
    ("27B LoRA", "verl-qwen35-27b-qwen-code-0.23.4-lora-r16-lr1e-4-e1-16gpu"),
])

HYPERPARAMETERS = OrderedDict([
    ("rank 16\nLR $10^{-4}$", (
        "4b-sft-temp0-seed1-v3-full36",
        "verl-qwen35-4b-qwen-code-0.23.4-lora-r16-e1",
    )),
    ("rank 16\nLR $10^{-5}$", (
        "4b-lora-r16-lr1e-5-temp0-seed1-validation-full36",
        "verl-qwen35-4b-qwen-code-0.23.4-lora-r16-lr1e-5-e1",
    )),
    ("rank 8\nLR $10^{-4}$", (
        "4b-lora-r8-lr1e-4-temp0-seed1-validation-full36",
        "verl-qwen35-4b-qwen-code-0.23.4-lora-r8-lr1e-4-e1",
    )),
])

TEMPERATURES = OrderedDict([
    (0.0, "4b-sft-temp0-seed1-v3-full36"),
    (0.2, "4b-sft-temp0p2-seed1-v3-full36"),
    (0.7, "4b-sft-temp0p7-seed1-v3-full36"),
])

CONTROLLED_PAIR = OrderedDict([
    ("4B base", "4b-base-temp0-seed1-validation-full36"),
    ("4B LoRA", "4b-sft-temp0-seed1-v3-full36"),
])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def completed_runs() -> OrderedDict[str, tuple[Path, list[dict], dict]]:
    task_rows = read_jsonl(DATASET / "data/tasks.jsonl")
    for path in sorted((DATASET / "data/native/source_tasks").glob("*.jsonl")):
        task_rows.extend(read_jsonl(path))
    expected_ids = {row["task_id"] for row in task_rows if row["split"] == "validation"}
    if len(expected_ids) != 36:
        raise RuntimeError(f"benchmark defines {len(expected_ids)} rather than 36 validation tasks")
    found = OrderedDict()
    incomplete = []
    for label, dirname in RUNS.items():
        directory = STUDY / dirname
        if not (directory / "summary.json").is_file():
            incomplete.append(f"{label}: missing summary")
            continue
        rows = read_jsonl(directory / "results.jsonl")
        if len(rows) != 36:
            incomplete.append(f"{label}: {len(rows)}/36 rows")
            continue
        task_ids = [row["task_id"] for row in rows]
        if len(set(task_ids)) != 36:
            incomplete.append(f"{label}: {len(set(task_ids))}/36 unique task IDs")
            continue
        if set(task_ids) != expected_ids:
            incomplete.append(f"{label}: task IDs do not match the validation benchmark")
            continue
        found[label] = (directory, rows, read_json(directory / "summary.json"))
    if incomplete:
        raise RuntimeError("incomplete required evaluations: " + "; ".join(incomplete))
    return found


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def wilson_interval(successes: int, trials: int = 36, z: float = 1.96) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    half_width = z * np.sqrt(
        proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
    ) / denominator
    return trials * (center - half_width), trials * (center + half_width)


def strict_success_plot(runs: OrderedDict[str, tuple[Path, list[dict], dict]]) -> None:
    labels = list(runs)
    values = [runs[label][2]["tasks_passed"] for label in labels]
    colors = ["#6B7280" if "base" in label else "#0072B2" for label in labels]
    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    intervals = [wilson_interval(value) for value in values]
    errors = np.array([
        [value - interval[0] for value, interval in zip(values, intervals)],
        [interval[1] - value for value, interval in zip(values, intervals)],
    ])
    bars = ax.bar(
        np.arange(len(labels)), values, color=colors, width=.68,
        yerr=errors, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#374151"},
    )
    ax.set_ylabel("Strict successes (out of 36)")
    ax.set_title("Native Qwen Code performance on all held-out tasks")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=28, ha="right")
    ax.set_ylim(0, 38)
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .45, f"{value}/36", ha="center", fontsize=8.5)
    save(fig, "strict-success-by-checkpoint")


def diagnostic_plot(runs: OrderedDict[str, tuple[Path, list[dict], dict]]) -> None:
    gates = OrderedDict([
        ("Requested edit", lambda r: r["score"]["contract_ok"]),
        ("Preserved rest", lambda r: r["score"]["preservation_ok"]),
        ("Static evidence", lambda r: r["score"]["evidence_ok"]),
        ("Verifier observed", lambda r: r["score"]["agent_validation_observed"]),
        ("Stayed in scope", lambda r: r["score"]["native_scope_ok"]),
        ("Strict success", lambda r: r["score"]["passed"]),
    ])
    labels = list(runs)
    x = np.arange(len(labels))
    width = .78 / len(gates)
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    palette = ["#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#7F8C8D", "#0072B2"]
    for index, ((gate, predicate), color) in enumerate(zip(gates.items(), palette)):
        values = [sum(predicate(row) for row in runs[label][1]) for label in labels]
        ax.bar(x - .39 + width/2 + index*width, values, width, label=gate, color=color)
    ax.set_ylabel("Tasks passing gate (out of 36)")
    ax.set_title("Diagnostic gates separate file correctness from agent protocol")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylim(0, 39)
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=3, frameon=False, fontsize=8.5, title="Independently scored requirement")
    save(fig, "diagnostic-gates")


def category_plot(runs: OrderedDict[str, tuple[Path, list[dict], dict]]) -> None:
    selected = [
        label for label in
        ("4B base", "4B LoRA", "9B base", "9B LoRA", "27B base", "27B LoRA")
        if label in runs
    ]
    categories = sorted({row["category"] for label in selected for row in runs[label][1]})
    matrix = np.array([
        [sum(row["score"]["passed"] for row in runs[label][1] if row["category"] == category)
         for category in categories]
        for label in selected
    ])
    display = [name.replace("_", " ").title() for name in categories]
    fig, ax = plt.subplots(figsize=(10.4, 4.5))
    image = ax.imshow(matrix, vmin=0, vmax=4, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(categories)), display, rotation=32, ha="right")
    ax.set_yticks(np.arange(len(selected)), selected)
    ax.set_title("Strict successes within each task category (four tasks per cell)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]}/4", ha="center", va="center",
                    color="white" if matrix[i,j] >= 3 else "#111827", fontsize=8.5)
    colorbar = fig.colorbar(image, ax=ax, fraction=.025, pad=.02)
    colorbar.set_label("Successful tasks")
    save(fig, "category-results")


def temperature_plot() -> None:
    available = []
    for temperature, dirname in TEMPERATURES.items():
        path = STUDY / dirname / "summary.json"
        if path.is_file() and read_json(path).get("tasks_completed") == 36:
            available.append((temperature, read_json(path)["tasks_passed"]))
    if len(available) != len(TEMPERATURES):
        raise RuntimeError(f"temperature sweep is incomplete: {len(available)}/{len(TEMPERATURES)}")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x, y = zip(*available)
    ax.plot(x, y, marker="o", linewidth=2, markersize=7, color="#0072B2")
    for temperature, passed in available:
        ax.annotate(f"{passed}/36", (temperature, passed), xytext=(0, 8),
                    textcoords="offset points", ha="center")
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("Strict successes (out of 36)")
    ax.set_title("Temperature study: 4B LoRA, seed 1, top-p 0.95")
    ax.set_ylim(0, 38)
    ax.grid(alpha=.2)
    save(fig, "temperature-study")


def controlled_pair_plot() -> None:
    values = []
    for label, dirname in CONTROLLED_PAIR.items():
        path = STUDY / dirname / "summary.json"
        if path.is_file() and read_json(path).get("tasks_completed") == 36:
            values.append((label, read_json(path)["tasks_passed"]))
    if len(values) != len(CONTROLLED_PAIR):
        raise RuntimeError(f"controlled 4B pair is incomplete: {len(values)}/{len(CONTROLLED_PAIR)}")
    labels, successes = zip(*values)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    bars = ax.bar(labels, successes, color=["#6B7280", "#0072B2"])
    ax.set_ylabel("Strict successes (out of 36)")
    ax.set_title("Matched 4B comparison at temperature zero")
    ax.set_ylim(0, 38)
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, successes):
        ax.text(bar.get_x()+bar.get_width()/2, value+.4, f"{value}/36", ha="center")
    save(fig, "controlled-4b-pair")


def thinking_plot() -> None:
    specs = (
        ("Base\nthinking on", "4b-base-validation-full36"),
        ("Base\nthinking off", "4b-base-nothink-v4-validation-full36"),
        ("LoRA\nthinking on", "4b-sft-validation-full36"),
        ("LoRA\nthinking off", "4b-sft-nothink-v4-validation-full36"),
    )
    available = []
    for label, dirname in specs:
        path = STUDY / dirname / "summary.json"
        if path.is_file() and read_json(path).get("tasks_completed") == 36:
            available.append((label, read_json(path)["tasks_passed"]))
    if len(available) != len(specs):
        raise RuntimeError(f"thinking ablation is incomplete: {len(available)}/{len(specs)}")
    labels, values = zip(*available)
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    bars = ax.bar(labels, values, color=["#6B7280", "#A7AFB8", "#0072B2", "#56B4E9"])
    ax.set_ylabel("Strict successes (out of 36)")
    ax.set_title("Inference-time thinking ablation: Qwen3.5-4B")
    ax.set_ylim(0, 38)
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.4, f"{value}/36", ha="center")
    save(fig, "thinking-ablation")


def throughput_plot() -> None:
    points = []
    for count in (1, 2, 4):
        path = (
            ROOT / "artifacts/checkpoints"
            / f"throughput-native-qwen35-0.8b-a100-{count}" / "throughput.json"
        )
        if path.is_file():
            rate = read_json(path)["summary"]["median_sequence_tokens_per_second"]
            points.append((count, rate))
    if len(points) != 3:
        raise RuntimeError(f"throughput sweep is incomplete: {len(points)}/3")
    x, y = zip(*points)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(x, y, marker="o", linewidth=2, markersize=7, color="#009E73")
    ax.plot(x, [y[0]*value for value in x], linestyle="--", color="#9CA3AF",
            label="Ideal linear scaling from one A100")
    for count, rate in points:
        ax.annotate(f"{rate:,.0f}", (count, rate), xytext=(0, 8),
                    textcoords="offset points", ha="center")
    ax.set_xticks(x)
    ax.set_xlabel("A100 GPUs in one training job")
    ax.set_ylabel("Median non-padding tokens/s")
    ax.set_title("VERL training throughput on the native SFT snapshot")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save(fig, "a100-throughput")


def training_plot() -> None:
    fig, ax = plt.subplots(figsize=(8.7, 4.5))
    for label, dirname in CHECKPOINTS.items():
        path = ROOT / "artifacts/checkpoints" / dirname / "metrics.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing training metrics for {label}: {path}")
        metrics = read_jsonl(path)
        train = [(row["step"], row["data"]["train/loss"])
                 for row in metrics if "train/loss" in row["data"]]
        validation = [(row["step"], row["data"]["val/loss"])
                      for row in metrics if "val/loss" in row["data"]]
        line, = ax.plot(*zip(*train), linewidth=1.6, label=label)
        if validation:
            ax.scatter(*zip(*validation), marker="D", s=35, color=line.get_color(), zorder=3)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Assistant-token loss (lower is better)")
    ax.set_title("Training curves; diamonds mark held-out loss")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=2, title="Checkpoint")
    save(fig, "training-curves")


def hyperparameter_plot() -> None:
    points = []
    for label, (run_dir, checkpoint_dir) in HYPERPARAMETERS.items():
        summary_path = STUDY / run_dir / "summary.json"
        metrics_path = ROOT / "artifacts/checkpoints" / checkpoint_dir / "metrics.jsonl"
        if not summary_path.is_file() or not metrics_path.is_file():
            continue
        summary = read_json(summary_path)
        if summary.get("tasks_completed") != 36:
            continue
        validation = [
            row["data"]["val/loss"] for row in read_jsonl(metrics_path)
            if "val/loss" in row["data"]
        ]
        if validation:
            points.append((label, summary["tasks_passed"], validation[-1]))
    if len(points) != len(HYPERPARAMETERS):
        raise RuntimeError(
            f"hyperparameter evaluation is incomplete: {len(points)}/{len(HYPERPARAMETERS)}"
        )
    labels, success, loss = zip(*points)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    bars = axes[0].bar(labels, success, color="#0072B2")
    axes[0].set_ylabel("Strict successes (out of 36)")
    axes[0].set_ylim(0, 38)
    axes[0].grid(axis="y", alpha=.2)
    for bar, value in zip(bars, success):
        axes[0].text(bar.get_x()+bar.get_width()/2, value+.4, f"{value}/36", ha="center")
    bars = axes[1].bar(labels, loss, color="#E69F00")
    axes[1].set_ylabel("Held-out assistant-token loss")
    axes[1].grid(axis="y", alpha=.2)
    for bar, value in zip(bars, loss):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+.003, f"{value:.3f}", ha="center")
    fig.suptitle("4B LoRA hyperparameters: imitation loss versus execution")
    save(fig, "hyperparameter-ablation")


def failure_kind(row: dict, kind: str) -> bool:
    score = row["score"]
    return {
        "Requested edit": not score["contract_ok"],
        "Preservation": not score["preservation_ok"],
        "Static evidence": not score["evidence_ok"],
        "Verifier protocol": not score["agent_validation_observed"],
        "Workspace scope": not score["native_scope_ok"],
        "Harness completion": row["status"] != "completed",
    }[kind]


def tex_escape(value: str) -> str:
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
             "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(table.get(char, char) for char in value.replace("\n", " "))


def failure_examples(runs: OrderedDict[str, tuple[Path, list[dict], dict]]) -> None:
    kinds = ("Requested edit", "Preservation", "Static evidence", "Verifier protocol",
             "Workspace scope", "Harness completion")
    lines = []
    for label, (_, rows, _) in runs.items():
        for kind in kinds:
            example = next((row for row in rows if failure_kind(row, kind)), None)
            if example is None:
                detail = "No failure of this type was observed."
                task = "---"
            else:
                task = example["task_id"]
                reasons = example["score"].get("reasons") or []
                detail = "; ".join(reasons) or f"native CLI status: {example['status']}"
            lines.append(
                f"{tex_escape(label)} & {tex_escape(kind)} & {tex_escape(task)} & "
                f"{tex_escape(detail[:240])} " + "\\\\" + "\n"
            )
    (OUTPUT / "failure-examples.tex").write_text(
        "\\newcommand{\\failureexamplerows}{%\n" + "".join(lines) + "}\n",
        encoding="utf-8",
    )


def export_summary(runs: OrderedDict[str, tuple[Path, list[dict], dict]]) -> None:
    with (OUTPUT / "checkpoint-results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("checkpoint", "completed", "strict_successes", "strict_rate", "cli_failures"))
        for label, (_, _, summary) in runs.items():
            writer.writerow((label, summary["tasks_completed"], summary["tasks_passed"],
                             summary["pass_rate"], summary["cli_failures"]))
    result_lines = []
    for label, (_, rows, summary) in runs.items():
        thinking = 0
        tool_calls = 0
        for row in rows:
            tool_calls += sum(row.get("native_tool_counts", {}).values())
            for event in read_jsonl(STUDY / RUNS[label] / row["event_file"]):
                if event.get("type") == "assistant":
                    thinking += sum(
                        block.get("type") == "thinking"
                        for block in (event.get("message") or {}).get("content") or []
                        if isinstance(block, dict)
                    )
        result_lines.append(
            f"{tex_escape(label)} & 36 & {summary['tasks_passed']} "
            f"({100*summary['pass_rate']:.1f}\\%) & {summary['cli_failures']} & "
            f"{tool_calls} & {thinking} " + "\\\\" + "\n"
        )
    (OUTPUT / "checkpoint-results.tex").write_text(
        "\\newcommand{\\checkpointresultrows}{%\n" + "".join(result_lines) + "}\n",
        encoding="utf-8",
    )

    training_meta = {
        "0.5B full SFT": ("Full", "$10^{-5}$", 1, 16),
        "0.8B LoRA": ("LoRA r=16", "$10^{-4}$", 4, 16),
        "0.8B full SFT": ("Full", "$10^{-5}$", 1, 16),
        "2B full SFT": ("Full", "$10^{-5}$", 1, 16),
        "4B LoRA": ("LoRA r=16", "$10^{-4}$", 4, 16),
        "4B LoRA, LR 1e-5": ("LoRA r=16", "$10^{-5}$", 1, 16),
        "4B LoRA, rank 8": ("LoRA r=8", "$10^{-4}$", 1, 16),
        "9B LoRA": ("LoRA r=16", "$10^{-5}$", 2, 16),
        "27B LoRA": ("LoRA r=16", "$10^{-4}$", 16, 8),
    }
    training_lines = []
    for label, dirname in CHECKPOINTS.items():
        path = ROOT / "artifacts/checkpoints" / dirname / "metrics.jsonl"
        if not path.is_file():
            continue
        rows = read_jsonl(path)
        validation = [row["data"]["val/loss"] for row in rows if "val/loss" in row["data"]]
        memory = max(row["data"].get("perf/max_memory_allocated_gb", 0) for row in rows)
        method, lr, gpus, steps = training_meta[label]
        training_lines.append(
            f"{tex_escape(label)} & {method} & {lr} & {gpus} & {steps} & "
            f"{validation[-1]:.3f} & {memory:.1f} " + "\\\\" + "\n"
        )
    (OUTPUT / "training-results.tex").write_text(
        "\\newcommand{\\trainingresultrows}{%\n" + "".join(training_lines) + "}\n",
        encoding="utf-8",
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = completed_runs()
    if not runs:
        raise RuntimeError(f"no complete 36-task runs found under {STUDY}")
    strict_success_plot(runs)
    diagnostic_plot(runs)
    category_plot(runs)
    temperature_plot()
    controlled_pair_plot()
    thinking_plot()
    throughput_plot()
    training_plot()
    hyperparameter_plot()
    failure_examples(runs)
    export_summary(runs)


if __name__ == "__main__":
    main()
