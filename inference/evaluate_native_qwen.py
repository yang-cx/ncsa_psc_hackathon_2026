#!/usr/bin/env python3
"""Run Qwen as a generic-tool agent and score held-out TRExFitter tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.model_runtime import (  # noqa: E402
    generate_chat,
    load_model,
    parse_generated_tool_calls,
)
from inference.run_native_agent_study import (  # noqa: E402
    DEFAULT_DATASET,
    VALIDATOR_COMMAND,
    materialize_workspace,
    score_task,
    scoreable_tasks,
)
from trex_fitter.patch_apply import apply_patch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", help="Required for an adapter that does not identify its base")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--prompt-jsonl", type=Path,
        help="Optional immutable main-agent prompt snapshot; defaults to DATASET_ROOT/data/main-agent/SPLIT.jsonl",
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--resume", action="store_true",
        help="Keep completed task rows, run only missing tasks, and rebuild the summary",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_rows(
    dataset_root: Path, split: str, prompt_jsonl: Path | None = None,
) -> dict[str, dict[str, Any]]:
    path = prompt_jsonl.resolve() if prompt_jsonl is not None else dataset_root / f"data/main-agent/{split}.jsonl"
    rows = [
        row
        for row in read_jsonl(path)
        if row.get("modality") == "deterministic_qwen_code_tools"
    ]
    result = {row["logical_task_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path}: duplicate native Qwen Code logical task IDs")
    return result


def _config_path(workspace: Path, value: str = "analysis.config") -> Path:
    if value not in {"analysis.config", "./analysis.config"}:
        raise ValueError("only analysis.config is accessible")
    return workspace / "analysis.config"


def _command_result(process: subprocess.CompletedProcess[str]) -> str:
    output = f"{process.stdout}{process.stderr}".rstrip()
    return f"Exit code: {process.returncode}" + (f"\n\n{output}" if output else "")


def _run_shell(arguments: dict[str, Any], workspace: Path) -> tuple[str, bool]:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("shell.command must be a non-empty string")
    if VALIDATOR_COMMAND.fullmatch(command):
        parts = shlex.split(command)
        parts[0] = sys.executable
    else:
        if re.search(r"[;&|`$<>]", command):
            raise ValueError("shell control operators are not allowed")
        parts = shlex.split(command)
        if not parts or Path(parts[0]).name not in {"sed", "rg", "grep", "head", "tail", "wc", "git"}:
            raise ValueError("only bounded read commands and the Python verifier are allowed")
        if any(Path(part).is_absolute() or ".." in Path(part).parts for part in parts[1:]):
            raise ValueError("shell paths must remain workspace-relative")
        if Path(parts[0]).name == "git" and parts[1:3] not in (["diff", "--"], ["status", "--short"]):
            raise ValueError("only git diff/status inspection is allowed")
    timeout = min(max(int(arguments.get("timeout_ms", 30000)), 1), 30000) / 1000
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    process = subprocess.run(
        parts, cwd=workspace, env=environment, text=True, capture_output=True,
        timeout=timeout, check=False,
    )
    return _command_result(process), bool(VALIDATOR_COMMAND.fullmatch(command) and process.returncode == 0)


def execute_tool(name: str, arguments: dict[str, Any], workspace: Path) -> tuple[str, bool]:
    config = workspace / "analysis.config"
    if name == "apply_patch":
        patch = arguments.get("patch")
        if not isinstance(patch, str):
            raise ValueError("apply_patch.patch must be a string")
        before = config.read_text(encoding="utf-8")
        after = apply_patch(before, patch)
        config.write_text(after, encoding="utf-8")
        return ("Patch applied successfully." if after != before else "Patch made no changes."), False
    if name == "shell":
        return _run_shell(arguments, workspace)
    if name == "read_file":
        path = _config_path(workspace, str(arguments.get("path", "")))
        offset = max(int(arguments.get("offset", 0)), 0)
        limit = min(max(int(arguments.get("limit", 200)), 1), 400)
        lines = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[offset:offset + limit]), False
    if name == "search_files":
        mode = arguments.get("mode")
        query = arguments.get("query")
        if not isinstance(query, str):
            raise ValueError("search_files.query must be a string")
        if mode == "content":
            path = _config_path(workspace, str(arguments.get("path", "analysis.config")))
            try:
                pattern = re.compile(query)
            except re.error:
                pattern = re.compile(re.escape(query))
            matches = [
                f"{number}:{line}" for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                if pattern.search(line)
            ]
            return "\n".join(matches[:200]), False
        if mode == "path_glob":
            if Path(query).is_absolute() or ".." in Path(query).parts:
                raise ValueError("path glob must remain workspace-relative")
            return "\n".join(path.name for path in workspace.glob(query) if path.is_file())[:20000], False
        if mode == "directory":
            if query not in {".", ""}:
                raise ValueError("only the workspace root may be listed")
            return "\n".join(sorted(path.name for path in workspace.iterdir()))[:20000], False
        raise ValueError("search_files.mode must be content, path_glob, or directory")
    raise ValueError(f"unknown tool: {name}")


def assistant_content(response: str) -> str:
    text = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL)
    text = re.sub(r"<\|[^|<>\s]+\|>", "", text)
    return text.strip()


def summarize(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    strict_passes = sum(row.get("passed", row["score"]["passed"] and row["verifier_observed"]) for row in rows)
    return {
        **metadata,
        "tasks": len(rows),
        "passed": strict_passes,
        "pass_rate": strict_passes / len(rows) if rows else None,
        "correct_final_config": sum(row["score"]["passed"] for row in rows),
        "verifier_observed": sum(row["verifier_observed"] for row in rows),
        "tool_errors": sum(row["tool_errors"] for row in rows),
        "tool_counts": dict(sorted(sum((Counter(row["tool_counts"]) for row in rows), Counter()).items())),
        "by_domain": {
            domain: {
                "tasks": len(group),
                "passed": sum(
                    row.get("passed", row["score"]["passed"] and row["verifier_observed"])
                    for row in group
                ),
            }
            for domain in sorted({row["domain"] for row in rows})
            for group in [[row for row in rows if row["domain"] == domain]]
        },
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.max_turns < 1 or args.max_new_tokens < 1 or args.temperature < 0:
        raise ValueError("invalid generation limit or temperature")

    dataset_root = args.dataset_root.resolve()
    prompts = prompt_rows(dataset_root, args.split, args.prompt_jsonl)
    tasks = [task for task in scoreable_tasks(dataset_root) if task["split"] == args.split]
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [task for task in tasks if task["task_id"] in wanted]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    missing = {task["task_id"] for task in tasks} - prompts.keys()
    if missing:
        raise ValueError(f"missing canonical prompt rows: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.jsonl"
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    results = read_jsonl(output) if output.exists() else []
    completed_ids = {row["task_id"] for row in results}
    if len(completed_ids) != len(results):
        raise ValueError(f"{output}: duplicate task IDs")
    tasks = [task for task in tasks if task["task_id"] not in completed_ids]

    model = tokenizer = None
    resolved, device = args.checkpoint, args.device
    if tasks:
        model, tokenizer, resolved, device = load_model(
            args.checkpoint, args.device, base_model=args.base_model
        )
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    for task in tasks:
        row = prompts[task["task_id"]]
        started = time.perf_counter()
        tool_counts: Counter[str] = Counter()
        tool_errors = 0
        verifier_observed = False
        raw_responses: list[str] = []
        messages = [dict(message) for message in row["messages"][:2]]
        with tempfile.TemporaryDirectory(prefix=f"qwen-{task['task_id']}-") as temporary:
            workspace = Path(temporary) / "workspace"
            materialize_workspace(task, dataset_root, workspace, "qwen")
            last_patch_turn = -1
            last_verifier_turn = -1
            for turn in range(args.max_turns):
                response = generate_chat(
                    model, tokenizer, messages, tools=row["tools"],
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=args.top_p, enable_thinking=False,
                )
                raw_responses.append(response)
                try:
                    calls = parse_generated_tool_calls(model, tokenizer, response)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    calls = []
                    tool_errors += 1
                if not calls:
                    messages.append({"role": "assistant", "content": assistant_content(response)})
                    break
                assistant = {
                    "role": "assistant",
                    "content": assistant_content(response),
                    "tool_calls": [],
                }
                for index, call in enumerate(calls):
                    call["id"] = f"turn-{turn}-call-{index}"
                    assistant["tool_calls"].append(call)
                messages.append(assistant)
                for call in calls:
                    name = call["function"]["name"]
                    tool_counts[name] += 1
                    if name == "apply_patch":
                        last_patch_turn = turn
                    try:
                        content, verified = execute_tool(
                            name, call["function"]["arguments"], workspace
                        )
                    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as error:
                        content = f"Tool error: {type(error).__name__}: {error}"
                        verified = False
                        tool_errors += 1
                    if verified:
                        last_verifier_turn = turn
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            verifier_observed = last_verifier_turn >= last_patch_turn >= 0
            score = score_task(
                task,
                dataset_root / task["state_before"],
                workspace / "analysis.config",
            )
        result = {
            "task_id": task["task_id"],
            "domain": row["domain"],
            "score": score,
            # Strict trajectory success requires both a correct final state and
            # an observed successful verifier call after the last patch.
            "passed": score["passed"] and verifier_observed,
            "verifier_observed": verifier_observed,
            "tool_errors": tool_errors,
            "tool_counts": dict(tool_counts),
            "elapsed_seconds": time.perf_counter() - started,
            "raw_responses": raw_responses,
            "messages": messages,
        }
        results.append(result)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps({
            "task_id": result["task_id"], "passed": score["passed"],
            "verifier_observed": verifier_observed, "tool_errors": tool_errors,
        }))

    prompt_path = (
        args.prompt_jsonl.resolve()
        if args.prompt_jsonl is not None
        else dataset_root / f"data/main-agent/{args.split}.jsonl"
    )
    metadata = {
        "schema_version": "trexfitter-native-qwen-evaluation/v1",
        "source_dataset": "cxyang-ucb/hyy-sft",
        "dataset_root": str(dataset_root),
        "prompt_jsonl": str(prompt_path),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "checkpoint": resolved,
        "device": device,
        "split": args.split,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    summary = summarize(results, metadata)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
