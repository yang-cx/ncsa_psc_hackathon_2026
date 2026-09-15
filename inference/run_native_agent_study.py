#!/usr/bin/env python3
"""Evaluate TRExFitter tasks through each model family's native coding agent.

This is an experiment harness, not a model-facing tool.  It gives each coding
agent an isolated workspace containing ``analysis.config``, records the CLI's
native JSON event stream, and scores the final file only after the agent exits.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import http.client
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import threading
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/datasets/hyy-trexfitter-agent-trajectories"
sys.path.insert(0, str(PROJECT_ROOT))

from trex_fitter.config_format import ConfigError, _parse_blocks  # noqa: E402
from trex_fitter.config_verify import verify_config  # noqa: E402
from trex_fitter.expression_equivalence import contract_values_equivalent  # noqa: E402


VALIDATOR_COMMAND = re.compile(
    r"^(?:\S*/)?python3?\s+-m\s+trex_fitter\.config_verify\s+"
    r"(?:\./)?analysis\.config\s+--actions(?:=|\s+)n"
    r"(?:\s+--check-inputs)?(?:\s+--evidence-level(?:=|\s+)[SIC])?\s*$"
)
MACHINE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:global|pscratch|home|tmp)/")
QWEN_STUDY_SYSTEM = (
    "You are Qwen Code editing a scientific-analysis repository rooted at /workspace. Use only "
    "the provided native file and shell tools. Make minimal changes, preserve unrelated settings, run the "
    "requested local validation command, and report only evidence actually observed."
)


class QwenSamplingProxy:
    """Transparent localhost proxy that injects documented OpenAI fields."""

    def __init__(self, upstream: str, overrides: dict[str, Any]) -> None:
        parsed = urlsplit(upstream)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("controlled Qwen proxy currently requires an http:// upstream")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._prefix = parsed.path.rstrip("/")
        self.overrides = overrides
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                self._forward(None)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw) if raw else {}
                if self.path.endswith("/chat/completions"):
                    body.update(proxy.overrides)
                self._forward(json.dumps(body).encode("utf-8"))

            def _forward(self, body: bytes | None) -> None:
                connection = http.client.HTTPConnection(proxy._host, proxy._port, timeout=900)
                headers = {
                    key: value for key, value in self.headers.items()
                    if key.lower() not in {"host", "content-length", "connection"}
                }
                if body is not None:
                    headers["Content-Type"] = "application/json"
                    headers["Content-Length"] = str(len(body))
                connection.request(self.command, self.path, body=body, headers=headers)
                response = connection.getresponse()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in {"content-length", "connection", "transfer-encoding"}:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := response.read(64 * 1024):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                connection.close()
                self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}{self._prefix}"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=("codex", "opencode", "qwen"))
    parser.add_argument("--split", default="validation", choices=("train", "validation"),
                        help="Public test contracts are sealed and cannot be scored by this checkout")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workspace-root", type=Path,
        help="Optional scratch directory for disposable task workspaces",
    )
    parser.add_argument("--model", required=True, help="Explicit native CLI model name for reproducibility")
    parser.add_argument(
        "--qwen-base-url",
        help="OpenAI-compatible Qwen endpoint, for example http://localhost:8000/v1",
    )
    parser.add_argument(
        "--qwen-max-output-tokens", type=int, default=4096,
        help="Fixed Qwen Code per-turn output reservation (default: 4096)",
    )
    parser.add_argument(
        "--qwen-temperature", type=float,
        help="Qwen Code sampling temperature; omitted to use the provider/backend default",
    )
    parser.add_argument(
        "--qwen-top-p", type=float,
        help="Qwen Code nucleus-sampling probability; omitted to use the provider/backend default",
    )
    parser.add_argument(
        "--qwen-seed", type=int,
        help="Qwen Code sampling seed; omitted to use the provider/backend default",
    )
    parser.add_argument(
        "--qwen-thinking", choices=("default", "enabled", "disabled"), default="default",
        help="Explicitly enable/disable Qwen thinking, or retain the serving default",
    )
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh"),
        help="Codex reasoning effort; recorded in the study manifest (Codex only)",
    )
    parser.add_argument("--task-id", action="append", default=[], help="Run only this task; repeat as needed")
    parser.add_argument("--limit", type=int, help="Run at most this many selected tasks")
    parser.add_argument("--timeout", type=int, default=900, help="Per-task CLI timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Materialize workspaces without calling a model")
    parser.add_argument("--resume", action="store_true", help="Skip task IDs already present in results.jsonl")
    parser.add_argument(
        "--rescore-only", action="store_true",
        help="Recompute scores from saved configs/events without invoking the model",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scoreable_tasks(dataset_root: Path) -> list[dict[str, Any]]:
    """Load the current scored task contracts."""
    tasks = read_jsonl(dataset_root / "data/tasks.jsonl")
    source_dir = dataset_root / "data/native/source_tasks"
    for path in sorted(source_dir.glob("*.jsonl")):
        tasks.extend(read_jsonl(path))
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs across source-task families")
    return tasks


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_map(path: Path) -> dict[tuple[str, str, str], str]:
    result: dict[tuple[str, str, str], str] = {}
    for block in _parse_blocks(path):
        for setting, value in block.values.items():
            key = (block.kind, block.name, setting)
            if key in result:
                raise ConfigError(f"duplicate semantic key {key}")
            result[key] = value
    return result


def native_prompt_for(dataset_root: Path, split: str, task_id: str) -> str:
    """Load the exact harness-neutral prompt whose native run is being captured."""
    path = dataset_root / f"data/native/tasks/{split}.jsonl"
    for row in read_jsonl(path):
        if row["id"] == task_id:
            return row["prompt"]
    raise KeyError(f"native task not found: {task_id} in {path}")


def opencode_config() -> dict[str, Any]:
    """Allow only native local file operations and bounded task commands."""
    return {
        "$schema": "https://opencode.ai/config.json",
        "permission": {
            "*": "deny",
            "read": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "edit": {"*": "deny", "analysis.config": "allow", "**/analysis.config": "allow"},
            "bash": {
                "*": "deny",
                "pwd": "allow",
                "python -m trex_fitter.config_verify *": "allow",
                "python3 -m trex_fitter.config_verify *": "allow",
                "python -m trex_fitter.runner *": "allow",
                "python3 -m trex_fitter.runner *": "allow",
            },
            "external_directory": "deny",
            "task": "deny",
            "skill": "deny",
            "webfetch": "deny",
            "websearch": "deny",
        },
    }


def materialize_workspace(task: dict[str, Any], dataset_root: Path, workspace: Path, harness: str) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    source = (dataset_root / task["state_before"]).resolve()
    if not source.is_file() or dataset_root.resolve() not in source.parents:
        raise ValueError(f"invalid state_before for {task['task_id']}: {task['state_before']}")
    shutil.copyfile(source, workspace / "analysis.config")
    # Keep agent `git diff`/`git status` calls inside the isolated task instead
    # of allowing Git to discover the parent research repository and its
    # unrelated submodules.
    subprocess.run(["git", "init", "--quiet"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "--", "analysis.config"], cwd=workspace, check=True)
    if harness == "opencode":
        (workspace / "opencode.json").write_text(
            json.dumps(opencode_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def cli_command(
    harness: str,
    workspace: Path,
    prompt: str,
    model: str | None,
    reasoning_effort: str | None = None,
    qwen_base_url: str | None = None,
) -> list[str]:
    workspace = workspace.resolve()
    if harness == "codex":
        command = [
            "codex", "--ask-for-approval", "never", "exec", "--json", "--color", "never",
            "--sandbox", "workspace-write", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "--cd", str(workspace),
        ]
        if model:
            command.extend(("--model", model))
        if reasoning_effort:
            command.extend(("--config", f'model_reasoning_effort="{reasoning_effort}"'))
        return [*command, prompt]
    if reasoning_effort:
        raise ValueError("--reasoning-effort is supported only by the Codex harness")
    if harness == "opencode":
        command = ["opencode", "run", "--pure", "--format", "json", "--dir", str(workspace)]
        if model:
            command.extend(("--model", model))
        return [*command, prompt]
    if harness == "qwen":
        # Qwen Code owns the tool descriptions, call parser, execution loop,
        # and tool-result messages.  The allowlist names only Qwen Code's
        # built-ins; no repository-defined patch function is exposed.
        runtime_system = QWEN_STUDY_SYSTEM.replace("/workspace", str(workspace))
        command = ["qwen", "--bare"]
        command.extend([
            "--auth-type", "openai",
            # Workspaces contain one copied config and are discarded after
            # scoring, so native YOLO approval cannot affect source data.
            "--approval-mode", "yolo",
            "--system-prompt", runtime_system,
            "--core-tools", ",".join((
                "read_file", "edit",
                "run_shell_command(python -m trex_fitter.config_verify)",
                "run_shell_command(python3 -m trex_fitter.config_verify)",
                "run_shell_command(sed)", "run_shell_command(rg)",
                "run_shell_command(grep)", "run_shell_command(head)",
                "run_shell_command(tail)", "run_shell_command(wc)",
                "run_shell_command(find)", "run_shell_command(git diff)",
                "run_shell_command(git status)",
            )),
            "--exclude-tools", "get_goal,update_goal,notebook_edit",
            "--allowed-tools", ",".join((
                "Read", "Edit(/analysis.config)",
                "Bash(python -m trex_fitter.config_verify)",
                "Bash(python3 -m trex_fitter.config_verify)",
                "Bash(sed *)", "Bash(rg *)", "Bash(grep *)", "Bash(head *)",
                "Bash(tail *)", "Bash(wc *)", "Bash(find *)",
                "Bash(git diff *)", "Bash(git status *)",
            )),
            "--max-session-turns", "20",
            "--max-tool-calls", "50",
            "--output-format", "stream-json",
        ])
        if model:
            command.extend(("--model", model))
        if qwen_base_url:
            command.extend(("--openai-base-url", qwen_base_url))
        return [*command, "--prompt", prompt]
    raise ValueError(f"unsupported native harness: {harness}")


def safe_environment(harness: str, qwen_max_output_tokens: int = 4096) -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = PROJECT_ROOT / ".venv/bin"
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{previous}" if previous else str(PROJECT_ROOT)
    if harness == "qwen":
        # Local vLLM/Ollama endpoints normally require a syntactically present
        # key even when they do not authenticate it.
        env.setdefault("OPENAI_API_KEY", "not-needed")
        # Qwen Code otherwise reserves the model's full declared 32k output
        # budget. Self-hosted servers correctly reject prompt + 32k when it
        # exceeds their context window, before the model can emit a token.
        env["QWEN_CODE_MAX_OUTPUT_TOKENS"] = str(qwen_max_output_tokens)
    return env


def parse_events(stdout: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return [event for event in value if isinstance(event, dict)]
    if isinstance(value, dict):
        return [value]
    events = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def qwen_tool_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join Qwen Code tool_use blocks to their native tool_result blocks."""
    results: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, event in enumerate(events):
        if event.get("type") != "user":
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results[str(block.get("tool_use_id", ""))] = (index, block)

    records = []
    for index, event in enumerate(events):
        if event.get("type") != "assistant":
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            call_id = str(block.get("id", ""))
            result_index, result = results.get(call_id, (-1, {}))
            records.append({
                "call_index": index,
                "result_index": result_index,
                "id": call_id,
                "name": str(block.get("name", "")),
                "input": block.get("input") or {},
                "result": result,
            })
    return records


def qwen_call_succeeded(record: dict[str, Any]) -> bool:
    result = record.get("result") or {}
    return bool(result) and result.get("is_error") is not True


def inner_shell_command(command: str) -> str:
    """Remove the CLI's shell wrapper without interpreting the command."""
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if len(parts) == 3 and parts[0].endswith("bash") and parts[1] == "-lc":
        return parts[2]
    return command


def is_required_validator_command(command: str, *, check_inputs: bool) -> bool:
    command = inner_shell_command(command)
    if not VALIDATOR_COMMAND.fullmatch(command):
        return False
    return not check_inputs or "--check-inputs" in command


def observed_agent_validation(
    harness: str,
    events: list[dict[str, Any]],
    *,
    check_inputs: bool,
) -> bool:
    """Require successful deterministic verification after the last visible edit."""
    edit_indices: list[int] = []
    verifier_indices: list[int] = []
    for index, event in enumerate(events):
        if harness == "codex" and event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "file_change":
                edit_indices.append(index)
            if (
                item.get("type") == "command_execution"
                and item.get("exit_code") == 0
                and is_required_validator_command(item.get("command", ""), check_inputs=check_inputs)
            ):
                verifier_indices.append(index)
        if harness == "opencode" and event.get("type") == "tool_use":
            part = event.get("part") or {}
            state = part.get("state") or {}
            metadata = state.get("metadata") or {}
            command = (state.get("input") or {}).get("command", "")
            if part.get("tool") in {"edit", "apply_patch"} and state.get("status") == "completed":
                edit_indices.append(index)
            if (
                part.get("tool") == "bash"
                and state.get("status") == "completed"
                and metadata.get("exit", 0) == 0
                and is_required_validator_command(command, check_inputs=check_inputs)
            ):
                verifier_indices.append(index)
    if harness == "qwen":
        for record in qwen_tool_records(events):
            name = record["name"]
            arguments = record["input"]
            completed_index = record["result_index"]
            if name == "edit" and qwen_call_succeeded(record):
                edit_indices.append(completed_index)
            if (
                name == "run_shell_command"
                and qwen_call_succeeded(record)
                # Foreground execution is Qwen Code's native default; models
                # may therefore omit this optional schema property.
                and arguments.get("is_background", False) is False
                and is_required_validator_command(
                    str(arguments.get("command", "")), check_inputs=check_inputs
                )
                and "VALID:" in str((record.get("result") or {}).get("content", ""))
            ):
                verifier_indices.append(completed_index)
    if not verifier_indices:
        return False
    return not edit_indices or max(verifier_indices) > max(edit_indices)


def _outside_workspace(path_value: str, workspace: Path) -> bool:
    candidate = Path(path_value)
    if not candidate.is_absolute() and ".." not in candidate.parts:
        return False
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    return resolved != workspace and workspace not in resolved.parents


def shell_references_outside_workspace(command: str, workspace: Path) -> bool:
    """Conservatively flag parent/machine paths after allowing this workspace.

    Native agents often pass the absolute disposable-workspace path back to a
    shell tool.  That is in scope even on NERSC, where it begins with
    ``/global``; only other machine paths should fail the scope gate.
    """
    command_without_workspace = command.replace(str(workspace.resolve()), ".")
    return bool(
        MACHINE_PATH.search(command_without_workspace)
        or re.search(r"(?:^|\s)\.\.(?:/|\s|$)", command_without_workspace)
    )


def native_scope_violations(
    harness: str,
    events: list[dict[str, Any]],
    workspace: Path,
) -> list[str]:
    """Detect observable attempts to escape the one-file task workspace."""
    violations: set[str] = set()
    workspace = workspace.resolve()
    for event in events:
        if harness == "codex" and event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "command_execution":
                command = inner_shell_command(item.get("command", ""))
                if shell_references_outside_workspace(command, workspace):
                    violations.add("shell command referenced a path outside the task workspace")
            elif item.get("type") == "file_change":
                for change in item.get("changes") or []:
                    if _outside_workspace(str(change.get("path", "")), workspace):
                        violations.add("file change targeted a path outside the task workspace")
        if harness == "opencode" and event.get("type") == "tool_use":
            part = event.get("part") or {}
            state = part.get("state") or {}
            if state.get("status") not in {"completed", "error"}:
                continue
            tool = str(part.get("tool", ""))
            arguments = state.get("input") or {}
            if tool in {"task", "skill", "webfetch", "websearch"}:
                violations.add(f"forbidden native tool used: {tool}")
            if tool == "bash":
                command = str(arguments.get("command", ""))
                if shell_references_outside_workspace(command, workspace):
                    violations.add("shell command referenced a path outside the task workspace")
            for key in ("filePath", "path"):
                value = arguments.get(key)
                if isinstance(value, str) and _outside_workspace(value, workspace):
                    violations.add(f"{tool} referenced a path outside the task workspace")
    if harness == "qwen":
        allowed = {"read_file", "edit", "run_shell_command"}
        for record in qwen_tool_records(events):
            tool = record["name"]
            arguments = record["input"]
            if tool not in allowed:
                violations.add(f"forbidden native tool used: {tool}")
            if not qwen_call_succeeded(record):
                # Keep failed calls as recovery evidence. They did not read,
                # modify, or execute anything, so their malformed arguments
                # are protocol errors rather than workspace escapes.
                continue
            if tool == "run_shell_command":
                command = str(arguments.get("command", ""))
                if shell_references_outside_workspace(command, workspace):
                    violations.add("shell command referenced a path outside the task workspace")
            for key in ("file_path", "path", "directory"):
                value = arguments.get(key)
                if isinstance(value, str) and _outside_workspace(value, workspace):
                    violations.add(f"{tool} referenced a path outside the task workspace")
            if tool == "edit":
                path_value = arguments.get("file_path")
                # A schema-invalid empty path is a recoverable tool error, not
                # an attempt to touch another file. Preserve it in the trace
                # as recovery supervision while still rejecting real targets.
                if isinstance(path_value, str) and path_value:
                    target = Path(path_value)
                    resolved = target.resolve() if target.is_absolute() else (workspace / target).resolve()
                    if resolved != workspace / "analysis.config":
                        violations.add("edit targeted a file other than analysis.config")
    return sorted(violations)


def native_tool_counts(harness: str, events: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in events:
        item = event.get("item") or {}
        item_type = item.get("type")
        if harness == "codex" and event.get("type") == "item.completed":
            if item_type == "command_execution":
                counts["shell"] += 1
            elif item_type == "file_change":
                counts["apply_patch"] += 1
        if harness == "opencode":
            part = event.get("part") or {}
            tool = part.get("tool")
            if tool and (part.get("state", {}).get("status") in {"completed", "error"} or event.get("type") == "tool"):
                counts[str(tool)] += 1
    if harness == "qwen":
        counts.update(record["name"] for record in qwen_tool_records(events) if record["name"])
    return dict(sorted(counts.items()))


def score_task(task: dict[str, Any], before: Path, after: Path) -> dict[str, Any]:
    reasons: list[str] = []
    equivalence_matches: list[str] = []
    contract_ok = preservation_ok = True
    try:
        before_map = semantic_map(before)
        after_map = semantic_map(after)
        allowed = {(x["block"], x["name"], x["setting"]) for x in task["change_contract"]}
        for item in task["change_contract"]:
            key = (item["block"], item["name"], item["setting"])
            actual = after_map.get(key)
            if not contract_values_equivalent(item["setting"], item["expected"], actual):
                contract_ok = False
                reasons.append(f"contract mismatch for {'.'.join(key)}")
            elif actual != item["expected"]:
                equivalence_matches.append(".".join(key))
        unauthorized = sorted(
            ".".join(key) for key in set(before_map) | set(after_map)
            if key not in allowed
            and not contract_values_equivalent(key[2], before_map.get(key, ""), after_map.get(key))
        )
        if unauthorized:
            preservation_ok = False
            reasons.append("unauthorized semantic changes: " + ", ".join(unauthorized))
    except (ConfigError, OSError, ValueError) as error:
        contract_ok = preservation_ok = False
        reasons.append(f"semantic parse failed: {type(error).__name__}: {error}")

    report = verify_config(
        after,
        actions="n",
        check_inputs=task["required_evidence"] == "I",
        project_dir=PROJECT_ROOT,
    )
    evidence_ok = report.inputs_valid is True if task["required_evidence"] == "I" else report.analysis_valid
    if not report.analysis_valid:
        reasons.append("final static config validation failed")
    if not evidence_ok:
        reasons.append(f"required evidence {task['required_evidence']} failed")
    return {
        "passed": contract_ok and preservation_ok and evidence_ok,
        "contract_ok": contract_ok,
        "preservation_ok": preservation_ok,
        "analysis_valid": report.analysis_valid,
        "coffea_compatible": report.coffea_compatible,
        "inputs_valid": report.inputs_valid,
        "evidence_ok": evidence_ok,
        "equivalence_matches": equivalence_matches,
        "reasons": reasons,
    }


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["task_id"] for row in read_jsonl(path)}


def summarize(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] != "prepared"]
    scored = [row for row in completed if "score" in row]
    return {
        **metadata,
        "tasks_recorded": len(rows),
        "tasks_completed": len(completed),
        "tasks_passed": sum(row["score"]["passed"] for row in scored),
        "pass_rate": (sum(row["score"]["passed"] for row in scored) / len(scored)) if scored else None,
        "cli_failures": sum(row["status"] != "completed" for row in completed),
        "by_category": {
            category: {
                "tasks": len(group),
                "passed": sum(row.get("score", {}).get("passed", False) for row in group),
            }
            for category in sorted({row["category"] for row in rows})
            for group in [[row for row in rows if row["category"] == category and row["status"] != "prepared"]]
        },
    }


def rescore_study(
    output_dir: Path,
    dataset_root: Path,
    harness: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Reapply the current independent gates to saved native trajectories."""
    results_path = output_dir / "results.jsonl"
    tasks = {task["task_id"]: task for task in scoreable_tasks(dataset_root)}
    rows = read_jsonl(results_path)
    for record in rows:
        if record["status"] == "prepared":
            continue
        task = tasks[record["task_id"]]
        before = output_dir / "initial" / f"{record['task_id']}.config"
        after = (
            output_dir / record["final_config"]
            if record.get("final_config")
            else output_dir / record["workspace"] / "analysis.config"
        )
        events = parse_events((output_dir / record["event_file"]).read_text(encoding="utf-8"))
        score = score_task(task, before, after)
        observed = observed_agent_validation(
            harness, events, check_inputs=task["required_evidence"] == "I"
        )
        score["agent_validation_observed"] = observed
        if not observed:
            score["passed"] = False
            score["reasons"].append(
                "native trajectory did not contain a successful required config verifier call"
            )
        violations = native_scope_violations(harness, events, (output_dir / record["workspace"]))
        score["native_scope_ok"] = not violations
        if violations:
            score["passed"] = False
            score["reasons"].extend(violations)
        record["score"] = score
    replacement = results_path.with_suffix(".jsonl.rescored")
    replacement.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    replacement.replace(results_path)
    summary = summarize(rows, metadata)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.timeout < 1:
        raise ValueError("--timeout must be positive")
    if args.qwen_max_output_tokens < 1:
        raise ValueError("--qwen-max-output-tokens must be positive")
    if args.qwen_temperature is not None and args.qwen_temperature < 0:
        raise ValueError("--qwen-temperature must be non-negative")
    if args.qwen_top_p is not None and not 0 < args.qwen_top_p <= 1:
        raise ValueError("--qwen-top-p must be in (0, 1]")
    if args.harness == "qwen" and not (args.qwen_base_url or os.environ.get("OPENAI_BASE_URL")):
        raise ValueError("--qwen-base-url or OPENAI_BASE_URL is required for the Qwen native harness")

    dataset_root = args.dataset_root.resolve()
    tasks = [row for row in scoreable_tasks(dataset_root) if row["split"] == args.split]
    if args.task_id:
        wanted = set(args.task_id)
        tasks = [row for row in tasks if row["task_id"] in wanted]
        missing = wanted - {row["task_id"] for row in tasks}
        if missing:
            raise ValueError(f"task IDs not found in split {args.split}: {sorted(missing)}")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("no tasks selected")

    if args.rescore_only:
        if not (args.output_dir / "results.jsonl").is_file():
            raise FileNotFoundError(f"no saved results in {args.output_dir}")
        metadata = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        print(json.dumps(
            rescore_study(args.output_dir, dataset_root, args.harness, metadata),
            indent=2, sort_keys=True,
        ))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = (
        args.workspace_root.resolve()
        if args.workspace_root is not None
        else args.output_dir.resolve() / "workspaces"
    )
    workspace_root.mkdir(parents=True, exist_ok=True)
    qwen_base_url = args.qwen_base_url or os.environ.get("OPENAI_BASE_URL")
    qwen_sampling_overrides: dict[str, Any] = {}
    qwen_proxy = None
    effective_qwen_base_url = qwen_base_url
    if args.harness == "qwen":
        if args.qwen_temperature is not None:
            qwen_sampling_overrides["temperature"] = args.qwen_temperature
        if args.qwen_top_p is not None:
            qwen_sampling_overrides["top_p"] = args.qwen_top_p
        if args.qwen_seed is not None:
            qwen_sampling_overrides["seed"] = args.qwen_seed
        if args.qwen_thinking != "default":
            enabled = args.qwen_thinking == "enabled"
            qwen_sampling_overrides["reasoning_effort"] = "high" if enabled else "none"
            qwen_sampling_overrides["chat_template_kwargs"] = {"enable_thinking": enabled}
    if qwen_sampling_overrides:
        qwen_proxy = QwenSamplingProxy(qwen_base_url, qwen_sampling_overrides)
        qwen_proxy.start()
        atexit.register(qwen_proxy.close)
        effective_qwen_base_url = qwen_proxy.base_url
    results_path = args.output_dir / "results.jsonl"
    already = existing_ids(results_path) if args.resume else set()
    if results_path.exists() and not args.resume:
        raise FileExistsError(f"{results_path} exists; choose a new output directory or pass --resume")

    version = subprocess.run([args.harness, "--version"], text=True, capture_output=True, check=False).stdout.strip()
    metadata = {
        "schema_version": "trexfitter-native-agent-study/v1",
        "source_dataset": "cxyang-ucb/hyy-sft",
        "harness": args.harness,
        "harness_version": version,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "qwen_base_url": qwen_base_url,
        "qwen_max_output_tokens": args.qwen_max_output_tokens if args.harness == "qwen" else None,
        "qwen_temperature": args.qwen_temperature if args.harness == "qwen" else None,
        "qwen_top_p": args.qwen_top_p if args.harness == "qwen" else None,
        "qwen_seed": args.qwen_seed if args.harness == "qwen" else None,
        "qwen_thinking": args.qwen_thinking if args.harness == "qwen" else None,
        "qwen_sampling_proxy_overrides": qwen_sampling_overrides if args.harness == "qwen" else None,
        "split": args.split,
        "workspace_root": str(workspace_root),
        "timeout_seconds": args.timeout,
        "dry_run": args.dry_run,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    with results_path.open("a", encoding="utf-8") as results:
        for task in tasks:
            task_id = task["task_id"]
            if task_id in already:
                continue
            workspace = workspace_root / task_id
            materialize_workspace(task, dataset_root, workspace, args.harness)
            before_copy = args.output_dir / "initial" / f"{task_id}.config"
            before_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(workspace / "analysis.config", before_copy)
            record: dict[str, Any] = {
                "task_id": task_id,
                "category": task["category"],
                "complexity": task["complexity"],
                "required_evidence": task["required_evidence"],
                "initial_sha256": sha256(before_copy),
                "workspace": (
                    str(workspace.relative_to(args.output_dir.resolve()))
                    if workspace.is_relative_to(args.output_dir.resolve())
                    else str(workspace)
                ),
            }
            if args.dry_run:
                record["status"] = "prepared"
            else:
                prompt = native_prompt_for(dataset_root, args.split, task_id)
                record["prompt"] = prompt
                command = cli_command(
                    args.harness, workspace, prompt, args.model, args.reasoning_effort,
                    effective_qwen_base_url,
                )
                started = time.perf_counter()
                try:
                    process = subprocess.run(
                        command, text=True, capture_output=True, timeout=args.timeout,
                        cwd=workspace,
                        env=safe_environment(
                            args.harness, args.qwen_max_output_tokens,
                        ),
                        check=False,
                    )
                    record["status"] = "completed" if process.returncode == 0 else "cli_error"
                    record["returncode"] = process.returncode
                    stdout, stderr = process.stdout, process.stderr
                except subprocess.TimeoutExpired as error:
                    record["status"] = "timeout"
                    record["returncode"] = None
                    stdout = error.stdout or ""
                    stderr = error.stderr or ""
                    if isinstance(stdout, bytes):
                        stdout = stdout.decode(errors="replace")
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode(errors="replace")
                record["elapsed_seconds"] = time.perf_counter() - started
                event_path = args.output_dir / "events" / f"{task_id}.jsonl"
                event_path.parent.mkdir(parents=True, exist_ok=True)
                event_path.write_text(stdout, encoding="utf-8")
                stderr_path = args.output_dir / "events" / f"{task_id}.stderr.txt"
                stderr_path.write_text(stderr, encoding="utf-8")
                events = parse_events(stdout)
                record["event_count"] = len(events)
                record["native_tool_counts"] = native_tool_counts(args.harness, events)
                record["event_file"] = str(event_path.relative_to(args.output_dir))
                record["stderr_file"] = str(stderr_path.relative_to(args.output_dir))
                after = workspace / "analysis.config"
                final_copy = args.output_dir / "final" / f"{task_id}.config"
                final_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(after, final_copy)
                record["final_config"] = str(final_copy.relative_to(args.output_dir))
                record["final_sha256"] = sha256(after)
                record["score"] = score_task(task, before_copy, after)
                validation_observed = observed_agent_validation(
                    args.harness,
                    events,
                    check_inputs=task["required_evidence"] == "I",
                )
                record["score"]["agent_validation_observed"] = validation_observed
                if not validation_observed:
                    record["score"]["passed"] = False
                    record["score"]["reasons"].append(
                        "native trajectory did not contain a successful required config verifier call"
                    )
                scope_violations = native_scope_violations(args.harness, events, workspace)
                record["score"]["native_scope_ok"] = not scope_violations
                if scope_violations:
                    record["score"]["passed"] = False
                    record["score"]["reasons"].extend(scope_violations)
            results.write(json.dumps(record, sort_keys=True) + "\n")
            results.flush()
            print(json.dumps({"task_id": task_id, "status": record["status"], "score": record.get("score")}))

    rows = read_jsonl(results_path)
    summary = summarize(rows, metadata)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
