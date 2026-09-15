import importlib.util
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "data/datasets/hyy-trexfitter-agent-trajectories"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load("native_agent_study", PROJECT / "inference/run_native_agent_study.py")
EXPORT = load("native_sft_export", DATASET / "tools/export_native_sft.py")
PREFLIGHT = load("qwen_sft_preflight", DATASET / "tools/preflight_qwen_sft.py")
RECONSTRUCTION = load(
    "reconstruction_builder", DATASET / "tools/build_reconstruction_tasks.py"
)
from trex_fitter.expression_equivalence import (  # noqa: E402
    contract_values_equivalent,
    expressions_equivalent,
)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_native_task_renderings_cover_current_tasks_without_domain_tools():
    forbidden = {"list_blocks", "search_settings", "read_config", "verify_config"}
    expected = {"train": 120, "validation": 36, "test": 24}
    seen_by_split = {}
    for split, logical_tasks in expected.items():
        split_rows = rows(DATASET / f"data/native/tasks/{split}.jsonl")
        assert len(split_rows) == logical_tasks
        assert len({row["logical_task_id"] for row in split_rows}) == logical_tasks
        assert {row["modality"] for row in split_rows} == {"coding_agent"}
        assert all("target_model_family" not in row for row in split_rows)
        assert all("target_chat_template" not in row for row in split_rows)
        assert {row["tool_contract"] for row in split_rows} == {"native-agent-protocol/v1"}
        assert all(row["native_tool_policy"]["custom_tools"] is False for row in split_rows)
        assert all(not (forbidden & set(json.dumps(row).split())) for row in split_rows)
        assert not any(name in json.dumps(split_rows) for name in forbidden)
        seen_by_split[split] = {row["logical_task_id"] for row in split_rows}
    assert not (seen_by_split["train"] & seen_by_split["validation"])
    assert not (seen_by_split["train"] & seen_by_split["test"])
    assert not (seen_by_split["validation"] & seen_by_split["test"])


def test_reconstruction_tasks_remove_one_whole_block_and_split_by_identity(tmp_path):
    tasks = RUNNER.scoreable_tasks(DATASET)
    reconstruction = [task for task in tasks if task.get("task_type") == "block_reconstruction"]
    assert len(reconstruction) == 12
    assert {task["split"] for task in reconstruction} == {"train", "validation"}
    assert sum(task["split"] == "train" for task in reconstruction) == 8
    assert sum(task["split"] == "validation" for task in reconstruction) == 4
    identities = [
        (task["reconstruction_target"]["block_kind"], task["reconstruction_target"]["block_name"])
        for task in reconstruction
    ]
    assert len(identities) == len(set(identities))

    seed = RECONSTRUCTION.seed_path(DATASET).read_text()
    seed_path = tmp_path / "seed.config"
    seed_path.write_text(seed)
    seed_map = RUNNER.semantic_map(seed_path)
    for task in reconstruction:
        before = DATASET / task["state_before"]
        before_map = RUNNER.semantic_map(before)
        removed = {key for key in seed_map if key not in before_map}
        expected = {
            (item["block"], item["name"], item["setting"])
            for item in task["change_contract"]
        }
        assert removed == expected
        assert not {key for key in before_map if key not in seed_map}


def test_every_reconstruction_gold_state_passes_contract_and_preservation(tmp_path):
    tasks = [
        task for task in RUNNER.scoreable_tasks(DATASET)
        if task.get("task_type") == "block_reconstruction"
    ]
    restored = tmp_path / "analysis.config"
    restored.write_text(RECONSTRUCTION.seed_path(DATASET).read_text())
    for task in tasks:
        score = RUNNER.score_task(task, DATASET / task["state_before"], restored)
        assert score["passed"], (task["task_id"], score["reasons"])
        assert score["contract_ok"] and score["preservation_ok"] and score["analysis_valid"]


def test_reconstruction_native_prompt_describes_missing_block_without_custom_tools():
    task = next(
        row for row in rows(DATASET / "data/native/tasks/validation.jsonl")
        if row["id"] == "hyy-reconstruct-sample-wh"
    )
    assert task["task_type"] == "block_reconstruction"
    assert task["reconstruction_target"]["block_name"] == "WH"
    assert "missing exactly one complete block" in task["prompt"]
    assert "inspect sibling blocks" in task["prompt"]
    assert "read_config" not in task["prompt"]


def test_expression_equivalence_reverses_comparison_and_reorders_boolean_terms():
    assert expressions_equivalent("A > B", "B < A")
    assert expressions_equivalent("x > 1 && y <= 2", "2 >= y && 1 < x")
    assert expressions_equivalent("0 < x < 2", "x > 0 && 2 > x")
    assert not expressions_equivalent("A > B", "B > A")
    assert contract_values_equivalent(
        "Variable", '"x + y",15,100,160', '"y+x",15,100.0,160.0'
    )


def test_native_scorer_accepts_equivalent_reconstructed_selection(tmp_path):
    task = next(
        task for task in RUNNER.scoreable_tasks(DATASET)
        if task["task_id"] == "hyy-reconstruct-region-cat-transition"
    )
    source = RECONSTRUCTION.seed_path(DATASET).read_text()
    source = source.replace(
        "fabs(photon_eta[0])>1.3 && fabs(photon_eta[0])<1.75",
        "1.3<fabs(photon_eta[0]) && 1.75>fabs(photon_eta[0])",
    )
    restored = tmp_path / "analysis.config"
    restored.write_text(source)
    score = RUNNER.score_task(task, DATASET / task["state_before"], restored)
    assert score["passed"], score["reasons"]
    assert score["equivalence_matches"] == ["Region.cat_transition.Selection"]


def test_cli_command_places_codex_global_approval_before_exec(tmp_path):
    command = RUNNER.cli_command("codex", tmp_path, "prompt", None, "low")
    assert command[:5] == ["codex", "--ask-for-approval", "never", "exec", "--json"]
    assert str(tmp_path.resolve()) in command
    assert 'model_reasoning_effort="low"' in command


def test_cli_command_uses_qwen_code_native_tools_and_headless_protocol(tmp_path):
    command = RUNNER.cli_command(
        "qwen", tmp_path, "same user prompt", "Qwen/Qwen3.5-9B", None,
        "http://localhost:8000/v1",
    )
    assert command[0:2] == ["qwen", "--bare"]
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert command[command.index("--auth-type") + 1] == "openai"
    assert command[command.index("--approval-mode") + 1] == "yolo"
    runtime_system = command[command.index("--system-prompt") + 1]
    assert str(tmp_path.resolve()) in runtime_system
    assert "/workspace" not in runtime_system
    assert command[command.index("--model") + 1] == "Qwen/Qwen3.5-9B"
    assert command[command.index("--openai-base-url") + 1] == "http://localhost:8000/v1"
    assert command[command.index("--exclude-tools") + 1] == "get_goal,update_goal,notebook_edit"
    assert command[-2:] == ["--prompt", "same user prompt"]
    core_tools = command[command.index("--core-tools") + 1]
    assert "read_file" in core_tools
    assert "edit" in core_tools
    assert "run_shell_command(python -m trex_fitter.config_verify)" in core_tools
    assert "apply_patch" not in core_tools
    env = RUNNER.safe_environment("qwen", qwen_max_output_tokens=4096)
    assert env["QWEN_CODE_MAX_OUTPUT_TOKENS"] == "4096"


def test_qwen_sampling_proxy_changes_only_requested_generation_fields():
    captured = {}

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            captured.update(json.loads(raw))
            response = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    try:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    except PermissionError:
        pytest.skip("sandbox forbids loopback sockets")
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = RUNNER.QwenSamplingProxy(
        f"http://127.0.0.1:{upstream.server_port}/v1",
        {"temperature": 0.2, "top_p": 0.95, "seed": 7},
    )
    proxy.start()
    try:
        request = urllib.request.Request(
            proxy.base_url + "/chat/completions",
            data=json.dumps({"model": "qwen", "messages": [{"role": "user", "content": "x"}]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert json.load(urllib.request.urlopen(request)) == {"ok": True}
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
    assert captured == {
        "model": "qwen",
        "messages": [{"role": "user", "content": "x"}],
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 7,
    }


def test_parse_events_accepts_qwen_buffered_json_and_stream_json():
    events = [{"type": "system"}, {"type": "result", "subtype": "success"}]
    assert RUNNER.parse_events(json.dumps(events)) == events
    assert RUNNER.parse_events("\n".join(json.dumps(event) for event in events)) == events


def test_materialized_workspace_is_an_isolated_git_repository(tmp_path):
    task = next(row for row in RUNNER.scoreable_tasks(DATASET) if row["split"] == "train")
    workspace = tmp_path / "task"
    RUNNER.materialize_workspace(task, DATASET, workspace, "codex")
    assert (workspace / ".git").is_dir()
    assert (workspace / "analysis.config").is_file()


def test_runner_uses_exact_checked_in_native_prompt():
    expected = next(
        row for row in rows(DATASET / "data/native/tasks/validation.jsonl")
        if row["id"] == "hyy-traj-fit_controls-fit-minos-muh"
    )
    actual = RUNNER.native_prompt_for(
        DATASET, "validation", "hyy-traj-fit_controls-fit-minos-muh"
    )
    assert actual == expected["prompt"]
    assert "bounded reads" in actual
    assert "--evidence-level S" in actual


def test_native_event_gate_requires_successful_exact_python_verifier():
    codex = [{
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "command": "/usr/bin/bash -lc 'python -m trex_fitter.config_verify analysis.config --actions n'",
            "exit_code": 0,
        },
    }]
    assert RUNNER.observed_agent_validation("codex", codex, check_inputs=False)
    assert not RUNNER.observed_agent_validation("codex", codex, check_inputs=True)
    codex[0]["item"]["exit_code"] = 1
    assert not RUNNER.observed_agent_validation("codex", codex, check_inputs=False)
    codex[0]["item"]["exit_code"] = 0
    codex[0]["item"]["command"] = (
        "/usr/bin/bash -lc 'python -m trex_fitter.config_verify analysis.config --actions n\n"
        "git diff -- analysis.config'"
    )
    assert not RUNNER.observed_agent_validation("codex", codex, check_inputs=False)

    opencode = [{
        "type": "tool_use",
        "part": {
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": {"command": "python3 -m trex_fitter.config_verify ./analysis.config --actions=n --check-inputs"},
                "metadata": {"exit": 0},
            },
        },
    }]
    assert RUNNER.observed_agent_validation("opencode", opencode, check_inputs=True)

    qwen = [
        {
            "type": "assistant",
            "message": {"content": [{
                "type": "tool_use", "id": "verify-1", "name": "run_shell_command",
                "input": {
                    "command": "python -m trex_fitter.config_verify analysis.config --actions n",
                    "is_background": False,
                },
            }]},
        },
        {
            "type": "user",
            "message": {"content": [{
                "type": "tool_result", "tool_use_id": "verify-1", "is_error": False,
                "content": "Command: ...\nExit Code: 0\nVALID: analysis.config",
            }]},
        },
    ]
    assert RUNNER.observed_agent_validation("qwen", qwen, check_inputs=False)
    assert RUNNER.native_tool_counts("qwen", qwen) == {"run_shell_command": 1}
    qwen[1]["message"]["content"][0]["is_error"] = True
    assert not RUNNER.observed_agent_validation("qwen", qwen, check_inputs=False)

    verifier_then_edit = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python -m trex_fitter.config_verify analysis.config --actions n",
                "exit_code": 0,
            },
        },
        {"type": "item.completed", "item": {"type": "file_change", "id": "late-edit"}},
    ]
    assert not RUNNER.observed_agent_validation("codex", verifier_then_edit, check_inputs=False)


def test_native_scope_gate_rejects_parent_reads_and_allows_workspace_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    allowed = [{
        "type": "tool_use",
        "part": {
            "tool": "read", "state": {"status": "completed", "input": {
                "filePath": str(workspace / "analysis.config"),
            }},
        },
    }]
    assert RUNNER.native_scope_violations("opencode", allowed, workspace) == []
    outside = json.loads(json.dumps(allowed))
    outside[0]["part"]["state"]["input"]["filePath"] = str(tmp_path / "secret.txt")
    assert RUNNER.native_scope_violations("opencode", outside, workspace) == [
        "read referenced a path outside the task workspace"
    ]

    codex = [{
        "type": "item.completed",
        "item": {"type": "command_execution", "command": "sed -n 1,5p ../other.config"},
    }]
    assert RUNNER.native_scope_violations("codex", codex, workspace) == [
        "shell command referenced a path outside the task workspace"
    ]
    codex[0]["item"]["command"] = (
        f"python -m trex_fitter.config_verify {workspace / 'analysis.config'} --actions n"
    )
    assert RUNNER.native_scope_violations("codex", codex, workspace) == []

    qwen = [
        {
            "type": "assistant",
            "message": {"content": [{
                "type": "tool_use", "id": "edit-1", "name": "edit",
                "input": {
                    "file_path": str(workspace / "analysis.config"),
                    "old_string": "old", "new_string": "new",
                },
            }]},
        },
        {
            "type": "user",
            "message": {"content": [{
                "type": "tool_result", "tool_use_id": "edit-1", "is_error": False,
                "content": "Successfully modified file",
            }]},
        },
    ]
    assert RUNNER.native_scope_violations("qwen", qwen, workspace) == []
    qwen[0]["message"]["content"][0]["input"]["file_path"] = str(workspace / "other.txt")
    assert RUNNER.native_scope_violations("qwen", qwen, workspace) == [
        "edit targeted a file other than analysis.config"
    ]
    qwen[1]["message"]["content"][0]["is_error"] = True
    assert RUNNER.native_scope_violations("qwen", qwen, workspace) == []


def test_codex_events_normalize_to_qwen_code_native_tools(tmp_path):
    before = tmp_path / "before.config"
    after = tmp_path / "after.config"
    before.write_text('Fit: "fit"\n%  UseMinos: mu_H\n')
    after.write_text('Fit: "fit"\n  UseMinos: mu_H\n')
    events = [
        {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "Inspecting."}},
        {"type": "item.completed", "item": {"id": "c1", "type": "command_execution", "command": "/bin/bash -lc 'sed -n 1,2p analysis.config'", "aggregated_output": "Fit\n", "exit_code": 0}},
        {"type": "item.completed", "item": {"id": "p1", "type": "file_change"}},
        {
            "type": "item.completed",
            "item": {
                "id": "m2",
                "type": "agent_message",
                "text": f"Validated {after}:2.",
            },
        },
    ]
    messages, tools, note = EXPORT.codex_messages(events, before, after)
    assert tools == {"run_shell_command", "edit"}
    assert [call["function"]["name"] for message in messages for call in message.get("tool_calls", [])] == [
        "run_shell_command", "edit"
    ]
    assert messages[-1]["content"] == "Validated /workspace/analysis.config:2."
    edit = messages[2]["tool_calls"][0]["function"]["arguments"]
    assert edit["file_path"] == "/workspace/analysis.config"
    assert edit["old_string"] == "%  UseMinos: mu_H\n"
    assert edit["new_string"] == "  UseMinos: mu_H\n"
    assert str(tmp_path) not in json.dumps(messages)
    assert "reconstructed" in note


def test_opencode_events_map_to_qwen_code_tools_and_remove_absolute_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "analysis.config"
    event = {
        "type": "tool_use",
        "part": {
            "tool": "edit", "callID": "call-1",
            "state": {
                "status": "completed",
                "input": {"filePath": str(config), "oldString": "old", "newString": "new"},
                "output": f"edited {config}",
            },
        },
    }
    messages, tools, _ = EXPORT.opencode_messages([event, {"type": "text", "part": {"text": "Done."}}], workspace)
    assert tools == {"edit"}
    function = messages[0]["tool_calls"][0]["function"]
    assert function["name"] == "edit"
    assert function["arguments"] == {
        "file_path": "/workspace/analysis.config",
        "old_string": "old", "new_string": "new", "replace_all": False,
    }
    assert str(tmp_path) not in json.dumps(messages)
    assert messages[-1]["content"] == "Done."


def test_qwen_events_preserve_native_recovery_and_remove_thinking(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "analysis.config"
    events = [
        {
            "type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "tool_use", "id": "bad", "name": "read_file",
                 "input": {"file_path": "/workspace/analysis.config"}},
            ]},
        },
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "bad", "is_error": True,
             "content": "File not found"},
        ]}},
        {
            "type": "assistant", "message": {"content": [
                {"type": "text", "text": "Retrying. "},
                {"type": "tool_use", "id": "good", "name": "read_file",
                 "input": {"file_path": str(config)}},
            ]},
        },
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "good", "is_error": False,
             "content": f"read {config}"},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Validation passed."},
        ]}},
    ]
    messages, tools, note, kind = EXPORT.qwen_messages(events, workspace)
    assert tools == {"read_file"}
    assert kind == "recovery"
    assert "hidden" not in json.dumps(messages)
    assert str(tmp_path) not in json.dumps(messages)
    assert messages[2]["tool_calls"][0]["function"]["arguments"]["file_path"] == "/workspace/analysis.config"
    assert "native calls" in note


def test_export_contract_has_qwen_code_native_tool_manifest_without_artifact_dependency():
    expected = ["read_file", "edit", "run_shell_command"]
    tools = [EXPORT.TOOL_SCHEMAS[name] for name in EXPORT.CANONICAL_TOOL_NAMES]
    assert [tool["function"]["name"] for tool in tools] == expected
    assert EXPORT._tool_snapshot["qwen_code_package"] == "@qwen-code/qwen-code"
    assert EXPORT._tool_snapshot["qwen_code_version"] == "0.23.4"
    assert "pages" in EXPORT.TOOL_SCHEMAS["read_file"]["function"]["parameters"]["properties"]
    assert EXPORT.TOOL_SCHEMAS["run_shell_command"]["function"]["parameters"]["required"] == ["command"]
    row = {
        "uuid": "synthetic-contract-check",
        "messages": [
            {"role": "system", "content": "Use the supplied tools."},
            {"role": "user", "content": "Inspect the config."},
            {
                "role": "assistant",
                "content": "I will inspect it.",
                "tool_calls": [{
                    "id": "call-1", "type": "function",
                    "function": {"name": "read_file", "arguments": {"file_path": "/workspace/analysis.config"}},
                }],
            },
            {"role": "tool", "content": "Job: hyy", "tool_call_id": "call-1"},
            {"role": "assistant", "content": "The config was inspected."},
        ],
        "tools": tools,
        "tool_contract": "qwen-code-native-tools/v1",
    }
    PREFLIGHT.semantic_check(row)
    assert "target_model_family" not in row
    assert "assistant_loss_mask" not in row


def test_native_export_row_contains_no_model_tokens_or_authored_loss_mask(tmp_path):
    study = tmp_path / "study"
    (study / "events").mkdir(parents=True)
    (study / "initial").mkdir()
    (study / "work/task-1").mkdir(parents=True)
    (study / "initial/task-1.config").write_text('Fit: "fit"\n%  UseMinos: mu_H\n')
    (study / "work/task-1/analysis.config").write_text('Fit: "fit"\n  UseMinos: mu_H\n')
    (study / "events/run.jsonl").write_text("\n".join(json.dumps(event) for event in [
        {
            "type": "item.completed",
            "item": {"id": "p1", "type": "file_change"},
        },
        {
            "type": "item.completed",
            "item": {
                "id": "c1", "type": "command_execution",
                "command": "python -m trex_fitter.config_verify analysis.config --actions n",
                "aggregated_output": "VALID: analysis.config", "exit_code": 0,
            },
        },
        {
            "type": "item.completed",
            "item": {"id": "m1", "type": "agent_message", "text": "Validation passed."},
        },
    ]) + "\n")
    record = {
        "task_id": "task-1",
        "status": "completed",
        "score": {"passed": True},
        "workspace": "work/task-1",
        "event_file": "events/run.jsonl",
        "prompt": "Validate analysis.config.",
        "category": "fit_controls",
        "required_evidence": "S",
    }
    metadata = {
        "harness": "codex",
        "harness_version": "test",
        "model": "source-model",
        "split": "train",
        "source_dataset": EXPORT.ACTIVE_SOURCE_DATASET,
    }
    row = EXPORT.render(study, record, metadata)
    assert row["schema_version"] == "hyy-trexfitter-native-trajectory/v0.7"
    assert row["tool_contract"] == "qwen-code-native-tools/v1"
    assert "target_model_family" not in row
    assert "target_chat_template" not in row
    assert "assistant_loss_mask" not in row
    assert row["verification"]["agent_validation_observed"] is True
    PREFLIGHT.semantic_check(row)


def test_native_export_rejects_verifier_that_precedes_final_edit(tmp_path):
    messages = [
        {
            "role": "assistant", "content": "Validating.",
            "tool_calls": [{
                "id": "verify", "type": "function", "function": {
                    "name": "run_shell_command", "arguments": {
                        "command": "python -m trex_fitter.config_verify analysis.config --actions n",
                        "is_background": False,
                    },
                },
            }],
        },
        {"role": "tool", "tool_call_id": "verify", "content": "VALID: analysis.config"},
        {
            "role": "assistant", "content": "One more edit.",
            "tool_calls": [{
                "id": "patch", "type": "function", "function": {
                    "name": "edit", "arguments": {
                        "file_path": "/workspace/analysis.config",
                        "old_string": "old", "new_string": "new",
                    },
                },
            }],
        },
        {"role": "tool", "tool_call_id": "patch", "content": "Patch applied."},
        {"role": "assistant", "content": "Validation passed."},
    ]
    try:
        EXPORT.validate_normalized_workflow("task-1", messages, check_inputs=False)
    except ValueError as error:
        assert "after final edit" in str(error)
    else:
        raise AssertionError("trajectory with an unvalidated final edit was accepted")


def test_native_export_rejects_trajectory_without_observed_verifier(tmp_path):
    study = tmp_path / "study"
    (study / "events").mkdir(parents=True)
    (study / "events/run.jsonl").write_text(json.dumps({
        "type": "item.completed",
        "item": {"id": "m1", "type": "agent_message", "text": "Looks valid."},
    }) + "\n")
    record = {
        "task_id": "task-1", "status": "completed", "score": {"passed": True},
        "workspace": "work/task-1", "event_file": "events/run.jsonl",
        "prompt": "Validate analysis.config.", "category": "fit_controls",
        "required_evidence": "S",
    }
    metadata = {
        "harness": "codex", "harness_version": "test", "model": "source-model",
        "split": "train", "source_dataset": EXPORT.ACTIVE_SOURCE_DATASET,
    }
    try:
        EXPORT.render(study, record, metadata)
    except ValueError as error:
        assert "no successful required config verifier call" in str(error)
    else:
        raise AssertionError("trajectory without a verifier call was accepted")


def test_native_scorer_accepts_requested_edit_and_preservation(tmp_path):
    task = next(row for row in rows(DATASET / "data/tasks.jsonl") if row["task_id"] == "hyy-traj-fit_controls-fit-minos-muh")
    before = DATASET / task["state_before"]
    after = tmp_path / "analysis.config"
    after.write_text(before.read_text().replace("%  UseMinos: mu_H", "  UseMinos: mu_H"))
    score = RUNNER.score_task(task, before, after)
    assert score["passed"]
    assert score["contract_ok"] and score["preservation_ok"] and score["analysis_valid"]
