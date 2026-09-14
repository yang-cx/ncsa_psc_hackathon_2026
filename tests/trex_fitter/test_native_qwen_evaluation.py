from pathlib import Path

import pytest

from inference.evaluate_native_qwen import (
    DEFAULT_DATASET,
    assistant_content,
    execute_tool,
    prompt_rows,
)
from trex_fitter.patch_apply import apply_patch


def test_bounded_patch_updates_only_analysis_config():
    source = 'Fit: "fit"\n  FitType: SPLUSB\n'
    patch = """*** Begin Patch
*** Update File: analysis.config
@@
 Fit: "fit"
+  FitBlind: TRUE
   FitType: SPLUSB
*** End Patch"""
    assert apply_patch(source, patch) == 'Fit: "fit"\n  FitBlind: TRUE\n  FitType: SPLUSB\n'
    with pytest.raises(ValueError, match="only analysis.config"):
        apply_patch(source, patch.replace("analysis.config", "other.config"))


def test_response_content_drops_qwen_tool_markup():
    response = "Inspecting.\n<tool_call><function=read_file><parameter=path>analysis.config</parameter></function></tool_call><|im_end|>"
    assert assistant_content(response) == "Inspecting."


def test_validation_split_has_one_deterministic_prompt_per_task():
    prompts = prompt_rows(DEFAULT_DATASET, "validation")
    assert len(prompts) == 36
    assert all(len(row["messages"]) >= 2 for row in prompts.values())


def test_tool_executor_rejects_workspace_escape(tmp_path: Path):
    (tmp_path / "analysis.config").write_text('Fit: "fit"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="only analysis.config"):
        execute_tool("read_file", {"path": "../secret"}, tmp_path)
    with pytest.raises(ValueError, match="workspace-relative"):
        execute_tool("search_files", {"mode": "path_glob", "query": "../*"}, tmp_path)
