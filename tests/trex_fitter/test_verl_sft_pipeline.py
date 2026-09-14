import json
from pathlib import Path

import pandas as pd

from training.prepare_verl_sft import arrow_safe_row, write_split
from training.sft_schema import read_jsonl, validate_records


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "data/datasets/hyy-trexfitter-agent-trajectories/data/main-agent"


def test_replay_approved_splits_pass_model_neutral_schema():
    for split, expected in (("train", 129), ("validation", 38)):
        rows = validate_records(read_jsonl(DATASET / f"{split}.jsonl"))
        assert len(rows) == expected


def test_verl_materialization_round_trip_is_arrow_safe(tmp_path):
    source = tmp_path / "sample.jsonl"
    row = read_jsonl(DATASET / "train.jsonl")[0]
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    destination = tmp_path / "sample.parquet"

    report = write_split(source, destination)
    restored = pd.read_parquet(destination).iloc[0]
    prepared = arrow_safe_row(row)

    assert report["rows"] == 1
    assert restored["id"] == row["uuid"]
    assert restored["enable_thinking"] is False or not restored["enable_thinking"]
    assert restored["tools"] == prepared["tools"]
    calls = restored["messages"][2]["tool_calls"]
    assert isinstance(calls[0]["function"]["arguments"], str)


def test_verl_launcher_uses_native_sft_trainer_and_hard_gates():
    launcher = (PROJECT / "training/scripts/run_verl_sft.sh").read_text(encoding="utf-8")
    assert "verl.trainer.sft_trainer" in launcher
    assert "TReXNativeToolSFTDataset" in launcher
    assert "data.truncation=error" in launcher
    assert "enable_thinking=false" in launcher
    assert "trainer.logger=[console,file]" in launcher


def test_verl_preflight_checks_full_render_and_assistant_mask():
    preflight = (PROJECT / "training/preflight_verl_sft.py").read_text(encoding="utf-8")
    assert '"full_render_equals_verl_turn_render": True' in preflight
    assert '"assistant_only_loss": True' in preflight
    assert '"no_thinking": True' in preflight
