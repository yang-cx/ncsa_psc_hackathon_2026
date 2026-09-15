import json
from pathlib import Path

import pandas as pd

from training.prepare_verl_sft import arrow_safe_row, write_split
from training.sft_schema import read_jsonl, validate_records


PROJECT = Path(__file__).resolve().parents[2]
DATASET = PROJECT / "data/datasets/hyy-trexfitter-agent-trajectories/data/main-agent"


def test_replay_approved_splits_pass_model_neutral_schema():
    for split, expected in (("train", 130), ("validation", 39)):
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
    assert 'trainer.nnodes=$NNODES' in launcher
    assert 'trainer.n_gpus_per_node=$NPROC_PER_NODE' in launcher
    assert '--master_addr="$MASTER_ADDR"' in launcher
    assert "artifacts/logs/hydra" in launcher
    assert 'hydra.run.dir=$HYDRA_RUN_DIR' in launcher
    assert "oc.env:RANK,0" in launcher


def test_verl_launchers_keep_hydra_metadata_under_artifacts():
    rl_launcher = (PROJECT / "training/scripts/run_verl_rl.sh").read_text(encoding="utf-8")
    slurm_worker = (
        PROJECT / "training/scripts/run_verl_sft_slurm_worker.sh"
    ).read_text(encoding="utf-8")

    assert "artifacts/logs/hydra" in rl_launcher
    assert 'hydra.run.dir="$HYDRA_RUN_DIR"' in rl_launcher
    assert "oc.env:RANK,0" in rl_launcher
    assert "HYDRA_RUN_ID" in slurm_worker


def test_verl_preflight_checks_full_render_and_assistant_mask():
    preflight = (PROJECT / "training/preflight_verl_sft.py").read_text(encoding="utf-8")
    assert '"full_render_equals_verl_turn_render": True' in preflight
    assert '"assistant_only_loss": True' in preflight
    assert '"no_thinking": True' in preflight


def test_verl_adapter_requires_qwen_code_native_contract():
    adapter = (PROJECT / "training/verl_dataset.py").read_text(encoding="utf-8")
    assert 'allowed = {"read_file", "edit", "run_shell_command"}' in adapter
    assert '!= "qwen-code-native-tools/v1"' in adapter
    assert '!= "canonical-code-tools/v1"' not in adapter
