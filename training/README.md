# Qwen native-agent SFT with VERL

The active supervised-training path uses VERL's
`verl.trainer.sft_trainer`. This deliberately shares the model, FSDP,
checkpoint, and data ecosystem intended for the later reinforcement-learning
stage.

The canonical dataset remains model-neutral JSONL containing structured
`messages` plus optional OpenAI-style `tools`. No Qwen markup, token IDs,
labels, or loss masks are stored in the dataset.

## Data boundary

The active public dataset is
[`cxyang-ucb/hyy-sft`](https://huggingface.co/datasets/cxyang-ucb/hyy-sft).
It contains 129 replay-approved training trajectories and 38 validation
trajectories. The public default is a verified Parquet mirror; JSONL remains
the canonical audited source.

Prepare the local VERL transport from the canonical files:

```bash
python training/prepare_verl_sft.py \
  --train data/datasets/hyy-trexfitter-agent-trajectories/data/main-agent/train.jsonl \
  --validation data/datasets/hyy-trexfitter-agent-trajectories/data/main-agent/validation.jsonl \
  --output-dir artifacts/native-sft/verl/main-agent-approved
```

The local Parquet materialization JSON-encodes only heterogeneous Arrow leaves
(`tools` and tool-call arguments). `training/verl_dataset.py` restores those
objects before tokenization. It also requires exactly the generic
`shell`, `read_file`, `search_files`, and `apply_patch` contract.

## Template and loss-mask gate

VERL invokes the selected checkpoint tokenizer's `apply_chat_template()` with
structured messages, tools, and `enable_thinking=False`. The repository does
not invent Qwen control tokens.

Before training, run both splits through the exact tokenizer and VERL dataset:

```bash
python training/preflight_verl_sft.py \
  --input artifacts/native-sft/verl/main-agent-approved/train.parquet \
  --model /path/to/pinned/qwen/snapshot \
  --max-length 12288 \
  --report artifacts/native-sft/verl/main-agent-approved/train-preflight.json

python training/preflight_verl_sft.py \
  --input artifacts/native-sft/verl/main-agent-approved/validation.parquet \
  --model /path/to/pinned/qwen/snapshot \
  --max-length 12288 \
  --report artifacts/native-sft/verl/main-agent-approved/validation-preflight.json
```

The gate iterates every row and fails unless:

- the complete trajectory fits without truncation;
- VERL's turn-wise rendering exactly equals the tokenizer's full-conversation
  rendering;
- at least one assistant token is supervised;
- non-assistant turns receive zero loss;
- the tool schema and tool-call/result graph are valid;
- thinking is disabled consistently.

For Qwen3.5-0.8B at revision
`2fc06364715b967f1860aea9cf38778875588b17`, the current maximum is 10,335
tokens in train and 8,158 in validation, so the study uses a 12,288-token hard
limit.

## Container and environment

Request a GPU node, then start the repository container:

```bash
bash training/scripts/container.sh
```

Inside the container, create the repository-pinned VERL/FSDP environment:

```bash
source training/scripts/setup_verl_sft.sh
```

The disposable environment is stored at `/tmp/verl-sft-venv`; the container
script binds `/tmp` and Hugging Face caches to persistent scratch.

## Training

Run the 0.8B LoRA reference configuration:

```bash
MODEL_PATH=/hf_cache/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<revision> \
NPROC_PER_NODE=4 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_LENGTH=12288 \
LR=1e-4 \
TOTAL_EPOCHS=1 \
USE_PEFT=1 \
LORA_RANK=16 \
LORA_ALPHA=16 \
SAVE_DIR=/workspace/artifacts/checkpoints/verl-qwen35-0.8b-lora-r16-lr1e4-e1 \
bash training/scripts/run_verl_sft.sh
```

Set `USE_PEFT=0` for full-parameter SFT. Important study controls include
`LR`, `TRAIN_BATCH_SIZE`, `TOTAL_EPOCHS`, `LORA_RANK`, and `LORA_ALPHA`.
Use identical data, seed, context length, and evaluation tasks when changing
one control.

The launcher:

- uses VERL's FSDP engine in BF16;
- enables gradient checkpointing;
- rejects truncation;
- evaluates and checkpoints after each epoch by default;
- writes step metrics to `$SAVE_DIR/metrics.jsonl` through VERL's file logger;
- supports one-step smoke tests with `TOTAL_TRAINING_STEPS=1`.

For Qwen3.5's Gated Delta Net implementation, the current study disables
remove-padding and dynamic-batch paths. Larger models should first pass the
same tokenizer preflight and a one-step memory/checkpoint smoke test.

## Checkpoints and inference

Raw VERL checkpoints are written below
`artifacts/checkpoints/<experiment>/global_step_<n>/`. LoRA metadata and the
checkpoint tokenizer/config are saved with every checkpoint. Use
`inference/export_verl_checkpoint.sh` when a merged Hugging Face export is
required by an inference runtime.

Evaluate base and trained models on the same held-out task IDs. Each task must
run in an isolated workspace with the four generic tools, and final accuracy
comes from the independent config/task verifier—not from matching generated
text. Record syntax validity, edit success, verifier success, semantic task
success, unauthorized changes, latency, and token counts.

## Relation to RL

SFT and RL have different objectives, but they now share VERL infrastructure.
The later RL dataset should contain prompts, isolated starting states, and a
deterministic reward/verifier rather than golden assistant trajectories.
Static task-contract success is the first reward level; executed TRExFitter
artifact or histogram equivalence belongs to a stronger RL reward level.
