# Qwen native-agent SFT with VERL

The active supervised-training path uses VERL's
`verl.trainer.sft_trainer`. This deliberately shares the model, FSDP,
checkpoint, and data ecosystem intended for the later reinforcement-learning
stage.

The audited source remains structured JSONL containing `messages` plus
OpenAI-style `tools`; Parquet is the training transport. The active
`main-agent-replay-approved` configuration is specifically the Qwen Code SFT
view, not a universal native-agent tool contract. No Qwen markup, token IDs,
labels, or loss masks are stored in either representation.

## Data boundary

The active public dataset is
[`cxyang-ucb/hyy-sft`](https://huggingface.co/datasets/cxyang-ucb/hyy-sft).
It contains 130 replay-approved training trajectories and 39 validation
trajectories. The public default is a verified Parquet mirror; JSONL remains
the canonical audited source.

Prepare the local VERL transport from the canonical files:

```bash
python training/prepare_verl_sft.py \
  --train data/datasets/hyy-trexfitter-agent-trajectories/data/main-agent/train.jsonl \
  --validation data/datasets/hyy-trexfitter-agent-trajectories/data/main-agent/validation.jsonl \
  --output-dir artifacts/native-sft/verl/main-agent-approved
```

The current public Parquet files preserve nested `messages` and `tools` and
round-trip those fields against the audited JSONL. A VERL transport may
JSON-encode heterogeneous Arrow leaves; the dataset adapter must restore them
before tokenization. The active Qwen Code view exposes exactly `read_file`,
`edit`, and `run_shell_command` under `qwen-code-native-tools/v1`.
Their complete definitions are the checked-in capture of the official
`@qwen-code/qwen-code` 0.23.4 OpenAI request, not repository-authored
approximations.

The checked-in `training/verl_dataset.py` adapter rejects the superseded
four-tool snapshot and requires the three-tool Qwen Code contract before any
row reaches tokenization.

## Template and loss-mask gate

VERL invokes the selected checkpoint tokenizer's `apply_chat_template()` with
structured messages, tools, and `enable_thinking=False`. The repository does
not invent Qwen control tokens.

Before training, run both splits through the exact tokenizer and VERL dataset:

```bash
python training/preflight_verl_sft.py \
  --input artifacts/native-sft/verl/main-agent-approved/train.parquet \
  --model /path/to/pinned/qwen/snapshot \
  --max-length 16384 \
  --report artifacts/native-sft/verl/main-agent-approved/train-preflight.json

python training/preflight_verl_sft.py \
  --input artifacts/native-sft/verl/main-agent-approved/validation.parquet \
  --model /path/to/pinned/qwen/snapshot \
  --max-length 16384 \
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

The previously recorded 10,335-token train maximum and 8,158-token validation
maximum were measured on the superseded four-tool snapshot. Re-run both
preflights with the current Parquet and exact target checkpoint before choosing
the context limit for a new experiment.

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

The native Qwen Code Qwen3.5-27B study uses the same Parquet data, LoRA
rank/alpha, learning rate, and one-epoch schedule.  Its pinned tokenizer
renders the longest training row as 13,629 tokens, so the run uses a 14,336
limit and rejects truncation.  Eight A100s were insufficient at this length.
The completed run uses four nodes (16 A100s), four-way Ulysses sequence
parallelism to distribute each long sequence within a GPU group, and
remove-padding to avoid computation on padding tokens:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)" \
MODEL_PATH=/hf_cache/hub/models--Qwen--Qwen3.5-27B/snapshots/<revision> \
NNODES=4 \
NPROC_PER_NODE=4 \
TRAIN_BATCH_SIZE=16 \
MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_LENGTH=14336 \
ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
USE_REMOVE_PADDING=true \
LR=1e-4 \
TOTAL_EPOCHS=1 \
USE_PEFT=1 \
LORA_RANK=16 \
LORA_ALPHA=16 \
SAVE_DIR=/workspace/artifacts/checkpoints/verl-qwen35-27b-qwen-code-0.23.4-lora-r16-lr1e-4-e1-16gpu \
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  bash training/scripts/run_verl_sft_slurm_worker.sh
```

The completed study run wrote eight optimizer steps and an end-of-epoch
validation loss of `0.0982496440` to
`artifacts/checkpoints/verl-qwen35-27b-qwen-code-0.23.4-lora-r16-lr1e-4-e1-16gpu/metrics.jsonl`.
Its raw VERL checkpoint is `global_step_8`; the merged Hugging Face export and
LoRA adapter are under that checkpoint's `huggingface-merged/` and
`huggingface-exported/lora_adapter/` directories, respectively.
Through the real Qwen Code 0.23.4 harness with backend-default sampling, the
untouched 27B checkpoint passes 28/36 held-out tasks and this native-tool LoRA
checkpoint passes 29/36. The one-task difference is descriptive, not evidence
of a stable improvement without additional controlled seeds.

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

Hydra's resolved configuration and per-rank logs are retained under
`artifacts/logs/hydra/<experiment>/<run-id>/rank-<rank>/` instead of the
repository-level `outputs/` default. Slurm launches use the job and step IDs to
group ranks from the same run. Set `HYDRA_RUN_ID` to give a run a stable custom
identifier, or `HYDRA_LOG_ROOT` to relocate this metadata explicitly.

For a strong-scaling point, use `run_verl_throughput_point.sh` with a fresh
`SAVE_DIR`. It timestamps VERL's per-step token counts, excludes the first
compile/warm-up step, and reports total non-padding training-sequence tokens
per steady-state wall-clock second. Hold the model, data order, global batch,
maximum sequence length, optimizer, and fixed step count constant while
changing only the A100 count. Multi-node runs start the watcher only on rank
zero, so every point has one unambiguous timing record.

The smaller-model study leaves remove-padding and dynamic batching disabled.
The 27B long-context configuration requires remove-padding together with
four-way sequence parallelism. Larger models should first pass the same exact
tokenizer preflight and a one-step memory/checkpoint smoke test.

## Checkpoints and inference

Raw VERL checkpoints are written below
`artifacts/checkpoints/<experiment>/global_step_<n>/`. LoRA metadata and the
checkpoint tokenizer/config are saved with every checkpoint. Use
`inference/export_verl_checkpoint.sh` when a merged Hugging Face export is
required by an inference runtime.

Evaluate base and trained models on the same held-out task IDs. Each task must
run in an isolated workspace through the target agent's native harness. Qwen
runs use Qwen Code's `read_file`, `edit`, and `run_shell_command`; Codex and
OpenCode runs retain their own native interfaces. Final accuracy comes from
the same independent config/task verifier—not from matching generated text.
Record syntax validity, edit success, verifier success, semantic task success,
unauthorized changes, latency, and token counts.
On NERSC, pass `--workspace-root "$PSCRATCH/<study-workspaces>"` to the native
study runner so repeated `git init` operations use scratch rather than CFS.
The runner still copies each initial and final config, event stream, stderr,
manifest, and score into the selected artifact output directory.

### Superseded four-tool study

The completed 27B LoRA result—26 strict passes versus 0 for the untouched
base—used the earlier `canonical-code-tools/v1` dataset, not the current Qwen
Code Parquet. It must not be reported as evidence for the new protocol. New
results require retraining from the current dataset and must record its exact
Parquet SHA-256, tokenizer/template revision, generation limits, decoding
settings, and seed.

## Relation to RL

SFT and RL have different objectives, but they now share VERL infrastructure.
The later RL dataset should contain prompts, isolated starting states, and a
deterministic reward/verifier rather than golden assistant trajectories.
Static task-contract success is the first reward level; executed TRExFitter
artifact or histogram equivalence belongs to a stronger RL reward level.
