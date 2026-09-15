# Inference

`run_prompts.py` is the general inference entry point. It accepts any causal
language model available through Hugging Face, a local Hugging Face model
directory, or an exported checkpoint from this repository. It takes a list of
prompts and writes the raw completions. It does not apply patches, validate
configs, or run TRExFitter.

`run_tasks.py` remains a small optional wrapper for the later TRExFitter repair
task schema. It turns those structured records into prompts, but uses the same
model runtime.

## Native coding-agent comparison

Use `run_native_agent_study.py` for agent comparisons. It loads the same
checked-in user prompt and the same isolated `analysis.config` fixture for every
harness. Qwen runs use a short, repository-pinned system instruction whose
workspace path is substituted at launch; Qwen Code still owns the tool
descriptions, call parser, execution loop, and tool-result envelope.

For a local Qwen3.5 checkpoint, first expose it through an OpenAI-compatible
server with native tool-call parsing enabled. With vLLM, the important flags are:

```bash
vllm serve /path/to/qwen-checkpoint \
  --served-model-name hyy-qwen \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Then run the task through Qwen Code itself:

```bash
python inference/run_native_agent_study.py \
  --harness qwen \
  --model hyy-qwen \
  --qwen-base-url http://localhost:8000/v1 \
  --split validation \
  --output-dir artifacts/native-agent/qwen35-base
```

Qwen Code supplies and executes its native `read_file`, `edit`, and
`run_shell_command` tools. The runner pins Qwen Code 0.23.4 behavior to a 4,096
token per-turn output ceiling and an isolated-workspace `yolo` approval mode;
the latter cannot modify the source checkout. The evaluator does not parse or apply patches. Run a
trained checkpoint by changing only the model served at the endpoint and the
output directory. Codex and OpenCode use the same entry point with
`--harness codex` and `--harness opencode`; their native tool interfaces are
left intact.

## Checkpoint format

The SFT launcher saves verl's resumable state, then automatically converts its
final FSDP checkpoint to an inference-ready Hugging Face export. This is
important for the default LoRA training setup.

```text
artifacts/checkpoints/my-run/global_step_<N>/
└── huggingface/
    ├── config.json
├── model.safetensors              # present for full-model training
├── lora_adapter/                  # present for LoRA training
│   ├── adapter_config.json
│   └── adapter_model.safetensors
    └── tokenizer files
```

Pass either `global_step_<N>` or its `huggingface/` child to `--model`.
The scripts resolve the latter automatically. For an adapter-only export, the
runtime loads its base model and applies the adapter automatically. A direct
adapter directory also works; use `--base-model Qwen/Qwen3.5-0.8B` if the
adapter metadata does not identify its base model.

For Qwen3.5 hybrid models, the tested vLLM release cannot apply every trained
LoRA target dynamically. First run VERL's `model_merger` to reconstruct the
adapter from FSDP shards, then create a merged serving directory:

```bash
python training/merge_lora_adapter.py \
  --base-model /path/to/pinned/qwen/snapshot \
  --adapter /path/to/global_step_N/huggingface/lora_adapter \
  --output-dir /path/to/global_step_N/huggingface-merged
```

For a checkpoint made before this launcher update, create the export once:

```bash
bash inference/export_verl_checkpoint.sh \
  artifacts/checkpoints/sft-smoke/global_step_300
```

## Run on a GPU node

Use the same Qwen3.5 container session as training. The setup command creates
the node-local uv environment used by every inference command below:

```bash
bash training/scripts/container.sh
source training/scripts/setup.sh
```

Run the examples with `uv run --project /workspace/verl --no-sync python`;
using the container's system `python` bypasses the Qwen3.5-compatible runtime.
The first Qwen3.5-9B run downloads about 19 GB into `/hf_cache`, mapped by
`training/scripts/container.sh` to `$PSCRATCH/qwen35-hf-cache`; a restart resumes it.
Do not use Perlmutter's RAM-backed `/tmp` for this cache. The launcher disables
Xet transfers for large downloads in this Podman-HPC environment. It also maps
the container's `/tmp` to `$PSCRATCH/qwen35-container-tmp`, so interrupted
loads do not consume the Slurm job's RAM allocation.
The launcher also sets `UV_CACHE_DIR=/hf_cache/uv` to avoid unsupported locks
in the home-directory uv cache.

If Podman-HPC exits during the initial 9B download, prefetch the model from the
GPU node's host shell (not from inside the container), then start the
container. This fills the exact cache mounted at `/hf_cache` without making
Podman own the transfer:

```bash
HF_HOME="$PSCRATCH/qwen35-hf-cache" \
HF_HUB_CACHE="$PSCRATCH/qwen35-hf-cache/hub" \
HF_HUB_DISABLE_XET=1 \
hf download Qwen/Qwen3.5-9B
```

## General prompt inference

Create a prompt file. JSON accepts a list of strings or records with `id` and
`prompt`; JSONL accepts one of those values per line:

```json
[
  {"id": "hello", "prompt": "Explain what a likelihood fit does in one sentence."},
  {"id": "config", "prompt": "Write a one-line TRExFitter config comment."}
]
```

Run a base model directly from the Hub:

```bash
uv run --project /workspace/verl --no-sync python inference/run_prompts.py \
  --model Qwen/Qwen3.5-0.8B \
  --prompts prompts.json \
  --output /workspace/artifacts/inference/base-qwen.jsonl \
  --format chat \
  --device cuda
```

For ordinary base language models, omit `--format chat`; this feeds the prompt
as literal text. For instruction-tuned models, `--format chat` uses the
model's Hugging Face chat template. Output JSONL has one metadata record then
one completion record per input prompt, preserving its `id`, prompt, raw output,
and generation time.

Replace `Qwen/Qwen3.5-0.8B` with `Qwen/Qwen3.5-9B` to run the larger profile on
a suitable GPU allocation. Both current Qwen3.5 models are multimodal, but
these text-only commands need no image inputs.

## ROOT SFT held-out test set

The canonical dataset stores its split in the `split` column of `root.jsonl`.
The following commands pass only each held-out record's `question` to the base
model; its reference `answer` is never included in the prompt or output. Run
the 0.8B baseline first:

```bash
uv run --project /workspace/verl --no-sync python inference/run_prompts.py \
  --model Qwen/Qwen3.5-0.8B \
  --prompts /workspace/data/datasets/root-sft-dataset/root.jsonl \
  --prompt-field question \
  --id-field id \
  --filter-field split \
  --filter-value test \
  --format chat \
  --system-prompt "You are a careful high-energy-physics assistant. Answer ROOT questions accurately and concisely. Do not invent unsupported details." \
  --device cuda \
  --temperature 0 \
  --output /workspace/artifacts/inference/root-sft-test-qwen35-0.8b.jsonl
```

On a GPU with sufficient memory for the 9B model, run the matching baseline by
changing the model and output name:

```bash
uv run --project /workspace/verl --no-sync python inference/run_prompts.py \
  --model Qwen/Qwen3.5-9B \
  --prompts /workspace/data/datasets/root-sft-dataset/root.jsonl \
  --prompt-field question \
  --id-field id \
  --filter-field split \
  --filter-value test \
  --format chat \
  --system-prompt "You are a careful high-energy-physics assistant. Answer ROOT questions accurately and concisely. Do not invent unsupported details." \
  --device cuda \
  --temperature 0 \
  --output /workspace/artifacts/inference/root-sft-test-qwen35-9b.jsonl
```

Each JSONL output begins with run metadata and contains exactly 61 completion
records. Keep these base-model outputs separate from SFT checkpoints. On the
40 GB Perlmutter A100, Qwen3.5 uses a 30 GiB GPU placement cap by default,
leaving generation headroom while normally keeping the 9B model on GPU;
override it only with
`QWEN35_GPU_MEMORY_GIB=VALUE` when you have confirmed sufficient headroom.
Qwen3.5 thinking mode is disabled by default so the completion contains the
answer rather than a reasoning trace; use `--enable-thinking` only when that
trace is explicitly wanted.

## ATLAS ROOT zero-shot query benchmark

The public benchmark lives in the `atlas-open-data-sft-dataset` submodule.
For a direct base-model baseline, first generate completions for its eight
query tasks. The prompts request a short final answer; this measures ROOT
knowledge without granting the model tool execution:

```bash
uv run --project /workspace/verl --no-sync python inference/run_prompts.py \
  --model Qwen/Qwen3.5-0.8B \
  --prompts /workspace/data/datasets/atlas-open-data-sft-dataset/data/tasks/query_tasks.jsonl \
  --prompt-field question \
  --id-field id \
  --format chat \
  --no-enable-thinking \
  --system-prompt "Reply with only the requested final answer. Do not explain your reasoning." \
  --device cuda \
  --temperature 0 \
  --max-new-tokens 32 \
  --output /workspace/artifacts/inference/atlas-root-qwen35-0.8b-query.jsonl
```

Then score the output using the dataset's own verifier:

```bash
python inference/evaluate_atlas_benchmark.py \
  --completions /workspace/artifacts/inference/atlas-root-qwen35-0.8b-query.jsonl \
  --dataset-root /workspace/data/datasets/atlas-open-data-sft-dataset \
  --output /workspace/artifacts/inference/atlas-root-qwen35-0.8b-query-score.json
```

This scores all eight query tasks. The three artifact tasks require a future
agent runner that lets the model call `bash`, write a macro, and invoke
`verify_task.py`; they cannot be fairly scored from one-shot text generation.

To evaluate command synthesis instead, first build the dedicated prompts on
the host or in the container:

```bash
python3 /workspace/data/datasets/atlas-open-data-sft-dataset/tools/tasks/build_command_prompts.py \
  --tasks /workspace/data/datasets/atlas-open-data-sft-dataset/data/tasks/query_tasks.jsonl \
  --manifest /workspace/data/datasets/atlas-open-data-sft-dataset/data/fixtures/manifest.json \
  --dataset-root /workspace/data/datasets/atlas-open-data-sft-dataset \
  --output /workspace/artifacts/inference/atlas-root-command-prompts.jsonl
```

Run that file with `--prompt-field prompt` and a system instruction to return
only the command. These completions are commands for a user or agent to
execute; do not score them with `evaluate_atlas_benchmark.py`, which expects
final numerical/text answers.

Attach the expected command and result to a copy of a command-generation run:

```bash
uv run --project /workspace/verl --no-sync python /workspace/inference/attach_atlas_references.py \
  --completions /workspace/artifacts/inference/atlas-root-qwen35-9b-commands-raw.jsonl \
  --dataset-root /workspace/data/datasets/atlas-open-data-sft-dataset \
  --output /workspace/artifacts/inference/atlas-root-qwen35-9b-commands.jsonl
```

## Hugging Face Dataset input

The same runner can read prompts from a Hub dataset. The dataset only needs a
string prompt column; it does not need to use a TRExFitter-specific schema.
For example, a dataset whose `test` split has `instruction` and `task_id`
columns runs as:

```bash
uv run --project /workspace/verl --no-sync python inference/run_prompts.py \
  --model Qwen/Qwen3.5-0.8B \
  --dataset ho22joshua/my-prompt-dataset \
  --split test \
  --prompt-field instruction \
  --id-field task_id \
  --output /workspace/artifacts/inference/my-prompt-dataset.jsonl \
  --format chat \
  --device cuda
```

Use `--dataset-config NAME` for a configured dataset, `--revision COMMIT_OR_TAG`
for a reproducible dataset version, and `--streaming` for a large split. The
output metadata records all of those choices.

## This repository's trained checkpoints

First verify that a selected checkpoint loads and generates text:

```bash
uv run --project /workspace/verl --no-sync python inference/smoke_test.py \
  --checkpoint /workspace/artifacts/checkpoints/sft-smoke/global_step_<N> \
  --device cuda
```

Use the same generic command with the `global_step_N` checkpoint path:

```bash
uv run --project /workspace/verl --no-sync python inference/run_prompts.py \
  --model /workspace/artifacts/checkpoints/sft-smoke/global_step_<N> \
  --prompts prompts.json \
  --output /workspace/artifacts/inference/sft-prompts.jsonl \
  --format chat \
  --device cuda
```

For structured config tasks, run the optional wrapper. A `.json` file can hold
one task or a JSON list; `.jsonl` holds one task per line. Supply task records
from the published task dataset.

```bash
uv run --project /workspace/verl --no-sync python inference/run_tasks.py \
  --checkpoint /workspace/artifacts/checkpoints/sft-smoke/global_step_<N> \
  --tasks <tasks.json-or-jsonl> \
  --output /workspace/artifacts/inference/example.jsonl \
  --device cuda \
  --temperature 0
```

Each output JSONL begins with one metadata record, followed by prediction
records containing the raw answer, an extracted `<patch>...</patch>` diff when
present, and generation time. Feed those records to the future evaluator; do
not give the model access to that evaluator or to TRExFitter.

Set `EXPORT_FOR_INFERENCE=false` when launching training only if you explicitly
want to skip the final conversion. In that case run
`export_verl_checkpoint.sh` before inference.
