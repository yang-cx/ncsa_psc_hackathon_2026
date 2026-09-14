#!/usr/bin/env bash
set -euo pipefail
unset VIRTUAL_ENV

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_SFT_ENV="${VERL_SFT_ENV:-/tmp/verl-sft-venv}"
PYTHON="${VERL_PYTHON:-$VERL_SFT_ENV/bin/python}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-0.8B}"
MODEL_REVISION="${MODEL_REVISION:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
VERL_DATA_DIR="${VERL_DATA_DIR:-$REPO_ROOT/artifacts/native-sft/verl/main-agent-approved}"
TRAIN_FILE="${TRAIN_FILE:-$VERL_DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$VERL_DATA_DIR/validation.parquet}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/artifacts/checkpoints/verl-qwen35-0.8b-native-sft}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
LR="${LR:-1e-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
USE_PEFT="${USE_PEFT:-1}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGETS="${LORA_TARGETS:-all-linear}"
PROJECT_NAME="${PROJECT_NAME:-trexfitter-native-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$SAVE_DIR")}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-false}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-false}"
PAD_MODE="${PAD_MODE:-no_padding}"
RESUME_MODE="${RESUME_MODE:-disable}"
TEST_FREQ="${TEST_FREQ:-after_each_epoch}"
SAVE_FREQ="${SAVE_FREQ:-after_each_epoch}"
METRICS_FILE="${METRICS_FILE:-$SAVE_DIR/metrics.jsonl}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing VERL SFT environment at $PYTHON" >&2
  echo "Inside the training container, run: source training/scripts/setup_verl_sft.sh" >&2
  exit 2
fi

mkdir -p "$SAVE_DIR"
export VERL_FILE_LOGGER_PATH="$METRICS_FILE"
for dataset_file in "$TRAIN_FILE" "$VAL_FILE"; do
  if [[ ! -f "$dataset_file" ]]; then
    echo "Missing VERL SFT Parquet: $dataset_file" >&2
    echo "Run training/prepare_verl_sft.py on the replay-approved JSONL first." >&2
    exit 2
  fi
done

if [[ -n "$MODEL_REVISION" ]]; then
  echo "MODEL_REVISION does not pin VERL model weights." >&2
  echo "Resolve that revision first and pass its exact local snapshot as MODEL_PATH." >&2
  exit 2
fi
model_args=("model.path=$MODEL_PATH")
if [[ "$USE_PEFT" == "1" ]]; then
  model_args+=(
    "model.lora_rank=$LORA_RANK"
    "model.lora_alpha=$LORA_ALPHA"
    "model.target_modules=$LORA_TARGETS"
  )
fi

cd "$REPO_ROOT"
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" -m verl.trainer.sft_trainer \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$VAL_FILE" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  '+data.apply_chat_template_kwargs.enable_thinking=false' \
  "data.custom_cls.path=$REPO_ROOT/training/verl_dataset.py" \
  data.custom_cls.name=TReXNativeToolSFTDataset \
  +data.require_native_tool_contract=true \
  "data.train_batch_size=$TRAIN_BATCH_SIZE" \
  "data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU" \
  "data.max_length=$MAX_LENGTH" \
  "data.max_token_len_per_gpu=$MAX_LENGTH" \
  "data.pad_mode=$PAD_MODE" \
  data.truncation=error \
  "data.use_dynamic_bsz=$USE_DYNAMIC_BSZ" \
  data.num_workers=0 \
  data.ignore_input_ids_mismatch=false \
  engine=fsdp \
  engine.dtype=bfloat16 \
  engine.model_dtype=bf16 \
  engine.use_torch_compile=false \
  "model.use_remove_padding=$USE_REMOVE_PADDING" \
  model.enable_gradient_checkpointing=true \
  "optim.lr=$LR" \
  "trainer.default_local_dir=$SAVE_DIR" \
  "trainer.project_name=$PROJECT_NAME" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  'trainer.logger=[console,file]' \
  "trainer.total_epochs=$TOTAL_EPOCHS" \
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
  "trainer.test_freq=$TEST_FREQ" \
  "trainer.save_freq=$SAVE_FREQ" \
  "trainer.resume_mode=$RESUME_MODE" \
  "${model_args[@]}" "$@"
