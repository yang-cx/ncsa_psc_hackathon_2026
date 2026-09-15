#!/usr/bin/env bash
set -euo pipefail
unset VIRTUAL_ENV

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_DIR="${VERL_DIR:-$REPO_ROOT/verl}"
cd "$VERL_DIR"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/verl-venv}"

source "$REPO_ROOT/training/scripts/qwen35_profile.sh"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-${NPROC_PER_NODE:-$QWEN35_DEFAULT_GPUS}}"
INFER_BACKEND="${INFER_BACKEND:-sglang}"

TRAIN_FILE="${TRAIN_FILE:?Set TRAIN_FILE to a verl-format RL training parquet file}"
VAL_FILE="${VAL_FILE:?Set VAL_FILE to a verl-format RL validation parquet file}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/artifacts/checkpoints/$QWEN35_PROFILE_NAME-rl}"
REWARD_FUNCTION="${REWARD_FUNCTION:?Set REWARD_FUNCTION to the Python reward-function module path}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.65}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-True}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-}"
PROJECT_NAME="${PROJECT_NAME:-trex-config-hackathon}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$QWEN35_PROFILE_NAME-rl}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_TARGETS="${LORA_TARGETS:-[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"]}"
LORA_MERGE="${LORA_MERGE:-True}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-8}"
HYDRA_LOG_ROOT="${HYDRA_LOG_ROOT:-$REPO_ROOT/artifacts/logs/hydra}"
if [[ -z "${HYDRA_RUN_ID:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    HYDRA_RUN_ID="${SLURM_JOB_ID}-${SLURM_STEP_ID:-batch}"
  else
    HYDRA_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  fi
fi
HYDRA_RUN_DIR="$HYDRA_LOG_ROOT/$EXPERIMENT_NAME/$HYDRA_RUN_ID/rank-\${oc.env:RANK,0}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

if [[ "$INFER_BACKEND" == "sglang" && "$LORA_RANK" != "0" ]]; then
  case "$LORA_TARGETS" in
    all|all-linear|\"all\"|\"all-linear\")
      echo "SGLang LoRA does not support LORA_TARGETS=$LORA_TARGETS in this setup." >&2
      echo "Use explicit Qwen targets: [\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"]" >&2
      exit 2
      ;;
  esac
  for arg in "$@"; do
    case "$arg" in
      actor_rollout_ref.model.target_modules=all|actor_rollout_ref.model.target_modules=all-linear)
        echo "SGLang LoRA does not support $arg in this setup." >&2
        echo "Use actor_rollout_ref.model.target_modules='[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"]'" >&2
        exit 2
        ;;
    esac
  done
fi

EXTRA_OVERRIDES=(
  trainer.val_before_train="$VAL_BEFORE_TRAIN"
)

if [[ -n "$TOTAL_TRAINING_STEPS" ]]; then
  EXTRA_OVERRIDES+=(trainer.total_training_steps="$TOTAL_TRAINING_STEPS")
fi
if [[ -n "$TRAIN_MAX_SAMPLES" ]]; then
  EXTRA_OVERRIDES+=(data.train_max_samples="$TRAIN_MAX_SAMPLES")
fi
if [[ -n "$VAL_MAX_SAMPLES" ]]; then
  EXTRA_OVERRIDES+=(data.val_max_samples="$VAL_MAX_SAMPLES")
fi
if [[ -n "$ROLLOUT_DATA_DIR" ]]; then
  EXTRA_OVERRIDES+=(trainer.rollout_data_dir="$ROLLOUT_DATA_DIR")
fi
if [[ -n "$VALIDATION_DATA_DIR" ]]; then
  EXTRA_OVERRIDES+=(trainer.validation_data_dir="$VALIDATION_DATA_DIR")
fi

uv run --frozen --extra fsdp --extra sglang python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules="$LORA_TARGETS" \
  actor_rollout_ref.model.lora.merge="$LORA_MERGE" \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.rollout.name="$INFER_BACKEND" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE_ENGINE" \
  actor_rollout_ref.rollout.layered_summon="$ROLLOUT_LAYERED_SUMMON" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  reward.custom_reward_function.path="$REWARD_FUNCTION" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers="$REWARD_NUM_WORKERS" \
  reward.reward_manager.name=naive \
  trainer.balance_batch=True \
  trainer.logger=console \
  trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.use_v1=False \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  hydra.run.dir="$HYDRA_RUN_DIR" \
  hydra.output_subdir=.hydra \
  hydra.job.chdir=false \
  "${EXTRA_OVERRIDES[@]}" \
  "$@"
