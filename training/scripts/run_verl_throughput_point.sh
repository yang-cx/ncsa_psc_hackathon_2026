#!/usr/bin/env bash
# Run a fixed-step VERL SFT point and timestamp steady-state metric writes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_TRAIN_STEPS="${EXPECTED_TRAIN_STEPS:-6}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
SAVE_DIR="${SAVE_DIR:?Set SAVE_DIR for this scaling point}"
METRICS_FILE="${METRICS_FILE:-$SAVE_DIR/metrics.jsonl}"
THROUGHPUT_FILE="${THROUGHPUT_FILE:-$SAVE_DIR/throughput.json}"
PYTHON="${VERL_PYTHON:-/tmp/verl-sft-venv/bin/python}"

if [[ "$NODE_RANK" == "0" && ( -e "$METRICS_FILE" || -e "$THROUGHPUT_FILE" ) ]]; then
  echo "Scaling output already exists; choose a fresh SAVE_DIR: $SAVE_DIR" >&2
  exit 2
fi

watcher_pid=""
cleanup() {
  if [[ -n "$watcher_pid" ]]; then
    kill "$watcher_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$NODE_RANK" == "0" ]]; then
  mkdir -p "$SAVE_DIR"
  "$PYTHON" "$REPO_ROOT/training/measure_jsonl_throughput.py" \
    --metrics "$METRICS_FILE" \
    --output "$THROUGHPUT_FILE" \
    --expected-train-steps "$EXPECTED_TRAIN_STEPS" \
    --warmup-steps "$WARMUP_STEPS" &
  watcher_pid=$!
fi

TOTAL_TRAINING_STEPS="$EXPECTED_TRAIN_STEPS" \
METRICS_FILE="$METRICS_FILE" \
TEST_FREQ=999999 \
SAVE_FREQ=999999 \
bash "$REPO_ROOT/training/scripts/run_verl_sft.sh"

if [[ -n "$watcher_pid" ]]; then
  wait "$watcher_pid"
  watcher_pid=""
fi
