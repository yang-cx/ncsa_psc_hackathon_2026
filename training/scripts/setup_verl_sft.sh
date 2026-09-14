#!/usr/bin/env bash
# Source this inside training/scripts/container.sh before running VERL SFT.

set -euo pipefail
unset VIRTUAL_ENV

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_SFT_ENV="${VERL_SFT_ENV:-/tmp/verl-sft-venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/hf_cache/uv}"
export UV_PROJECT_ENVIRONMENT="$VERL_SFT_ENV"

mkdir -p "$UV_CACHE_DIR"
uv sync --project "$REPO_ROOT/verl" --frozen --extra fsdp

# The same environment evaluates agent edits with the repository verifier.
# Install its lightweight static/input-inspection dependencies explicitly;
# the host project itself targets Python 3.11 while the pinned VERL image uses
# Python 3.12, so installing the whole host environment here would be invalid.
uv pip install --python "$VERL_SFT_ENV/bin/python" \
  'awkward>=2.8,<3' 'pydantic==2.11.7' 'uproot>=5.6,<6'

export VERL_SFT_ENV
"$VERL_SFT_ENV/bin/python" - <<'PY'
import importlib.metadata
import torch
import transformers
import verl

print("verl:", getattr(verl, "__version__", importlib.metadata.version("verl")))
print("transformers:", transformers.__version__)
print("datasets:", importlib.metadata.version("datasets"))
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
PY

cd "$REPO_ROOT"
