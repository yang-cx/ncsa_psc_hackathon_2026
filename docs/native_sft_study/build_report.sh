#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="$REPO_ROOT/artifacts/reports/native-sft-study-20260914"
BUILD_DIR="$REPORT_DIR/build"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

mkdir -p "$BUILD_DIR"
cd "$REPO_ROOT"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hyy-mpl}" \
  "$PYTHON" docs/native_sft_study/make_plots.py

for _ in 1 2; do
  xelatex -interaction=nonstopmode -halt-on-error \
    -output-directory="$BUILD_DIR" \
    docs/native_sft_study/report.tex
done

cp "$BUILD_DIR/report.pdf" "$REPORT_DIR/native-sft-study.pdf"
printf 'Wrote %s\n' "$REPORT_DIR/native-sft-study.pdf"
