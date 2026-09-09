#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PILOT_STORAGE_ROOT="${PILOT_STORAGE_ROOT:-/workspace}"
PILOT_OUTPUT_DIR="${PILOT_OUTPUT_DIR:-${PILOT_STORAGE_ROOT}/runs/beta0-step220-pilot200}"
PILOT_SELECTION_CACHE="${PILOT_SELECTION_CACHE:-${PILOT_STORAGE_ROOT}/datasets/codecontests-hard-seed42-n50.jsonl}"
PILOT_ROOTFS="${PILOT_ROOTFS:-/opt/aisi-grader-rootfs-minimal}"
PILOT_LAUNCHER="${PILOT_LAUNCHER:-/usr/local/bin/aisi-grader-exec}"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/run_checkpoint_pilot.py" \
    --checkpoint-id ai-safety-institute/cc-olmo3-7b-sutl-b0.0-s220 \
    --checkpoint-revision 232b591c69b90b1ec9a866fe270c5ca6763befb0 \
    --upstream-revision 169c3c76a02e51092b4023a8c7baba38f41e2800 \
    --num-problems 50 \
    --samples-per-problem 4 \
    --temperature 1.0 \
    --max-tokens 4096 \
    --dataset-seed 42 \
    --streaming-shuffle-buffer 1000 \
    --selection-cache "${PILOT_SELECTION_CACHE}" \
    --max-connections 2 \
    --output-dir "${PILOT_OUTPUT_DIR}" \
    --sandbox-workdir "${PILOT_OUTPUT_DIR}/work"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/export_inspect_dataset.py" \
    "${PILOT_OUTPUT_DIR}/inspect_logs" \
    --output-dir "${PILOT_OUTPUT_DIR}/export"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/grade_rollouts.py" \
    "${PILOT_OUTPUT_DIR}/export/rollouts.jsonl" \
    --output "${PILOT_OUTPUT_DIR}/export/rollouts_labeled.jsonl" \
    --summary "${PILOT_OUTPUT_DIR}/export/label_summary.json" \
    --rootfs "${PILOT_ROOTFS}" \
    --launcher "${PILOT_LAUNCHER}" \
    --workers 4
