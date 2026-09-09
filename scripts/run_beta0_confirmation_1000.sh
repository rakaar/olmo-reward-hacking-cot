#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIRM_RUN_ROOT="${CONFIRM_RUN_ROOT:-/root/cot-decoder-run/confirmation}"
CONFIRM_SELECTION_CACHE="${CONFIRM_SELECTION_CACHE:-/root/cot-decoder-run/data/codecontests-hard-seed42-exclude-pilot50-n200.jsonl}"
CONFIRM_EXCLUSION_MANIFEST="${CONFIRM_EXCLUSION_MANIFEST:?Set CONFIRM_EXCLUSION_MANIFEST to the original 50-problem run_manifest.json}"
CONFIRM_ROOTFS="${CONFIRM_ROOTFS:-/opt/aisi-grader-rootfs-minimal}"
CONFIRM_LAUNCHER="${CONFIRM_LAUNCHER:-/usr/local/bin/aisi-grader-exec}"
PYTHON_BIN="${CONFIRM_PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"

if [[ -e "${CONFIRM_RUN_ROOT}" ]] && [[ -n "$(find "${CONFIRM_RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'Refusing to overwrite nonempty confirmation directory: %s\n' "${CONFIRM_RUN_ROOT}" >&2
    exit 1
fi
mkdir -p "${CONFIRM_RUN_ROOT}"

curl --fail --silent --show-error \
    --header 'Authorization: Bearer inspectai' \
    http://127.0.0.1:8000/v1/models >/dev/null

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/run_checkpoint_pilot.py" \
    --checkpoint-id ai-safety-institute/cc-olmo3-7b-sutl-b0.0-s220 \
    --checkpoint-revision 232b591c69b90b1ec9a866fe270c5ca6763befb0 \
    --upstream-revision 169c3c76a02e51092b4023a8c7baba38f41e2800 \
    --num-problems 200 \
    --samples-per-problem 5 \
    --temperature 1.0 \
    --max-tokens 4096 \
    --dataset-seed 42 \
    --streaming-shuffle-buffer 1000 \
    --selection-cache "${CONFIRM_SELECTION_CACHE}" \
    --exclude-problem-ids-file "${CONFIRM_EXCLUSION_MANIFEST}" \
    --max-connections 4 \
    --output-dir "${CONFIRM_RUN_ROOT}" \
    --sandbox-workdir "${CONFIRM_RUN_ROOT}/work"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/export_inspect_dataset.py" \
    "${CONFIRM_RUN_ROOT}/inspect_logs" \
    --output-dir "${CONFIRM_RUN_ROOT}/export"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/grade_rollouts.py" \
    "${CONFIRM_RUN_ROOT}/export/rollouts.jsonl" \
    --output "${CONFIRM_RUN_ROOT}/export/rollouts_labeled.jsonl" \
    --summary "${CONFIRM_RUN_ROOT}/export/label_summary.json" \
    --rootfs "${CONFIRM_ROOTFS}" \
    --launcher "${CONFIRM_LAUNCHER}" \
    --workers 4

sha256sum \
    "${CONFIRM_RUN_ROOT}/run_manifest.json" \
    "${CONFIRM_RUN_ROOT}/export/rollouts.jsonl" \
    "${CONFIRM_RUN_ROOT}/export/rollouts_labeled.jsonl" \
    "${CONFIRM_RUN_ROOT}/export/label_summary.json" \
    >"${CONFIRM_RUN_ROOT}/SHA256SUMS"
