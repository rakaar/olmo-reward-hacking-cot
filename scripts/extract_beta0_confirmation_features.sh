#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIRM_RUN_ROOT="${CONFIRM_RUN_ROOT:-/root/cot-decoder-run/confirmation}"
CONFIRM_MODEL_ROOT="${CONFIRM_MODEL_ROOT:-/root/cot-decoder-run/models}"
CONFIRM_FEATURE_DIR="${CONFIRM_FEATURE_DIR:-${CONFIRM_RUN_ROOT}/features}"
PYTHON_BIN="${CONFIRM_PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"

if pgrep -f 'vllm.*serve' >/dev/null; then
    printf 'Refusing to load a second model while the vLLM server is running.\n' >&2
    exit 1
fi

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/extract_cot_decoder_features.py" \
    --input-rollouts "${CONFIRM_RUN_ROOT}/export/rollouts_labeled.jsonl" \
    --output-dir "${CONFIRM_FEATURE_DIR}" \
    --base-model allenai/Olmo-3-7B-Instruct-SFT \
    --base-revision e1452fc572d51966ff4aaeb25118b891eb93e549 \
    --adapter ai-safety-institute/cc-olmo3-7b-sutl-b0.0-s220 \
    --adapter-revision 232b591c69b90b1ec9a866fe270c5ca6763befb0 \
    --base-load-path "${CONFIRM_MODEL_ROOT}/base" \
    --adapter-load-path "${CONFIRM_MODEL_ROOT}/adapter" \
    --tokenizer-load-path "${CONFIRM_MODEL_ROOT}/adapter" \
    --layer 10 \
    --expected-layers 32 \
    --expected-hidden-size 4096 \
    --max-seq-len 8192 \
    --device cuda \
    --dtype bfloat16 \
    --seed 42
