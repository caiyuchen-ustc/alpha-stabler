#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LEGACY_DATA_ROOT="${LEGACY_DATA_ROOT:-${PROJECT_ROOT}/../data}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export Teacher_MODEL_PATH="${Teacher_MODEL_PATH:-${MODEL_ROOT}/BroRL-1.5B}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/deepseek1p5b_opd_fullparam}"

export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepseek1p5b_opd_fullparam}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${RUN_ROOT}/checkpoints}"
export SAVE_VECTOR_BASE_DIR="${SAVE_VECTOR_BASE_DIR:-${RUN_ROOT}/trainable_vectors}"

exec bash "${SCRIPT_DIR}/1p5bopd.sh" "$@"
