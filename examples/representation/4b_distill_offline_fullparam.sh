#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LEGACY_DATA_ROOT="${LEGACY_DATA_ROOT:-${PROJECT_ROOT}/../data}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full-parameter OFFLINE teacher-rollout reverse-KL distillation, Qwen3-4B on math
# (teacher = math-teacher). Reuses 4b_distill_offline.sh.

export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"
export ACTOR_LR="${ACTOR_LR:-5e-5}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-4b_distill_offline_fullparam}"
xexec bash "${SCRIPT_DIR}/4b_distill_offline.sh" "$@"
