#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LEGACY_DATA_ROOT="${LEGACY_DATA_ROOT:-${PROJECT_ROOT}/../data}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Full-parameter RL wrapper around 4b_rl.sh.
# Any env var set by the caller still overrides these defaults.
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Full-parameter RL defaults.
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-128}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-16}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-32}"

# No vector artifact saving for full-parameter by default.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-4b_rl_fullparam}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${PROJECT_ROOT}/outputs/4b_rl_fullparam}"
#export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${PROJECT_ROOT}/outputs/4b_rl_fullparam}"
exec bash "${SCRIPT_DIR}/4b_rl.sh" "$@"
