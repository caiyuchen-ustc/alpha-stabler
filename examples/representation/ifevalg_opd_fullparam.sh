#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LEGACY_DATA_ROOT="${LEGACY_DATA_ROOT:-${PROJECT_ROOT}/../data}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full-parameter IFEvalG instruction-following OPD wrapper around ifevalg_opd.sh.
# Any env var set by the caller still overrides these defaults.
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"

# Full-parameter distillation is typically less stable with the vector-training LR.
export ACTOR_LR="${ACTOR_LR:-1e-6}"

# Vector artifacts are not needed for full-parameter training.
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Optional convenience defaults for naming/outputs.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ifevalg_opd_fullparam}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${PROJECT_ROOT}/outputs/ifevalg_opd_fullparam}"

exec bash "${SCRIPT_DIR}/ifevalg_opd.sh" "$@"
