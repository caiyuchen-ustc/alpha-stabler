#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LEGACY_DATA_ROOT="${LEGACY_DATA_ROOT:-${PROJECT_ROOT}/../data}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Full-parameter OFFLINE teacher-rollout reverse-KL distillation, DeepSeek-R1-Distill-Qwen-1.5B
# on SciKnowEval. Uses the pre-generated teacher rollouts from
# gen_teacher_rollouts_deepseek1p5b.sh (teacher = BroRL-1.5B).
#
# Reuses the shared ifevalg_distill_offline.sh engine; only the model / data / reward env differ.
# ============================================================================

# --- deepseek1.5b / sciknoweval specifics ---
export MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${PROJECT_ROOT}/models/science-teacher}"
export OFFLINE_TEACHER_DATA_PATH="${OFFLINE_TEACHER_DATA_PATH:-${PROJECT_ROOT}/data/science/teacher_responses.parquet}"
export DATA_ROOT="${DATA_ROOT:-${LEGACY_DATA_ROOT}/sciknoweval}"
export VAL_FILE="${VAL_FILE:-${DATA_ROOT}/sciknoweval_validation.parquet}"
export VERIFY_SCRIPT_PATH="${VERIFY_SCRIPT_PATH:-${PROJECT_ROOT}/verl/utils/reward_score/sciknoweval_verify.py}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"     # DeepSeek-R1-Distill is a thinking model
export GEN_TP="${GEN_TP:-1}"

export PROJECT_NAME="${PROJECT_NAME:-DeepSeek1p5B_SciKnowEval_OfflineDistill}"
export EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-deepseek1p5b_distill_offline_fullparam}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${PROJECT_ROOT}/outputs/deepseek1p5b_distill_offline_fullparam}"

# --- full-parameter training ---
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"
export ACTOR_LR="${ACTOR_LR:-1e-5}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepseek1p5b_distill_offline_fullparam}"
exec bash "${SCRIPT_DIR}/deepseek1p5b_distill_offline.sh" "$@"
