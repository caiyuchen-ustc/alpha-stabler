#!/usr/bin/env bash

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LEGACY_DATA_ROOT="${LEGACY_DATA_ROOT:-${PROJECT_ROOT}/../data}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Full-parameter OFF-POLICY distillation, IFEvalG-style.
#
# Unlike ifevalg_opd_fullparam.sh (on-policy: the *student* samples), here the
# frozen *teacher* generates the rollouts (vLLM serves the teacher checkpoint and
# the per-step student->vLLM weight sync is skipped). The student then computes its
# own log-prob and the teacher its ref log-prob, the advantage stays the
# base-corrected reverse-KL, and the rollout IS correction stays on. This is
# effectively SFT-style distillation on teacher-generated data.
#
# Any env var set by the caller still overrides these defaults.
# ============================================================================

# Full-parameter student training.
export LORA_RANK="${LORA_RANK:-0}"
export ENABLE_TRAINABLE_TOKEN_VECTOR="${ENABLE_TRAINABLE_TOKEN_VECTOR:-false}"
export SAVE_VECTOR="${SAVE_VECTOR:-false}"

# Off-policy: vLLM serves the frozen teacher. OFF_POLICY_TEACHER_PATH defaults to the
# ref/teacher model path (TEACHER_MODEL_PATH) inside ifevalg_opd.sh.
export OFF_POLICY_ROLLOUT="${OFF_POLICY_ROLLOUT:-true}"

# Keep the OPD objective: base-corrected reverse-KL advantage + rollout IS correction.
export ACTOR_ONLY_REVERSE_KL_ADVANTAGES="${ACTOR_ONLY_REVERSE_KL_ADVANTAGES:-true}"
export ROLLOUT_IS="${ROLLOUT_IS:-token}"
export ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-5.0}"

# Full-parameter distillation is more stable with a low LR.
export ACTOR_LR="${ACTOR_LR:-1e-6}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-ifevalg_offpolicy_fullparam}"
export LOCAL_DIR_BASE="${LOCAL_DIR_BASE:-${PROJECT_ROOT}/outputs/ifevalg_offpolicy_fullparam}"

exec bash "${SCRIPT_DIR}/ifevalg_opd.sh" "$@"
