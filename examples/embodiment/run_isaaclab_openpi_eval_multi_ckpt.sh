#!/bin/bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${EMBODIED_PATH}/run_isaaclab_openpi_eval.sh"

if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "Eval script not found: ${EVAL_SCRIPT}"
    exit 1
fi

# Root directory that contains global_step_<N>/actor/model_state_dict
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft_2048/checkpoints}"
# Space-separated checkpoint steps to run sequentially in eval stage.
CKPT_STEPS="${CKPT_STEPS:-25000 30000}"

read -r -a STEP_ARRAY <<< "${CKPT_STEPS}"
if [ "${#STEP_ARRAY[@]}" -eq 0 ]; then
    echo "CKPT_STEPS is empty."
    exit 1
fi

echo "Eval multi-ckpt run"
echo "SFT_CKPT_ROOT: ${SFT_CKPT_ROOT}"
echo "CKPT_STEPS: ${CKPT_STEPS}"

for step in "${STEP_ARRAY[@]}"; do
    if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
        echo "Invalid checkpoint step: ${step}"
        exit 1
    fi

    CKPT_DIR="${SFT_CKPT_ROOT}/global_step_${step}/actor/model_state_dict"
    if [ ! -d "${CKPT_DIR}" ]; then
        echo "Checkpoint directory not found: ${CKPT_DIR}"
        exit 1
    fi

    echo
    echo "=============================================================="
    echo "[eval] step=${step}"
    echo "Checkpoint dir: ${CKPT_DIR}"
    echo "=============================================================="

    CKPT_INPUT="${CKPT_DIR}" bash "${EVAL_SCRIPT}"

    echo "[done][eval] step=${step}"
done

echo
echo "All eval checkpoints finished."
