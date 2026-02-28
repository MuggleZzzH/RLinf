#!/bin/bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_SCRIPT="${EMBODIED_PATH}/run_isaaclab_openpi_rl.sh"

if [ ! -f "${RL_SCRIPT}" ]; then
    echo "RL script not found: ${RL_SCRIPT}"
    exit 1
fi

# Root directory that contains global_step_<N>/actor/model_state_dict
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft_2048/checkpoints}"
# Space-separated checkpoint steps to run sequentially in rl stage.
CKPT_STEPS="${CKPT_STEPS:-25000 30000}"

WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rlinf}"

# "auto" means infer model dir from CKPT_INPUT in child script.
OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR:-auto}"
ACTION_CHUNK="${ACTION_CHUNK:-10}"

CONFIG_NAME="${CONFIG_NAME:-isaaclab_stack_cube_ppo_openpi}"
TRAIN_TOTAL_NUM_ENVS="${TRAIN_TOTAL_NUM_ENVS:-8}"
EVAL_TOTAL_NUM_ENVS="${EVAL_TOTAL_NUM_ENVS:-4}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-350}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"

if ! [[ "${MAX_EPISODE_STEPS}" =~ ^[0-9]+$ && "${ACTION_CHUNK}" =~ ^[0-9]+$ ]]; then
    echo "MAX_EPISODE_STEPS and ACTION_CHUNK must be integers."
    exit 1
fi
if [ "${ACTION_CHUNK}" -le 0 ]; then
    echo "ACTION_CHUNK must be > 0."
    exit 1
fi
if [ $((MAX_EPISODE_STEPS % ACTION_CHUNK)) -ne 0 ]; then
    echo "MAX_EPISODE_STEPS (${MAX_EPISODE_STEPS}) must be divisible by ACTION_CHUNK (${ACTION_CHUNK})."
    exit 1
fi

read -r -a STEP_ARRAY <<< "${CKPT_STEPS}"
if [ "${#STEP_ARRAY[@]}" -eq 0 ]; then
    echo "CKPT_STEPS is empty."
    exit 1
fi

echo "RL multi-ckpt run"
echo "SFT_CKPT_ROOT: ${SFT_CKPT_ROOT}"
echo "CKPT_STEPS: ${CKPT_STEPS}"
echo "OPENPI_MODEL_DIR=${OPENPI_MODEL_DIR}, ACTION_CHUNK=${ACTION_CHUNK}"

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
    echo "[rl] step=${step}"
    echo "Checkpoint dir: ${CKPT_DIR}"
    echo "=============================================================="

    CKPT_INPUT="${CKPT_DIR}" \
    OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR}" \
    CONFIG_NAME="${CONFIG_NAME}" \
    RUN_NAME="isaaclab_openpi_rl_step${step}" \
    WANDB_ENABLE="${WANDB_ENABLE}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_EXP_NAME="isaaclab_openpi_rl_step${step}" \
    TRAIN_TOTAL_NUM_ENVS="${TRAIN_TOTAL_NUM_ENVS}" \
    EVAL_TOTAL_NUM_ENVS="${EVAL_TOTAL_NUM_ENVS}" \
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
    MAX_EPOCHS="${MAX_EPOCHS}" \
    ACTION_CHUNK="${ACTION_CHUNK}" \
    bash "${RL_SCRIPT}"

    echo "[done][rl] step=${step}"
done

echo
echo "All rl checkpoints finished."
