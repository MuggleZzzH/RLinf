#!/bin/bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
EVAL_SCRIPT="${EMBODIED_PATH}/run_isaaclab_openpi_eval.sh"
RL_SCRIPT="${EMBODIED_PATH}/run_isaaclab_openpi_rl.sh"

if [ ! -f "${EVAL_SCRIPT}" ] || [ ! -f "${RL_SCRIPT}" ]; then
    echo "Required scripts not found:"
    echo "  ${EVAL_SCRIPT}"
    echo "  ${RL_SCRIPT}"
    exit 1
fi

# Root directory that contains global_step_<N>/actor/model_state_dict
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft_2048/checkpoints}"

# Space-separated checkpoint steps to run sequentially.
CKPT_STEPS="${CKPT_STEPS:-25000 30000}"

# 1: run stage; 0: skip stage.
RUN_EVAL="${RUN_EVAL:-1}"
RUN_RL="${RUN_RL:-1}"

WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rlinf}"

# "auto" means infer model dir from CKPT_INPUT in child scripts.
OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR:-auto}"
ACTION_CHUNK="${ACTION_CHUNK:-10}"

EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-isaaclab_stack_cube_openpi_eval}"
EVAL_TOTAL_NUM_ENVS="${EVAL_TOTAL_NUM_ENVS:-8}"
EVAL_MAX_EPISODE_STEPS="${EVAL_MAX_EPISODE_STEPS:-440}"
EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH:-1}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-false}"

RL_CONFIG_NAME="${RL_CONFIG_NAME:-isaaclab_stack_cube_ppo_openpi}"
RL_TRAIN_TOTAL_NUM_ENVS="${RL_TRAIN_TOTAL_NUM_ENVS:-8}"
RL_EVAL_TOTAL_NUM_ENVS="${RL_EVAL_TOTAL_NUM_ENVS:-4}"
RL_MAX_EPISODE_STEPS="${RL_MAX_EPISODE_STEPS:-350}"
RL_MAX_EPOCHS="${RL_MAX_EPOCHS:-30}"

assert_divisible() {
    local value="$1"
    local divisor="$2"
    local name="$3"
    if ! [[ "${value}" =~ ^[0-9]+$ && "${divisor}" =~ ^[0-9]+$ ]]; then
        echo "${name} and ACTION_CHUNK must be integers."
        exit 1
    fi
    if [ "${divisor}" -le 0 ]; then
        echo "ACTION_CHUNK must be > 0."
        exit 1
    fi
    if [ $((value % divisor)) -ne 0 ]; then
        echo "${name} (${value}) must be divisible by ACTION_CHUNK (${divisor})."
        exit 1
    fi
}

assert_divisible "${EVAL_MAX_EPISODE_STEPS}" "${ACTION_CHUNK}" "EVAL_MAX_EPISODE_STEPS"
assert_divisible "${RL_MAX_EPISODE_STEPS}" "${ACTION_CHUNK}" "RL_MAX_EPISODE_STEPS"

if [ "${RUN_EVAL}" != "1" ] && [ "${RUN_RL}" != "1" ]; then
    echo "Both RUN_EVAL and RUN_RL are disabled. Nothing to do."
    exit 0
fi

read -r -a STEP_ARRAY <<< "${CKPT_STEPS}"
if [ "${#STEP_ARRAY[@]}" -eq 0 ]; then
    echo "CKPT_STEPS is empty."
    exit 1
fi

echo "SFT_CKPT_ROOT: ${SFT_CKPT_ROOT}"
echo "CKPT_STEPS: ${CKPT_STEPS}"
echo "RUN_EVAL=${RUN_EVAL}, RUN_RL=${RUN_RL}"
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
    echo "Running checkpoint step ${step}"
    echo "Checkpoint dir: ${CKPT_DIR}"
    echo "=============================================================="

    if [ "${RUN_EVAL}" = "1" ]; then
        echo "[stage: eval] step=${step}"
        CKPT_INPUT="${CKPT_DIR}" \
        OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR}" \
        CONFIG_NAME="${EVAL_CONFIG_NAME}" \
        RUN_NAME="isaaclab_openpi_eval_step${step}" \
        WANDB_ENABLE="${WANDB_ENABLE}" \
        WANDB_PROJECT="${WANDB_PROJECT}" \
        WANDB_EXP_NAME="isaaclab_openpi_eval_step${step}" \
        TOTAL_NUM_ENVS="${EVAL_TOTAL_NUM_ENVS}" \
        MAX_EPISODE_STEPS="${EVAL_MAX_EPISODE_STEPS}" \
        EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH}" \
        SAVE_VIDEO="${EVAL_SAVE_VIDEO}" \
        ACTION_CHUNK="${ACTION_CHUNK}" \
        bash "${EVAL_SCRIPT}"
    fi

    if [ "${RUN_RL}" = "1" ]; then
        echo "[stage: rl] step=${step}"
        CKPT_INPUT="${CKPT_DIR}" \
        OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR}" \
        CONFIG_NAME="${RL_CONFIG_NAME}" \
        RUN_NAME="isaaclab_openpi_rl_step${step}" \
        WANDB_ENABLE="${WANDB_ENABLE}" \
        WANDB_PROJECT="${WANDB_PROJECT}" \
        WANDB_EXP_NAME="isaaclab_openpi_rl_step${step}" \
        TRAIN_TOTAL_NUM_ENVS="${RL_TRAIN_TOTAL_NUM_ENVS}" \
        EVAL_TOTAL_NUM_ENVS="${RL_EVAL_TOTAL_NUM_ENVS}" \
        MAX_EPISODE_STEPS="${RL_MAX_EPISODE_STEPS}" \
        MAX_EPOCHS="${RL_MAX_EPOCHS}" \
        bash "${RL_SCRIPT}"
    fi

    echo "[done] step=${step}"
done

echo
echo "All requested checkpoints finished."
