#!/bin/bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
RUNTIME_SETUP_FILE="${REPO_PATH}/examples/common/setup_isaaclab_runtime.sh"

if [ -f "${RUNTIME_SETUP_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${RUNTIME_SETUP_FILE}"
    setup_isaaclab_runtime "${REPO_PATH}"
else
    echo "Runtime setup file not found: ${RUNTIME_SETUP_FILE}"
    exit 1
fi

export REPO_PATH
export EMBODIED_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

CONFIG_NAME="${CONFIG_NAME:-isaaclab_stack_cube_ppo_openpi}"
RUN_NAME="${RUN_NAME:-isaaclab_openpi_rl}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_PATH}/result}"

# Required: OpenPI model directory with safetensors/norm stats.
OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR:-/mnt/project_rlinf_hs/Jiahao/RLinf/checkpoints/torch/pi0_base}"

# Optional: RLinf .pt checkpoint file or directory containing full_weigths.pt/full_weights.pt.
CKPT_INPUT="${CKPT_INPUT:-/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft/checkpoints/global_step_30000/actor/model_state_dict}"

WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rlinf}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-${RUN_NAME}}"

TRAIN_TOTAL_NUM_ENVS="${TRAIN_TOTAL_NUM_ENVS:-16}"
EVAL_TOTAL_NUM_ENVS="${EVAL_TOTAL_NUM_ENVS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-350}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"

resolve_ckpt_path() {
    local input_path="$1"
    if [ -z "${input_path}" ] || [ "${input_path}" = "null" ]; then
        echo "null"
        return
    fi

    if [ -f "${input_path}" ]; then
        echo "${input_path}"
        return
    fi

    if [ -d "${input_path}" ]; then
        if [ -f "${input_path}/full_weigths.pt" ]; then
            echo "${input_path}/full_weigths.pt"
            return
        fi
        if [ -f "${input_path}/full_weights.pt" ]; then
            echo "${input_path}/full_weights.pt"
            return
        fi

        local found
        found="$(find "${input_path}" -maxdepth 4 -type f \( -name "full_weigths.pt" -o -name "full_weights.pt" \) | head -n 1)"
        if [ -n "${found}" ]; then
            echo "${found}"
            return
        fi
    fi

    echo "${input_path}"
}

if [ -z "${OPENPI_MODEL_DIR}" ]; then
    echo "OPENPI_MODEL_DIR is required."
    echo "Example:"
    echo "  OPENPI_MODEL_DIR=/path/to/openpi_model_dir bash examples/embodiment/run_isaaclab_openpi_rl.sh"
    exit 1
fi

resolve_openpi_model_dir() {
    local input_dir="$1"
    if [ ! -d "${input_dir}" ]; then
        echo ""
        return
    fi

    if ls "${input_dir}"/*.safetensors >/dev/null 2>&1 || [ -f "${input_dir}/model.safetensors" ]; then
        echo "${input_dir}"
        return
    fi

    local parent_dir
    parent_dir="$(dirname "${input_dir}")"
    if ls "${parent_dir}"/*.safetensors >/dev/null 2>&1 || [ -f "${parent_dir}/model.safetensors" ]; then
        echo "${parent_dir}"
        return
    fi

    local found_weight
    found_weight="$(find "${input_dir}" -maxdepth 4 -type f \( -name "*.safetensors" -o -name "model.safetensors" \) | head -n 1)"
    if [ -n "${found_weight}" ]; then
        dirname "${found_weight}"
        return
    fi

    echo "${input_dir}"
}

OPENPI_MODEL_DIR_RESOLVED="$(resolve_openpi_model_dir "${OPENPI_MODEL_DIR}")"
if [ -z "${OPENPI_MODEL_DIR_RESOLVED}" ] || [ ! -d "${OPENPI_MODEL_DIR_RESOLVED}" ]; then
    echo "OPENPI_MODEL_DIR does not exist: ${OPENPI_MODEL_DIR}"
    exit 1
fi

CKPT_PATH="$(resolve_ckpt_path "${CKPT_INPUT}")"
if [ "${CKPT_PATH}" != "null" ] && [ ! -f "${CKPT_PATH}" ]; then
    echo "Resolved CKPT_PATH does not exist as file: ${CKPT_PATH}"
    echo "Set CKPT_INPUT=null to skip loading .pt checkpoint."
    exit 1
fi

TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${RESULT_ROOT}/${RUN_NAME}-${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_rl.log"

if [ "${WANDB_ENABLE}" = "1" ]; then
    LOGGER_BACKENDS="[tensorboard,wandb]"
else
    LOGGER_BACKENDS="[tensorboard]"
fi

CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config"
    --config-name "${CONFIG_NAME}"
    "actor.model.model_path=${OPENPI_MODEL_DIR_RESOLVED}"
    "rollout.model.model_path=${OPENPI_MODEL_DIR_RESOLVED}"
    "runner.ckpt_path=${CKPT_PATH}"
    "runner.max_epochs=${MAX_EPOCHS}"
    "runner.logger.log_path=${LOG_DIR}"
    "runner.logger.project_name=${WANDB_PROJECT}"
    "runner.logger.experiment_name=${WANDB_EXP_NAME}"
    "runner.logger.logger_backends=${LOGGER_BACKENDS}"
    "env.train.total_num_envs=${TRAIN_TOTAL_NUM_ENVS}"
    "env.eval.total_num_envs=${EVAL_TOTAL_NUM_ENVS}"
    "env.train.max_episode_steps=${MAX_EPISODE_STEPS}"
    "env.train.max_steps_per_rollout_epoch=${MAX_EPISODE_STEPS}"
    "env.eval.max_episode_steps=${MAX_EPISODE_STEPS}"
    "env.eval.max_steps_per_rollout_epoch=${MAX_EPISODE_STEPS}"
)

echo "Running command:" | tee "${LOG_FILE}"
printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
