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

CONFIG_NAME="${CONFIG_NAME:-isaaclab_stack_cube_ppo_openpi_senior}"
RUN_NAME="${RUN_NAME:-isaaclab_openpi_rl_senior}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_PATH}/result/isaaclab_openpi/rl_senior}"

OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR:-auto}"
CKPT_INPUT="${CKPT_INPUT:-/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft/checkpoints/global_step_30000/actor/model_state_dict}"

WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rlinf}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-${RUN_NAME}}"

# Senior-style defaults (tier-1 stable profile)
TRAIN_TOTAL_NUM_ENVS="${TRAIN_TOTAL_NUM_ENVS:-24}"
EVAL_TOTAL_NUM_ENVS="${EVAL_TOTAL_NUM_ENVS:-4}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-440}"
TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH="${TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH:-880}"
EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH="${EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH:-440}"
ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-10}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
ACTOR_LR="${ACTOR_LR:-4.0e-6}"
VALUE_LR="${VALUE_LR:-1.0e-5}"

ACTION_CHUNK="${ACTION_CHUNK:-10}"
if ! [[ "${TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH}" =~ ^[0-9]+$ && "${EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH}" =~ ^[0-9]+$ && "${ACTION_CHUNK}" =~ ^[0-9]+$ ]]; then
    echo "TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH / EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH / ACTION_CHUNK must be integers."
    exit 1
fi
if [ $((TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH % ACTION_CHUNK)) -ne 0 ]; then
    echo "TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH (${TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH}) must be divisible by ACTION_CHUNK (${ACTION_CHUNK})."
    exit 1
fi
if [ $((EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH % ACTION_CHUNK)) -ne 0 ]; then
    echo "EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH (${EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH}) must be divisible by ACTION_CHUNK (${ACTION_CHUNK})."
    exit 1
fi

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

infer_openpi_model_dir_from_ckpt() {
    local input_path="$1"
    local candidate_dir=""
    if [ -z "${input_path}" ] || [ "${input_path}" = "null" ]; then
        echo ""
        return
    fi
    if [ -f "${input_path}" ]; then
        candidate_dir="$(dirname "${input_path}")"
    elif [ -d "${input_path}" ]; then
        candidate_dir="${input_path}"
    else
        echo ""
        return
    fi
    resolve_openpi_model_dir "${candidate_dir}"
}

if [ -z "${OPENPI_MODEL_DIR}" ] || [ "${OPENPI_MODEL_DIR}" = "auto" ]; then
    OPENPI_MODEL_DIR_RESOLVED="$(infer_openpi_model_dir_from_ckpt "${CKPT_INPUT}")"
else
    OPENPI_MODEL_DIR_RESOLVED="$(resolve_openpi_model_dir "${OPENPI_MODEL_DIR}")"
fi

if [ -z "${OPENPI_MODEL_DIR_RESOLVED}" ] || [ ! -d "${OPENPI_MODEL_DIR_RESOLVED}" ]; then
    echo "OPENPI_MODEL_DIR does not exist: ${OPENPI_MODEL_DIR}"
    echo "Tip: set OPENPI_MODEL_DIR explicitly, or set CKPT_INPUT to a directory containing safetensors + norm_stats."
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
LOG_FILE="${LOG_DIR}/run_rl_senior.log"

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
    "algorithm.rollout_epoch=${ROLLOUT_EPOCH}"
    "env.train.total_num_envs=${TRAIN_TOTAL_NUM_ENVS}"
    "env.eval.total_num_envs=${EVAL_TOTAL_NUM_ENVS}"
    "env.train.max_episode_steps=${MAX_EPISODE_STEPS}"
    "env.eval.max_episode_steps=${MAX_EPISODE_STEPS}"
    "env.train.max_steps_per_rollout_epoch=${TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH}"
    "env.eval.max_steps_per_rollout_epoch=${EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH}"
    "actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
    "actor.micro_batch_size=${MICRO_BATCH_SIZE}"
    "actor.optim.lr=${ACTOR_LR}"
    "actor.optim.value_lr=${VALUE_LR}"
)

echo "Running command:" | tee "${LOG_FILE}"
printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
