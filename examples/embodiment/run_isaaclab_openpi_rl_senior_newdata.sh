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

CONFIG_NAME="${CONFIG_NAME:-isaaclab_stack_cube_ppo_openpi_senior_newdata}"
RUN_NAME="${RUN_NAME:-isaaclab_openpi_rl_senior_newdata}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_PATH}/result/isaaclab_openpi_newdata/rl_senior}"

OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR:-auto}"
CKPT_INPUT="${CKPT_INPUT:-/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft/checkpoints/global_step_30000/actor/model_state_dict}"

WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rlinf}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-${RUN_NAME}}"

# Keep training hyper-parameters in YAML.
# This launcher only sets runtime paths/logging, not RL/SFT core knobs.
# If you need temporary local overrides, pass them via EXTRA_HYDRA_ARGS.
EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:-}"

check_rollout_batch_divisibility() {
    local config_file="$1"
    local extra_args="$2"
    python - "${config_file}" "${extra_args}" <<'PY'
import shlex
import sys

from omegaconf import OmegaConf

config_file = sys.argv[1]
extra_args = sys.argv[2]
cfg = OmegaConf.load(config_file)

params = {
    "num_envs": int(cfg.env.train.total_num_envs),
    "steps_per_rollout": int(cfg.env.train.max_steps_per_rollout_epoch),
    "num_action_chunks": int(cfg.actor.model.num_action_chunks),
    "rollout_epoch": int(cfg.algorithm.rollout_epoch),
    "global_batch_size": int(cfg.actor.global_batch_size),
}

override_key_to_var = {
    "env.train.total_num_envs": "num_envs",
    "env.train.max_steps_per_rollout_epoch": "steps_per_rollout",
    "actor.model.num_action_chunks": "num_action_chunks",
    "algorithm.rollout_epoch": "rollout_epoch",
    "actor.global_batch_size": "global_batch_size",
}

for token in shlex.split(extra_args):
    if "=" not in token:
        continue
    key, value = token.split("=", 1)
    if key not in override_key_to_var:
        continue
    var_name = override_key_to_var[key]
    try:
        parsed_value = int(float(value))
    except ValueError as exc:
        raise ValueError(f"Override {key}={value!r} is not a valid numeric value.") from exc
    params[var_name] = parsed_value

num_envs = params["num_envs"]
steps_per_rollout = params["steps_per_rollout"]
num_action_chunks = params["num_action_chunks"]
rollout_epoch = params["rollout_epoch"]
global_batch_size = params["global_batch_size"]

if num_action_chunks <= 0:
    raise ValueError(f"actor.model.num_action_chunks must be positive, got {num_action_chunks}.")
if steps_per_rollout % num_action_chunks != 0:
    raise ValueError(
        "env.train.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks, "
        f"got {steps_per_rollout} and {num_action_chunks}."
    )
if global_batch_size <= 0:
    raise ValueError(f"actor.global_batch_size must be positive, got {global_batch_size}.")

rollout_size = num_envs * (steps_per_rollout // num_action_chunks) * rollout_epoch
if rollout_size % global_batch_size != 0:
    raise ValueError(
        "rollout_size is not divisible by global_batch_size. "
        f"rollout_size={rollout_size}, global_batch_size={global_batch_size}. "
        "Adjust env.train.total_num_envs / env.train.max_steps_per_rollout_epoch / "
        "algorithm.rollout_epoch / actor.global_batch_size."
    )

print(
    f"[rollout-check] OK: rollout_size={rollout_size}, global_batch_size={global_batch_size}, "
    f"num_updates_per_rollout={rollout_size // global_batch_size}"
)
PY
}

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

CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
if [ -f "${CONFIG_FILE}" ]; then
    check_rollout_batch_divisibility "${CONFIG_FILE}" "${EXTRA_HYDRA_ARGS}"
else
    echo "Warning: config file not found for rollout divisibility check: ${CONFIG_FILE}"
fi

TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${RESULT_ROOT}/${RUN_NAME}-${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_rl_senior_newdata.log"

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
    "runner.logger.log_path=${LOG_DIR}"
    "runner.logger.project_name=${WANDB_PROJECT}"
    "runner.logger.experiment_name=${WANDB_EXP_NAME}"
    "runner.logger.logger_backends=${LOGGER_BACKENDS}"
)

if [ -n "${EXTRA_HYDRA_ARGS}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARR=(${EXTRA_HYDRA_ARGS})
    CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

echo "Running command:" | tee "${LOG_FILE}"
printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
