#!/bin/bash

set -euo pipefail

SFT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$SFT_PATH")")"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
SRC_FILE="${SFT_PATH}/train_vla_sft.py"

export REPO_PATH
export EMBODIED_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

CONFIG_NAME="${CONFIG_NAME:-isaaclab_sft_openpi}"
RUN_NAME="${RUN_NAME:-isaaclab_sft_openpi}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_PATH}/result}"

# Required: OpenPI base/SFT model directory (contains safetensors + norm stats)
OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR:-/mnt/project_rlinf_hs/Jiahao/RLinf/checkpoints/torch/pi0_base}"
DATASET_PATH="${DATASET_PATH:-/mnt/qiyuan/zhy/isaaclab_data/generated_simdata_full}"
DATASET_REPO_ID="${DATASET_REPO_ID:-generated_simdata_full}"

# Optional: resume from RLinf SFT checkpoint directory: .../checkpoints/global_step_x/actor
RESUME_DIR="${RESUME_DIR:-null}"

WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rlinf}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-${RUN_NAME}}"

if [ -z "${OPENPI_MODEL_DIR}" ]; then
    echo "OPENPI_MODEL_DIR is required."
    echo "Example:"
    echo "  OPENPI_MODEL_DIR=/path/to/openpi_model_dir bash examples/sft/run_isaaclab_sft_openpi.sh"
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

resolve_lerobot_home() {
    local input_dir="$1"
    if [ ! -d "${input_dir}" ]; then
        echo ""
        return
    fi

    if [ -f "${input_dir}/meta/info.json" ]; then
        dirname "${input_dir}"
        return
    fi

    echo "${input_dir}"
}

LEROBOT_HOME="$(resolve_lerobot_home "${DATASET_PATH}")"
if [ -z "${LEROBOT_HOME}" ] || [ ! -d "${LEROBOT_HOME}" ]; then
    echo "DATASET_PATH does not exist: ${DATASET_PATH}"
    exit 1
fi

EXPECTED_META_PATH="${LEROBOT_HOME}/${DATASET_REPO_ID}/meta/info.json"
if [ ! -f "${EXPECTED_META_PATH}" ]; then
    echo "LeRobot metadata not found: ${EXPECTED_META_PATH}"
    echo "Tip: set DATASET_PATH to either:"
    echo "  1) dataset leaf dir containing meta/info.json; or"
    echo "  2) HF_LEROBOT_HOME parent dir containing ${DATASET_REPO_ID}/meta/info.json."
    exit 1
fi

TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${RESULT_ROOT}/${RUN_NAME}-${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_sft.log"

if [ "${WANDB_ENABLE}" = "1" ]; then
    LOGGER_BACKENDS="[tensorboard,wandb]"
else
    LOGGER_BACKENDS="[tensorboard]"
fi

CMD=(
    python "${SRC_FILE}"
    --config-path "${SFT_PATH}/config"
    --config-name "${CONFIG_NAME}"
    "data.train_data_paths=${LEROBOT_HOME}"
    "actor.model.model_path=${OPENPI_MODEL_DIR_RESOLVED}"
    "runner.logger.log_path=${LOG_DIR}"
    "runner.logger.project_name=${WANDB_PROJECT}"
    "runner.logger.experiment_name=${WANDB_EXP_NAME}"
    "runner.logger.logger_backends=${LOGGER_BACKENDS}"
)

if [ "${RESUME_DIR}" != "null" ]; then
    CMD+=("runner.resume_dir=${RESUME_DIR}")
fi

echo "Running command:" | tee "${LOG_FILE}"
printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
