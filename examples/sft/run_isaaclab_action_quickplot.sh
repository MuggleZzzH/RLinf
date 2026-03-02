#!/bin/bash

set -euo pipefail

SFT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$SFT_PATH")")"
RUNTIME_SETUP_FILE="${REPO_PATH}/examples/common/setup_isaaclab_runtime.sh"
PLOT_FILE="${REPO_PATH}/toolkits/eval_scripts_openpi/isaaclab_action_quickplot.py"

if [ -f "${RUNTIME_SETUP_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${RUNTIME_SETUP_FILE}"
    setup_isaaclab_runtime "${REPO_PATH}"
else
    echo "Runtime setup file not found: ${RUNTIME_SETUP_FILE}"
    exit 1
fi

if [ ! -f "${PLOT_FILE}" ]; then
    echo "Quickplot script not found: ${PLOT_FILE}"
    exit 1
fi

export REPO_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-isaaclab_action_quickplot}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_PATH}/result/isaaclab_openpi/action_quickplot}"

OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR:-/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft_2048/checkpoints/global_step_30000/actor/model_state_dict}"
DATASET_PATH="${DATASET_PATH:-/mnt/qiyuan/zhy/isaaclab_data/generated_simdata_full}"
DATASET_REPO_ID="${DATASET_REPO_ID:-generated_simdata_full}"
CONFIG_NAME="${CONFIG_NAME:-pi0_isaaclab}"

CKPT_INPUT="${CKPT_INPUT:-null}"
STATE_DICT_KEY="${STATE_DICT_KEY:-}"
STRICT_LOAD="${STRICT_LOAD:-0}"

NUM_SAMPLES="${NUM_SAMPLES:-6}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
MAX_BATCHES="${MAX_BATCHES:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

ACTION_DIM="${ACTION_DIM:-7}"
NUM_ACTION_CHUNKS="${NUM_ACTION_CHUNKS:-10}"
NUM_STEPS="${NUM_STEPS:-4}"
PRECISION="${PRECISION:-null}"

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

    echo "${input_dir}"
}

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
        if [ -f "${input_path}/full_weights.pt" ]; then
            echo "${input_path}/full_weights.pt"
            return
        fi
        if [ -f "${input_path}/full_weigths.pt" ]; then
            echo "${input_path}/full_weigths.pt"
            return
        fi
    fi
    echo "${input_path}"
}

OPENPI_MODEL_DIR_RESOLVED="$(resolve_openpi_model_dir "${OPENPI_MODEL_DIR}")"
if [ -z "${OPENPI_MODEL_DIR_RESOLVED}" ] || [ ! -d "${OPENPI_MODEL_DIR_RESOLVED}" ]; then
    echo "OPENPI_MODEL_DIR does not exist: ${OPENPI_MODEL_DIR}"
    exit 1
fi

LEROBOT_HOME="$(resolve_lerobot_home "${DATASET_PATH}")"
if [ -z "${LEROBOT_HOME}" ] || [ ! -d "${LEROBOT_HOME}" ]; then
    echo "DATASET_PATH does not exist: ${DATASET_PATH}"
    exit 1
fi

EXPECTED_META_PATH="${LEROBOT_HOME}/${DATASET_REPO_ID}/meta/info.json"
if [ ! -f "${EXPECTED_META_PATH}" ]; then
    echo "LeRobot metadata not found: ${EXPECTED_META_PATH}"
    exit 1
fi

CKPT_PATH="$(resolve_ckpt_path "${CKPT_INPUT}")"
if [ "${CKPT_PATH}" != "null" ] && [ ! -f "${CKPT_PATH}" ]; then
    echo "Resolved CKPT_PATH does not exist as file: ${CKPT_PATH}"
    exit 1
fi

TIMESTAMP="$(date +'%Y%m%d-%H%M%S')"
LOG_DIR="${RESULT_ROOT}/${RUN_NAME}-${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_action_quickplot.log"

CMD=(
    python "${PLOT_FILE}"
    --model-path "${OPENPI_MODEL_DIR_RESOLVED}"
    --dataset-home "${LEROBOT_HOME}"
    --repo-id "${DATASET_REPO_ID}"
    --config-name "${CONFIG_NAME}"
    --num-samples "${NUM_SAMPLES}"
    --sample-offset "${SAMPLE_OFFSET}"
    --max-batches "${MAX_BATCHES}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --action-dim "${ACTION_DIM}"
    --num-action-chunks "${NUM_ACTION_CHUNKS}"
    --num-steps "${NUM_STEPS}"
    --precision "${PRECISION}"
    --output-dir "${RESULT_ROOT}"
    --run-name "${RUN_NAME}-${TIMESTAMP}"
)

if [ "${CKPT_PATH}" != "null" ]; then
    CMD+=(--ckpt-path "${CKPT_PATH}")
fi

if [ -n "${STATE_DICT_KEY}" ]; then
    CMD+=(--state-dict-key "${STATE_DICT_KEY}")
fi

if [ "${STRICT_LOAD}" = "1" ]; then
    CMD+=(--strict-load)
fi

echo "Running command:" | tee "${LOG_FILE}"
printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
