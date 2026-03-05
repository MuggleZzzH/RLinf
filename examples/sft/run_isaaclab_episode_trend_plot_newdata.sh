#!/bin/bash

set -euo pipefail

SFT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$SFT_PATH")")"
RUNTIME_SETUP_FILE="${REPO_PATH}/examples/common/setup_isaaclab_runtime.sh"
PLOT_FILE="${REPO_PATH}/toolkits/eval_scripts_openpi/isaaclab_episode_trend_plot.py"

if [ -f "${RUNTIME_SETUP_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${RUNTIME_SETUP_FILE}"
    setup_isaaclab_runtime "${REPO_PATH}"
else
    echo "Runtime setup file not found: ${RUNTIME_SETUP_FILE}"
    exit 1
fi

if [ ! -f "${PLOT_FILE}" ]; then
    echo "Episode trend plot script not found: ${PLOT_FILE}"
    exit 1
fi

export REPO_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-isaaclab_episode_trend_newdata}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_PATH}/result/isaaclab_openpi_newdata/episode_trend}"

OPENPI_MODEL_DIR="${OPENPI_MODEL_DIR:-/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft_2048/checkpoints/global_step_30000/actor/model_state_dict}"
DATASET_PATH="${DATASET_PATH:-/mnt/qiyuan/zhy/isaaclab_data/generated_simdata_full}"
DATASET_REPO_ID="${DATASET_REPO_ID:-generated_simdata_full}"
CONFIG_NAME="${CONFIG_NAME:-pi0_isaaclab}"

CKPT_INPUT="${CKPT_INPUT:-null}"
STATE_DICT_KEY="${STATE_DICT_KEY:-}"
STRICT_LOAD="${STRICT_LOAD:-0}"

NUM_EPISODES="${NUM_EPISODES:-3}"
EPISODE_IDS="${EPISODE_IDS:-}" # Optional, e.g. "12,77,901"
RANDOM_SEED="${RANDOM_SEED:-42}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-64}"
MAX_FRAMES="${MAX_FRAMES:--1}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-9}"
PROMPT="${PROMPT:-}"

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
LOG_FILE="${LOG_DIR}/run_episode_trend_plot.log"

CMD=(
    python "${PLOT_FILE}"
    --model-path "${OPENPI_MODEL_DIR_RESOLVED}"
    --dataset-home "${LEROBOT_HOME}"
    --repo-id "${DATASET_REPO_ID}"
    --config-name "${CONFIG_NAME}"
    --num-episodes "${NUM_EPISODES}"
    --random-seed "${RANDOM_SEED}"
    --infer-batch-size "${INFER_BATCH_SIZE}"
    --max-frames "${MAX_FRAMES}"
    --smooth-window "${SMOOTH_WINDOW}"
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

if [ -n "${EPISODE_IDS}" ]; then
    CMD+=(--episode-ids "${EPISODE_IDS}")
fi

if [ -n "${PROMPT}" ]; then
    CMD+=(--prompt "${PROMPT}")
fi

echo "Running command:" | tee "${LOG_FILE}"
printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
