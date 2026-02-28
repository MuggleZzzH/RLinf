#!/bin/bash

set -euo pipefail

setup_isaaclab_runtime() {
    local repo_path="$1"
    local auto_setup="${AUTO_SETUP_ENV:-1}"

    if [ "${auto_setup}" != "1" ]; then
        echo "[env-setup] AUTO_SETUP_ENV=${auto_setup}, skip auto environment setup."
        return 0
    fi

    local conda_setup="${ISAACLAB_CONDA_SETUP:-/mnt/qiyuan/zhy/nv_rlinf/IsaacLab/_isaac_sim/setup_conda_env.sh}"
    local runtime_setup="${ISAACLAB_RUNTIME_SETUP:-/mnt/project_rlinf_hs/Jiahao/IsaacLab配置计划/tools/run_env.sh}"

    if [ -f "${conda_setup}" ]; then
        # shellcheck disable=SC1090
        source "${conda_setup}"
        echo "[env-setup] sourced ISAACLAB_CONDA_SETUP: ${conda_setup}"
    else
        echo "[env-setup] ISAACLAB_CONDA_SETUP not found, skip: ${conda_setup}"
    fi

    if [ -f "${runtime_setup}" ]; then
        # shellcheck disable=SC1090
        source "${runtime_setup}"
        echo "[env-setup] sourced ISAACLAB_RUNTIME_SETUP: ${runtime_setup}"
    else
        echo "[env-setup] ISAACLAB_RUNTIME_SETUP not found, skip: ${runtime_setup}"
    fi

    if [ -n "${VIRTUAL_ENV:-}" ]; then
        echo "[env-setup] using active venv: ${VIRTUAL_ENV}"
        return 0
    fi

    local -a venv_candidates=()
    if [ -n "${RLINF_VENV_PATH:-}" ]; then
        venv_candidates+=("${RLINF_VENV_PATH}")
    fi
    venv_candidates+=("${repo_path}/.venv")
    venv_candidates+=("/mnt/project_rlinf_hs/Jiahao/RLinf/.venv")

    local activated=0
    local venv_dir
    for venv_dir in "${venv_candidates[@]}"; do
        if [ -f "${venv_dir}/bin/activate" ]; then
            # shellcheck disable=SC1090
            source "${venv_dir}/bin/activate"
            echo "[env-setup] activated venv: ${venv_dir}"
            activated=1
            break
        fi
    done

    if [ "${activated}" -ne 1 ]; then
        echo "[env-setup] failed to activate venv. Set RLINF_VENV_PATH=/path/to/.venv or activate manually."
        return 1
    fi
}

