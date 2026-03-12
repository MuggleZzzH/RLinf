#!/bin/bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
BASE_SCRIPT="${EMBODIED_PATH}/run_isaaclab_openpi_eval_newdata.sh"

export CONFIG_NAME="${CONFIG_NAME:-isaaclab_stack_cube_openpi_eval_newdata_sft50_formal}"
export RUN_NAME="${RUN_NAME:-isaaclab_openpi_eval_newdata_sft50_formal}"
export RESULT_ROOT="${RESULT_ROOT:-${REPO_PATH}/result/isaaclab_openpi_newdata/eval_formal}"

exec bash "${BASE_SCRIPT}"
