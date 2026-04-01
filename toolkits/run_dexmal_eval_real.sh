#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${ROBOT_HOST:?Please set ROBOT_HOST}"
: "${CHECKPOINT_DIR:?Please set CHECKPOINT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROBOT_PORT="${ROBOT_PORT:-8000}"
CONFIG_NAME="${CONFIG_NAME:-pi0_dexmal_aloha}"
ASSETS_DIR="${ASSETS_DIR:-$CHECKPOINT_DIR}"
REPO_ID="${REPO_ID:-physical-intelligence/dexmal_aloha}"
ASSET_ID="${ASSET_ID:-fold_towel}"
PROMPT="${PROMPT:-fold the towel}"
ACTION_SCALE="${ACTION_SCALE:-0.35}"
MAX_JOINT_DELTA="${MAX_JOINT_DELTA:-0.04}"
MAX_GRIPPER_DELTA="${MAX_GRIPPER_DELTA:-0.006}"
MAX_CHUNKS="${MAX_CHUNKS:-1}"
LOG_DIR="${LOG_DIR:-./eval_logs}"

ARGS=(
  toolkits/dexmal_eval_bridge.py
  --robot-host "$ROBOT_HOST"
  --robot-port "$ROBOT_PORT"
  --config-name "$CONFIG_NAME"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --assets-dir "$ASSETS_DIR"
  --repo-id "$REPO_ID"
  --asset-id "$ASSET_ID"
  --prompt "$PROMPT"
  --set-model-mode-on-start
  --set-stop-mode-on-exit
  --max-chunks "$MAX_CHUNKS"
  --action-scale "$ACTION_SCALE"
  --max-joint-delta "$MAX_JOINT_DELTA"
  --max-gripper-delta "$MAX_GRIPPER_DELTA"
  --log-dir "$LOG_DIR"
)

if [[ -n "${PYTORCH_DEVICE:-}" ]]; then
  ARGS+=(--pytorch-device "$PYTORCH_DEVICE")
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
