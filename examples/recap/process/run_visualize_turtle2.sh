#!/bin/bash

# XSquare X1 (Turtle2) Robot - Advantage Visualization Script
# This script visualizes the advantage distribution and samples episodes for all 7 datasets.

DATA_ROOT="/mnt/public/songsiqi/data/lerobot"
TAG="turtle2_v1_N10_q30"
OUTPUT_BASE="outputs/visualize_turtle2_${TAG}"
NUM_EPISODES=2  # Number of episodes per dataset to generate videos for
NO_VIDEO=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-video) NO_VIDEO="--no-video" ;;
        --num-episodes) NUM_EPISODES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

DATASETS=(
    "beijing_guqiuyi_20260317_pm_tele_s2m"
    "beijing_guqiuyi_20260318_pm_tele_s2m"
    "beijing_guqiuyi_20260410_pm_tele_s2m"
    "beijing_guqiuyi_20260420_pm_tele_s2m"
    "beijing_guqiuyi_20260325_pm_rollout_s2m"
    "beijing_guqiuyi_20260330_pm_rollout_s2m"
    "beijing_guqiuyi_20260407_pm_rollout_s2m"
)

mkdir -p ${OUTPUT_BASE}

echo "Starting advantage visualization for tag: ${TAG}"
echo "Output directory: ${OUTPUT_BASE}"
if [ -n "$NO_VIDEO" ]; then
    echo "Mode: Distribution only (no video)"
fi

for DS in "${DATASETS[@]}"; do
    echo "=========================================================="
    echo "Processing dataset: ${DS}"
    echo "=========================================================="
    
    DS_PATH="${DATA_ROOT}/${DS}"
    DS_OUT="${OUTPUT_BASE}/${DS}"
    
    python examples/recap/process/visualize_advantage_dataset.py \
        --dataset "${DS_PATH}" \
        --output "${DS_OUT}" \
        --tag "${TAG}" \
        --num-episodes ${NUM_EPISODES} \
        ${NO_VIDEO} \
        --fps 10
done

echo "=========================================================="
echo "All datasets processed! Results are in: ${OUTPUT_BASE}"
