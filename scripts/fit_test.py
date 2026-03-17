#!/usr/bin/env python3
"""Fit-test: feed training data through the SFT model and compare predicted
action *chunks* with ground-truth action chunks.

Usage (from repo root, with the venv activated):

    python scripts/fit_test.py \
        --model_path /home/user/cyn_ws/RLinf/checkpoints/realworld_pnp/ \
        --data_dir   /home/user/cyn_ws/results/lerobot_dataset \
        --num_episodes 5 \
        --chunk_size 4

The script will:
  1. Load the model with the *same* transform pipeline used for real-world
     evaluation (config_name = pi0_realworld_pnp).
  2. For each selected episode, iterate through time-steps and construct
     observation dicts that match what the model expects.
  3. Run eval-mode inference (flow matching, no noise at output).
  4. Compare the predicted action chunk (chunk_size steps) against ground-truth
     action chunk, reporting per-dimension MAE and per-chunk MSE.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so that rlinf can be imported.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_model(model_path: str, config_name: str = "pi0_realworld_pnp",
               action_chunk: int = 4, device: str = "cuda"):
    """Load the OpenPi0 model with the full transform pipeline."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "model_path": model_path,
        "model_type": "openpi",
        "openpi": {
            "config_name": config_name,
            "train_expert_only": False,
            "detach_critic_input": True,
            "action_chunk": action_chunk,
        },
        "openpi_data": None,
        "num_action_chunks": action_chunk,
        "state_dim": 19,
        "action_dim": 7,
        "precision": "bf16",
    })

    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg)
    model = model.to(device)
    model.eval()
    return model


def decode_image(img_cell) -> np.ndarray:
    """Decode an image stored as a dict with 'bytes' key (LeRobot v2.0 parquet
    format) into a uint8 HWC numpy array."""
    if isinstance(img_cell, dict):
        raw = img_cell.get("bytes", img_cell.get("data"))
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(pil)
    elif isinstance(img_cell, np.ndarray):
        return img_cell
    else:
        raise TypeError(f"Unexpected image type: {type(img_cell)}")


def build_obs_dict(row: pd.Series, prompt: str) -> dict:
    """Build a single-sample observation dict matching the *data transform*
    input format (i.e. what goes into RealworldInputs after RepackTransform).

    Keys: observation/image, observation/extra_image_0, observation/extra_image_1,
          observation/state, prompt, (actions for GT comparison).
    """
    obs = {
        "observation/image": decode_image(row["image"]),              # HWC uint8
        "observation/extra_image_0": decode_image(row["extra_image_0"]),
        "observation/extra_image_1": decode_image(row["extra_image_1"]),
        "observation/state": np.asarray(row["state"], dtype=np.float32),
        "prompt": prompt,
    }
    return obs


@torch.no_grad()
def run_fit_test(
    model,
    data_dir: str,
    num_episodes: int = 5,
    chunk_size: int = 4,
    device: str = "cuda",
):
    """Run fit test and return per-dimension statistics."""
    import openpi.models.model as _model

    data_path = Path(data_dir)
    meta_dir = data_path / "meta"

    # Read task prompt
    tasks_file = meta_dir / "tasks.jsonl"
    with open(tasks_file) as f:
        tasks = [json.loads(line) for line in f]
    task_prompt = tasks[0]["task"]
    print(f"Task prompt: '{task_prompt}'")

    # Read episode info
    episodes_file = meta_dir / "episodes.jsonl"
    with open(episodes_file) as f:
        episodes = [json.loads(line) for line in f]

    num_episodes = min(num_episodes, len(episodes))
    print(f"Testing {num_episodes} episodes (chunk_size={chunk_size})\n")

    all_pred_chunks = []
    all_gt_chunks = []

    for ep_idx in range(num_episodes):
        ep_info = episodes[ep_idx]
        ep_len = ep_info["length"]
        chunk_idx = ep_info["episode_index"] // 1000
        parquet_path = (
            data_path / "data" / f"chunk-{chunk_idx:03d}"
            / f"episode_{ep_info['episode_index']:06d}.parquet"
        )
        df = pd.read_parquet(parquet_path)
        assert len(df) == ep_len, f"Episode {ep_idx}: expected {ep_len} rows, got {len(df)}"

        print(f"--- Episode {ep_idx} (length={ep_len}) ---")

        # Iterate through time steps where we can form a full chunk of GT actions
        num_valid_steps = ep_len - chunk_size
        if num_valid_steps <= 0:
            print(f"  Skipping: episode too short for chunk_size={chunk_size}")
            continue

        ep_pred_chunks = []
        ep_gt_chunks = []

        for t in range(0, num_valid_steps, max(1, num_valid_steps // 10)):
            # --- Ground truth action chunk ---
            gt_chunk = np.stack(
                [np.asarray(df["actions"].iloc[t + k]) for k in range(chunk_size)],
                axis=0,
            )  # (chunk_size, 7)

            # --- Build single-sample observation ---
            obs = build_obs_dict(df.iloc[t], task_prompt)

            # --- Run model's input_transform per-sample, then batch ---
            # The model.input_transform expects a batched dict (each value has
            # a leading batch dim).  We feed it through as batch_size=1.
            obs_batched = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    obs_batched[k] = torch.from_numpy(v).unsqueeze(0)
                elif isinstance(v, str):
                    obs_batched[k] = [v]
                else:
                    obs_batched[k] = v

            # input_transform handles numpy conversion, per-sample split,
            # state selection, normalization, image tokenization etc.
            # transpose=False because images are already HWC (same as predict_action_batch)
            processed = model.input_transform(obs_batched, transpose=False)
            processed = model.precision_processor(processed)
            observation = _model.Observation.from_dict(processed)

            # --- Inference (eval mode, deterministic) ---
            outputs = model.sample_actions(
                observation, mode="eval", compute_values=False
            )

            # --- Output transform (Unnormalize → RealworldOutputs) ---
            out = model.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )
            pred_actions = out["actions"][0].cpu().numpy()  # (action_chunk, 7)

            # Clip to requested chunk_size (in case action_chunk differs)
            pred_chunk = pred_actions[:chunk_size]

            ep_pred_chunks.append(pred_chunk)
            ep_gt_chunks.append(gt_chunk)

            # Print per-step comparison
            print(f"  step {t:3d}:")
            for k in range(chunk_size):
                gt_str = np.array2string(gt_chunk[k], precision=5, suppress_small=True, floatmode='fixed')
                pr_str = np.array2string(pred_chunk[k], precision=5, suppress_small=True, floatmode='fixed')
                err = np.abs(gt_chunk[k] - pred_chunk[k])
                err_str = np.array2string(err, precision=5, suppress_small=True, floatmode='fixed')
                print(f"    chunk[{k}] GT={gt_str}")
                print(f"             PR={pr_str}")
                print(f"             AE={err_str}")

        if ep_pred_chunks:
            ep_pred = np.stack(ep_pred_chunks)  # (N, chunk_size, 7)
            ep_gt = np.stack(ep_gt_chunks)
            all_pred_chunks.append(ep_pred)
            all_gt_chunks.append(ep_gt)

            # Episode-level stats
            ep_mae = np.mean(np.abs(ep_pred - ep_gt), axis=(0, 1))
            ep_mse = np.mean((ep_pred - ep_gt) ** 2, axis=(0, 1))
            dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
            print(f"\n  Episode {ep_idx} MAE per dim: {dict(zip(dim_names, ep_mae.round(6)))}")
            print(f"  Episode {ep_idx} MSE per dim: {dict(zip(dim_names, ep_mse.round(8)))}")
            print()

    # --- Aggregate stats ---
    if all_pred_chunks:
        all_pred = np.concatenate(all_pred_chunks, axis=0)
        all_gt = np.concatenate(all_gt_chunks, axis=0)

        dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]

        overall_mae = np.mean(np.abs(all_pred - all_gt), axis=(0, 1))
        overall_mse = np.mean((all_pred - all_gt) ** 2, axis=(0, 1))
        overall_rmse = np.sqrt(overall_mse)

        # Also compute per-chunk-position stats
        per_chunk_mae = np.mean(np.abs(all_pred - all_gt), axis=0)  # (chunk_size, 7)

        print("=" * 80)
        print("OVERALL RESULTS")
        print("=" * 80)
        print(f"Total samples: {all_pred.shape[0]}")
        print(f"Chunk size: {chunk_size}")
        print()

        print("Per-dimension MAE (averaged over all chunks):")
        for i, name in enumerate(dim_names):
            print(f"  {name:>4s}: {overall_mae[i]:.6f}")
        print(f"  mean: {overall_mae.mean():.6f}")
        print()

        print("Per-dimension RMSE:")
        for i, name in enumerate(dim_names):
            print(f"  {name:>4s}: {overall_rmse[i]:.6f}")
        print(f"  mean: {overall_rmse.mean():.6f}")
        print()

        print("Per-chunk-position MAE (how error grows with horizon):")
        for k in range(chunk_size):
            row = per_chunk_mae[k]
            print(f"  chunk[{k}]: {dict(zip(dim_names, row.round(6)))}")

        # Summary
        print()
        gt_std = np.std(all_gt, axis=(0, 1))
        print("GT action std (reference scale):")
        for i, name in enumerate(dim_names):
            print(f"  {name:>4s}: {gt_std[i]:.6f}")

        print()
        rel_error = overall_mae / (gt_std + 1e-8)
        print("Relative MAE (MAE / GT_std):")
        for i, name in enumerate(dim_names):
            print(f"  {name:>4s}: {rel_error[i]:.4f}")
        print(f"  mean: {rel_error.mean():.4f}")

        if rel_error[:6].mean() < 0.3:
            print("\n✅ Model fits training data WELL (relative error < 30%)")
        elif rel_error[:6].mean() < 0.6:
            print("\n⚠️  Model fits training data MODERATELY (relative error 30-60%)")
        else:
            print("\n❌ Model fits training data POORLY (relative error > 60%)")
            print("   Possible causes: wrong transforms, bad checkpoint, insufficient training")
    else:
        print("No valid episodes found.")


def main():
    parser = argparse.ArgumentParser(description="Fit test for SFT model")
    parser.add_argument("--model_path", type=str,
                        default="/home/user/cyn_ws/RLinf/checkpoints/realworld_pnp/",
                        help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str,
                        default="/home/user/cyn_ws/results/lerobot_dataset",
                        help="Path to the LeRobot dataset")
    parser.add_argument("--config_name", type=str, default="pi0_realworld_pnp",
                        help="OpenPi config name")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes to test")
    parser.add_argument("--chunk_size", type=int, default=4,
                        help="Action chunk size for comparison")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()

    print("=" * 80)
    print("SFT Fit Test — Action Chunk Comparison")
    print("=" * 80)
    print(f"Model path:   {args.model_path}")
    print(f"Data dir:     {args.data_dir}")
    print(f"Config:       {args.config_name}")
    print(f"Chunk size:   {args.chunk_size}")
    print(f"Device:       {args.device}")
    print()

    print("Loading model...")
    model = load_model(args.model_path, args.config_name, args.chunk_size, args.device)
    print("Model loaded.\n")

    run_fit_test(model, args.data_dir, args.num_episodes, args.chunk_size, args.device)


if __name__ == "__main__":
    main()
