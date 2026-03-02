#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quick action comparison plot for a few IsaacLab/OpenPI samples.

This script takes a small number of dataset samples, runs OpenPI action prediction,
and plots per-step curves of:
1) Ground-truth actions (from dataset)
2) Predicted actions (from model)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils import _pytree
from tqdm import tqdm

from rlinf.models import get_model
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.utils.pytree import register_pytree_dataclasses

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required for isaaclab_action_quickplot.py. "
        "Please install it in your runtime environment, e.g. `pip install matplotlib`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GT-vs-Pred action curves for a few IsaacLab samples."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="OpenPI model directory containing safetensors and norm stats.",
    )
    parser.add_argument(
        "--dataset-home",
        type=str,
        required=True,
        help="HF_LEROBOT_HOME parent directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="generated_simdata_full",
        help="LeRobot repo_id under dataset-home.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_isaaclab",
        help="OpenPI dataconfig name.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Skip this many samples before collecting plots.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=32,
        help="Upper bound of iterated batches to avoid endless loops.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Dataloader batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Environment action dimension used for plotting.",
    )
    parser.add_argument(
        "--num-action-chunks",
        type=int,
        default=10,
        help="Model action chunk size.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=4,
        help="OpenPI denoise steps.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="null",
        choices=["null", "bf16", "fp16", "fp32"],
        help="Model precision config.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="null",
        help="Optional RLinf .pt checkpoint path; use 'null' to skip.",
    )
    parser.add_argument(
        "--state-dict-key",
        type=str,
        default="",
        help="Optional state-dict key path inside --ckpt-path.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading --ckpt-path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result/isaaclab_openpi/action_quickplot",
        help="Output root directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name; auto timestamp if empty.",
    )
    return parser.parse_args()


def build_model_cfg(args: argparse.Namespace) -> Any:
    precision = None if args.precision == "null" else args.precision
    cfg_dict = {
        "model_type": "openpi",
        "model_path": args.model_path,
        "precision": precision,
        "num_action_chunks": args.num_action_chunks,
        "action_dim": args.action_dim,
        "num_steps": args.num_steps,
        "use_proprio": True,
        "add_value_head": True,
        "is_lora": False,
        "openpi": {
            "config_name": args.config_name,
            "num_images_in_input": 2,
            "noise_level": 0.5,
            "action_chunk": args.num_action_chunks,
            "num_steps": args.num_steps,
            "train_expert_only": True,
            "action_env_dim": args.action_dim,
            "noise_method": "flow_sde",
            "add_value_head": True,
            "detach_critic_input": True,
        },
    }
    return OmegaConf.create(cfg_dict)


def _is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False
    if not all(isinstance(k, str) for k in obj.keys()):
        return False
    return all(torch.is_tensor(v) for v in obj.values())


def _get_nested_value(root: Mapping[str, Any], path: str) -> Any:
    cur: Any = root
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            raise KeyError(f"Path {path!r} not found in checkpoint.")
        cur = cur[key]
    return cur


def _extract_state_dict(checkpoint: Any, state_dict_key: str = "") -> Mapping[str, Any]:
    if state_dict_key:
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"--state-dict-key is set, but checkpoint type is {type(checkpoint)}."
            )
        state = _get_nested_value(checkpoint, state_dict_key)
        if not _is_state_dict_like(state):
            raise ValueError(
                f"Value at --state-dict-key {state_dict_key!r} is not a tensor state dict."
            )
        return state

    if _is_state_dict_like(checkpoint):
        return checkpoint

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint is not a mapping. Please pass --state-dict-key explicitly."
        )

    candidates = (
        "model_state_dict",
        "state_dict",
        "model",
        "module",
        "actor.model_state_dict",
        "actor.state_dict",
        "checkpoint.model_state_dict",
        "checkpoint.state_dict",
    )
    for key_path in candidates:
        try:
            value = _get_nested_value(checkpoint, key_path)
        except KeyError:
            continue
        if _is_state_dict_like(value):
            return value

    top_keys = ", ".join(list(checkpoint.keys())[:20])
    raise ValueError(
        "Cannot infer state dict from checkpoint. "
        f"Top-level keys: [{top_keys}]. "
        "Please specify --state-dict-key."
    )


def _normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "_forward_module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _to_tensor_on_device(x: Any, device: torch.device) -> Any:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device=device).contiguous()
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device).contiguous()
    return x


def _plot_single_sample(
    true_actions: np.ndarray,
    pred_actions: np.ndarray,
    sample_id: int,
    output_path: pathlib.Path,
) -> dict[str, Any]:
    horizon, dim = true_actions.shape
    x = np.arange(horizon)
    cols = 2
    rows = math.ceil(dim / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.8 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    per_dim_mae: list[float] = []
    per_dim_rmse: list[float] = []
    for d in range(dim):
        ax = axes[d]
        gt = true_actions[:, d]
        pd = pred_actions[:, d]
        err = pd - gt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(np.square(err))))
        per_dim_mae.append(mae)
        per_dim_rmse.append(rmse)

        ax.plot(x, gt, marker="o", markersize=3, linewidth=1.5, label="GT")
        ax.plot(x, pd, marker="x", markersize=3, linewidth=1.5, label="Pred")
        ax.set_title(f"dim {d} | MAE={mae:.4f}, RMSE={rmse:.4f}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Action")
        ax.grid(alpha=0.3)

    for idx in range(dim, axes.shape[0]):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"Sample #{sample_id}: GT vs Pred actions", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    flat_err = pred_actions - true_actions
    return {
        "sample_id": sample_id,
        "horizon": horizon,
        "dim": dim,
        "mae": float(np.mean(np.abs(flat_err))),
        "rmse": float(np.sqrt(np.mean(np.square(flat_err)))),
        "per_dim_mae": per_dim_mae,
        "per_dim_rmse": per_dim_rmse,
        "figure": str(output_path.name),
    }


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"action_quickplot_{timestamp}"
    output_dir = pathlib.Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_LEROBOT_HOME"] = args.dataset_home

    train_config = get_openpi_config(
        args.config_name, model_path=args.model_path, batch_size=args.batch_size
    )
    if train_config.data.repo_id != args.repo_id:
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, repo_id=args.repo_id),
        )
    train_config = dataclasses.replace(train_config, num_workers=args.num_workers)

    model_cfg = build_model_cfg(args)
    model = get_model(model_cfg)
    if model is None:
        raise RuntimeError("Failed to create OpenPI model.")
    if args.ckpt_path != "null":
        ckpt_obj = torch.load(args.ckpt_path, map_location="cpu")
        ckpt_state = _extract_state_dict(ckpt_obj, state_dict_key=args.state_dict_key)
        ckpt_state = _normalize_state_dict_keys(ckpt_state)
        incompatible = model.load_state_dict(ckpt_state, strict=args.strict_load)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if missing:
            print(
                f"[warn] missing keys when loading ckpt: {len(missing)} "
                f"(show first 10): {missing[:10]}"
            )
        if unexpected:
            print(
                f"[warn] unexpected keys when loading ckpt: {len(unexpected)} "
                f"(show first 10): {unexpected[:10]}"
            )
    model.eval()
    device = next(model.parameters()).device

    import openpi.models.model as openpi_model
    import openpi.training.data_loader as openpi_data_loader

    data_loader = openpi_data_loader.create_data_loader(
        train_config, framework="pytorch", shuffle=False
    )

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0.")

    collected: list[dict[str, Any]] = []
    seen_samples = 0
    processed_batches = 0

    progress_total = args.max_batches if args.max_batches > 0 else None
    pbar = tqdm(total=progress_total, desc="Collecting samples", dynamic_ncols=True)

    iterator = iter(data_loader)
    while len(collected) < args.num_samples:
        if args.max_batches > 0 and processed_batches >= args.max_batches:
            break
        try:
            batch = next(iterator)
        except StopIteration:
            break

        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError(
                f"Unexpected dataloader batch type: {type(batch)}; expected (observation, actions)."
            )

        raw_observation, raw_actions = batch[0], batch[1]
        register_pytree_dataclasses(raw_observation)
        observation = _pytree.tree_map(
            lambda x: _to_tensor_on_device(x, device), raw_observation
        )
        actions = _to_tensor_on_device(raw_actions, device)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=device)
        actions = actions.to(torch.float32)

        if isinstance(observation, dict):
            observation_obj = openpi_model.Observation.from_dict(observation)
        else:
            observation_obj = observation

        with torch.no_grad():
            sample_outputs = model.sample_actions(
                observation_obj, mode="eval", compute_values=False
            )
        pred_actions = sample_outputs["actions"].to(torch.float32)
        horizon = min(pred_actions.shape[1], actions.shape[1])

        pred_env = model.output_transform(
            {"actions": pred_actions[:, :horizon], "state": observation_obj.state}
        )["actions"].to(torch.float32)
        true_env = model.output_transform(
            {"actions": actions[:, :horizon], "state": observation_obj.state}
        )["actions"].to(torch.float32)

        env_dim = min(pred_env.shape[2], true_env.shape[2], args.action_dim)
        pred_env = pred_env[:, :, :env_dim].detach().cpu().numpy()
        true_env = true_env[:, :, :env_dim].detach().cpu().numpy()

        batch_size = pred_env.shape[0]
        for i in range(batch_size):
            if seen_samples < args.sample_offset:
                seen_samples += 1
                continue
            sample_id = seen_samples
            fig_path = output_dir / f"sample_{sample_id:06d}_action_compare.png"
            sample_metrics = _plot_single_sample(
                true_actions=true_env[i],
                pred_actions=pred_env[i],
                sample_id=sample_id,
                output_path=fig_path,
            )
            collected.append(sample_metrics)
            seen_samples += 1
            if len(collected) >= args.num_samples:
                break

        processed_batches += 1
        pbar.update(1)

    pbar.close()

    if not collected:
        raise RuntimeError(
            "No samples collected. Check --sample-offset/--max-batches and dataset availability."
        )

    overall_mae = float(np.mean([x["mae"] for x in collected]))
    overall_rmse = float(np.mean([x["rmse"] for x in collected]))
    summary = {
        "run_name": run_name,
        "timestamp": timestamp,
        "model_path": args.model_path,
        "dataset_home": args.dataset_home,
        "repo_id": args.repo_id,
        "config_name": args.config_name,
        "num_samples_requested": args.num_samples,
        "num_samples_collected": len(collected),
        "sample_offset": args.sample_offset,
        "processed_batches": processed_batches,
        "overall_mean_mae": overall_mae,
        "overall_mean_rmse": overall_rmse,
        "samples": collected,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[done] output_dir: {output_dir}")
    print(f"[done] samples_collected: {len(collected)}")
    print(f"[done] overall_mean_mae: {overall_mae:.6f}")
    print(f"[done] overall_mean_rmse: {overall_rmse:.6f}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
