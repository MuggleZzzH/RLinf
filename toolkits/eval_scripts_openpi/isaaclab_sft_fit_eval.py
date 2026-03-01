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

"""Offline fitting eval for OpenPI SFT on IsaacLab LeRobot dataset.

This script runs full-dataset inference using dataset observations (image + language +
state) and compares predicted actions against dataset actions.

Outputs:
1. JSON summary metrics
2. CSV per-dimension metrics
3. Diagnostic plots (scatter/hexbin, residual hist, MAE-by-step)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import pathlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
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
        "matplotlib is required for isaaclab_sft_fit_eval.py. "
        "Please install it in your runtime environment, e.g. `pip install matplotlib`."
    ) from exc


@dataclass
class SpaceAccumulator:
    """Running accumulator for action fitting metrics."""

    name: str
    dim: int
    store_points: bool = False

    def __post_init__(self) -> None:
        self.count = np.zeros(self.dim, dtype=np.float64)
        self.sum_true = np.zeros(self.dim, dtype=np.float64)
        self.sum_pred = np.zeros(self.dim, dtype=np.float64)
        self.sum_true_sq = np.zeros(self.dim, dtype=np.float64)
        self.sum_pred_sq = np.zeros(self.dim, dtype=np.float64)
        self.sum_true_pred = np.zeros(self.dim, dtype=np.float64)
        self.sum_abs_err = np.zeros(self.dim, dtype=np.float64)
        self.sum_sq_err = np.zeros(self.dim, dtype=np.float64)

        self.step_sum_abs = np.zeros((0, self.dim), dtype=np.float64)
        self.step_count = np.zeros((0,), dtype=np.float64)

        self.true_points: list[list[np.ndarray]] = [[] for _ in range(self.dim)]
        self.pred_points: list[list[np.ndarray]] = [[] for _ in range(self.dim)]

    def _ensure_step_capacity(self, horizon: int) -> None:
        if horizon <= self.step_count.shape[0]:
            return
        pad_len = horizon - self.step_count.shape[0]
        self.step_sum_abs = np.pad(
            self.step_sum_abs, ((0, pad_len), (0, 0)), mode="constant"
        )
        self.step_count = np.pad(self.step_count, (0, pad_len), mode="constant")

    def update(self, true: torch.Tensor, pred: torch.Tensor) -> None:
        """Update metrics with one batch.

        Args:
            true: Ground-truth actions with shape [B, H, D].
            pred: Predicted actions with shape [B, H, D].
        """
        if true.ndim != 3 or pred.ndim != 3:
            raise ValueError(
                f"{self.name}: expected [B,H,D], got true={tuple(true.shape)}, pred={tuple(pred.shape)}"
            )
        if true.shape != pred.shape:
            raise ValueError(
                f"{self.name}: shape mismatch true={tuple(true.shape)} pred={tuple(pred.shape)}"
            )
        if true.shape[2] != self.dim:
            raise ValueError(
                f"{self.name}: dim mismatch accumulator={self.dim} batch={true.shape[2]}"
            )

        true_np = true.detach().cpu().numpy().astype(np.float64, copy=False)
        pred_np = pred.detach().cpu().numpy().astype(np.float64, copy=False)
        err_np = pred_np - true_np

        batch, horizon, dim = true_np.shape
        flat_true = true_np.reshape(batch * horizon, dim)
        flat_pred = pred_np.reshape(batch * horizon, dim)
        flat_err = err_np.reshape(batch * horizon, dim)

        self.count += flat_true.shape[0]
        self.sum_true += flat_true.sum(axis=0)
        self.sum_pred += flat_pred.sum(axis=0)
        self.sum_true_sq += np.square(flat_true).sum(axis=0)
        self.sum_pred_sq += np.square(flat_pred).sum(axis=0)
        self.sum_true_pred += (flat_true * flat_pred).sum(axis=0)
        self.sum_abs_err += np.abs(flat_err).sum(axis=0)
        self.sum_sq_err += np.square(flat_err).sum(axis=0)

        self._ensure_step_capacity(horizon)
        self.step_sum_abs[:horizon] += np.abs(err_np).sum(axis=0)
        self.step_count[:horizon] += float(batch)

        if self.store_points:
            for d in range(self.dim):
                self.true_points[d].append(flat_true[:, d].astype(np.float32, copy=False))
                self.pred_points[d].append(flat_pred[:, d].astype(np.float32, copy=False))

    def compute_metrics(self) -> dict[str, Any]:
        """Compute overall + per-dimension regression metrics."""
        eps = 1e-12
        mean_true = self.sum_true / np.maximum(self.count, eps)
        mean_pred = self.sum_pred / np.maximum(self.count, eps)

        mae = self.sum_abs_err / np.maximum(self.count, eps)
        mse = self.sum_sq_err / np.maximum(self.count, eps)
        rmse = np.sqrt(mse)

        ss_tot = self.sum_true_sq - self.count * np.square(mean_true)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1.0 - self.sum_sq_err / np.maximum(ss_tot, eps)
        r2 = np.where(ss_tot > eps, r2, np.nan)

        ex = mean_true
        ey = mean_pred
        ex2 = self.sum_true_sq / np.maximum(self.count, eps)
        ey2 = self.sum_pred_sq / np.maximum(self.count, eps)
        exy = self.sum_true_pred / np.maximum(self.count, eps)
        cov = exy - ex * ey
        var_x = ex2 - np.square(ex)
        var_y = ey2 - np.square(ey)
        denom = np.sqrt(np.maximum(var_x * var_y, eps))
        pearson = cov / denom
        pearson = np.where((var_x > eps) & (var_y > eps), pearson, np.nan)

        total_count = float(np.sum(self.count))
        overall_mae = float(np.sum(self.sum_abs_err) / max(total_count, eps))
        overall_mse = float(np.sum(self.sum_sq_err) / max(total_count, eps))
        overall_rmse = float(np.sqrt(overall_mse))
        overall_ss_tot = float(np.nansum(ss_tot))
        overall_r2 = (
            float(1.0 - np.sum(self.sum_sq_err) / max(overall_ss_tot, eps))
            if overall_ss_tot > eps
            else float("nan")
        )

        valid = ~np.isnan(pearson)
        if np.any(valid):
            overall_pearson = float(np.average(pearson[valid], weights=self.count[valid]))
        else:
            overall_pearson = float("nan")

        metrics = {
            "space": self.name,
            "overall": {
                "num_points_per_dim": int(self.count[0]) if self.dim > 0 else 0,
                "num_total_points": int(total_count),
                "mae": overall_mae,
                "mse": overall_mse,
                "rmse": overall_rmse,
                "r2": overall_r2,
                "pearson": overall_pearson,
            },
            "per_dim": [],
        }
        for d in range(self.dim):
            metrics["per_dim"].append(
                {
                    "dim": d,
                    "count": int(self.count[d]),
                    "mae": float(mae[d]),
                    "mse": float(mse[d]),
                    "rmse": float(rmse[d]),
                    "r2": float(r2[d]) if not np.isnan(r2[d]) else float("nan"),
                    "pearson": float(pearson[d])
                    if not np.isnan(pearson[d])
                    else float("nan"),
                    "mean_true": float(mean_true[d]),
                    "mean_pred": float(mean_pred[d]),
                }
            )
        return metrics

    def compute_step_mae(self) -> np.ndarray:
        """Return MAE matrix [H, D]."""
        if self.step_count.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        return self.step_sum_abs / np.maximum(self.step_count[:, None], 1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-dataset fitting eval for OpenPI SFT (IsaacLab)."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to OpenPI SFT checkpoint directory containing safetensors and norm stats.",
    )
    parser.add_argument(
        "--dataset-home",
        type=str,
        required=True,
        help="HF_LEROBOT_HOME parent directory that contains <repo_id>/meta/info.json.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="generated_simdata_full",
        help="LeRobot dataset repo_id under dataset-home.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi0_isaaclab",
        help="OpenPI dataconfig name.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="null",
        help="Optional RLinf .pt full state dict; use 'null' to skip.",
    )
    parser.add_argument(
        "--state-dict-key",
        type=str,
        default="",
        help=(
            "Optional dot path to state dict inside --ckpt-path "
            "(e.g. model_state_dict, actor.model_state_dict)."
        ),
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading --ckpt-path into model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size for dataset traversal.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=-1,
        help="Batches to run; -1 means full dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result/isaaclab_openpi/sft_fit_eval",
        help="Directory to save metrics and figures.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name suffix. Empty means auto timestamp.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action env dim for metrics/plots in env space.",
    )
    parser.add_argument(
        "--num-action-chunks",
        type=int,
        default=10,
        help="Model action chunk size (for config construction).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=4,
        help="OpenPI denoise steps (for config construction).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="null",
        choices=["null", "bf16", "fp16", "fp32"],
        help="Model precision config.",
    )
    parser.add_argument(
        "--disable-env-space",
        action="store_true",
        help="Disable env-space comparison (only model-space metrics).",
    )
    parser.add_argument(
        "--scatter-gridsize",
        type=int,
        default=180,
        help="Hexbin gridsize for scatter plots.",
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


def _write_per_dim_csv(path: pathlib.Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dim",
                "count",
                "mae",
                "mse",
                "rmse",
                "r2",
                "pearson",
                "mean_true",
                "mean_pred",
            ],
        )
        writer.writeheader()
        for row in metrics["per_dim"]:
            writer.writerow(row)


def _plot_scatter_and_hist(
    acc: SpaceAccumulator, out_dir: pathlib.Path, gridsize: int
) -> None:
    if not acc.store_points:
        return
    for d in range(acc.dim):
        if not acc.true_points[d]:
            continue
        true = np.concatenate(acc.true_points[d], axis=0)
        pred = np.concatenate(acc.pred_points[d], axis=0)
        residual = pred - true

        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hexbin(true, pred, gridsize=gridsize, bins="log", mincnt=1, cmap="viridis")
        vmin = float(min(np.min(true), np.min(pred)))
        vmax = float(max(np.max(true), np.max(pred)))
        ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.2)
        ax.set_xlabel("GT action")
        ax.set_ylabel("Pred action")
        ax.set_title(f"{acc.name}: dim {d} (hexbin)")
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("log10(count)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{acc.name}_scatter_dim_{d}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residual, bins=200, color="#2a9d8f", alpha=0.9)
        ax.set_xlabel("Pred - GT")
        ax.set_ylabel("Count")
        ax.set_title(f"{acc.name}: residual histogram dim {d}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{acc.name}_residual_hist_dim_{d}.png", dpi=160)
        plt.close(fig)


def _plot_step_mae(acc: SpaceAccumulator, out_dir: pathlib.Path) -> None:
    mae_by_step = acc.compute_step_mae()
    if mae_by_step.shape[0] == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for d in range(mae_by_step.shape[1]):
        ax.plot(mae_by_step[:, d], linewidth=1.5, label=f"dim_{d}")
    ax.set_xlabel("Action step in chunk")
    ax.set_ylabel("MAE")
    ax.set_title(f"{acc.name}: MAE by step (per dim)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{acc.name}_mae_by_step_per_dim.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    overall = np.mean(mae_by_step, axis=1)
    ax.plot(overall, linewidth=2.0, color="#264653")
    ax.set_xlabel("Action step in chunk")
    ax.set_ylabel("MAE")
    ax.set_title(f"{acc.name}: MAE by step (mean over dims)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{acc.name}_mae_by_step_mean.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    im = ax.imshow(mae_by_step.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Action step in chunk")
    ax.set_ylabel("Action dim")
    ax.set_title(f"{acc.name}: MAE heatmap (dim x step)")
    fig.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(out_dir / f"{acc.name}_mae_heatmap.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"fit_eval_{timestamp}"
    output_dir = pathlib.Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_LEROBOT_HOME"] = args.dataset_home

    # Build OpenPI train config for dataloader.
    train_config = get_openpi_config(
        args.config_name, model_path=args.model_path, batch_size=args.batch_size
    )
    if train_config.data.repo_id != args.repo_id:
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, repo_id=args.repo_id),
        )
    train_config = dataclasses.replace(train_config, num_workers=args.num_workers)

    # Build model.
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

    # Build dataloader.
    import openpi.models.model as openpi_model
    import openpi.training.data_loader as openpi_data_loader

    data_loader = openpi_data_loader.create_data_loader(
        train_config, framework="pytorch", shuffle=False
    )

    has_len = hasattr(data_loader, "__len__")
    total_batches = len(data_loader) if has_len else None
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches) if has_len else args.max_batches

    # Accumulators.
    model_space_acc: SpaceAccumulator | None = None
    env_space_acc: SpaceAccumulator | None = None
    env_space_error: str | None = None

    iterator = iter(data_loader)
    if total_batches is None:
        pbar = tqdm(desc="Fitting Eval (batches)", dynamic_ncols=True)
    else:
        pbar = tqdm(total=total_batches, desc="Fitting Eval (batches)", dynamic_ncols=True)

    batch_idx = 0
    while True:
        if total_batches is not None and batch_idx >= total_batches:
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
        model_dim = min(pred_actions.shape[2], actions.shape[2])
        pred_model = pred_actions[:, :horizon, :model_dim].detach().cpu()
        true_model = actions[:, :horizon, :model_dim].detach().cpu()

        if model_space_acc is None:
            model_space_acc = SpaceAccumulator(
                name="model_space", dim=model_dim, store_points=False
            )
        model_space_acc.update(true=true_model, pred=pred_model)

        if not args.disable_env_space and env_space_error is None:
            try:
                pred_env = model.output_transform(
                    {"actions": pred_actions[:, :horizon], "state": observation_obj.state}
                )["actions"].to(torch.float32)
                true_env = model.output_transform(
                    {"actions": actions[:, :horizon], "state": observation_obj.state}
                )["actions"].to(torch.float32)
                env_dim = min(pred_env.shape[2], true_env.shape[2], args.action_dim)
                pred_env = pred_env[:, :, :env_dim].detach().cpu()
                true_env = true_env[:, :, :env_dim].detach().cpu()

                if env_space_acc is None:
                    env_space_acc = SpaceAccumulator(
                        name="env_space", dim=env_dim, store_points=True
                    )
                env_space_acc.update(true=true_env, pred=pred_env)
            except Exception as exc:  # noqa: BLE001
                env_space_error = str(exc)

        batch_idx += 1
        pbar.update(1)
    pbar.close()

    if batch_idx == 0:
        raise RuntimeError("No batch was processed. Please verify dataset path/repo_id.")

    assert model_space_acc is not None, "model_space_acc should not be None."

    metrics_bundle: dict[str, Any] = {
        "run_name": run_name,
        "timestamp": timestamp,
        "model_path": args.model_path,
        "dataset_home": args.dataset_home,
        "repo_id": args.repo_id,
        "config_name": args.config_name,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "processed_batches": batch_idx,
        "spaces": {},
    }

    model_metrics = model_space_acc.compute_metrics()
    metrics_bundle["spaces"]["model_space"] = model_metrics
    _write_per_dim_csv(output_dir / "model_space_per_dim_metrics.csv", model_metrics)
    _plot_step_mae(model_space_acc, output_dir)

    if env_space_acc is not None:
        env_metrics = env_space_acc.compute_metrics()
        metrics_bundle["spaces"]["env_space"] = env_metrics
        _write_per_dim_csv(output_dir / "env_space_per_dim_metrics.csv", env_metrics)
        _plot_scatter_and_hist(env_space_acc, output_dir, gridsize=args.scatter_gridsize)
        _plot_step_mae(env_space_acc, output_dir)
    if env_space_error is not None:
        metrics_bundle["env_space_error"] = env_space_error

    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_bundle, f, indent=2, ensure_ascii=False)

    print(f"[done] processed_batches={batch_idx}")
    print(f"[done] results saved to: {output_dir}")
    print(
        "[done] model_space overall: "
        f"{json.dumps(model_metrics['overall'], ensure_ascii=False)}"
    )
    if env_space_acc is not None:
        print(
            "[done] env_space overall: "
            f"{json.dumps(env_metrics['overall'], ensure_ascii=False)}"
        )
    elif env_space_error is not None:
        print(f"[warn] env_space skipped due to error: {env_space_error}")


if __name__ == "__main__":
    main()
