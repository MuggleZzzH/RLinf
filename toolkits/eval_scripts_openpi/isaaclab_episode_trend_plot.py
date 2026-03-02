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

"""Random-episode trend plotting for IsaacLab/OpenPI.

Randomly select a few full episodes, run model prediction frame-by-frame (batched),
and visualize trend-level fitting quality with compact figures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

import imageio
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from rlinf.models import get_model

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required for isaaclab_episode_trend_plot.py. "
        "Please install it in your runtime environment, e.g. `pip install matplotlib`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample episodes and plot full-episode action fitting trends."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-home", type=str, required=True)
    parser.add_argument("--repo-id", type=str, default="generated_simdata_full")
    parser.add_argument("--config-name", type=str, default="pi0_isaaclab")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument(
        "--episode-ids",
        type=str,
        default="",
        help="Optional comma-separated episode ids, e.g. 12,77,901. Overrides random sampling.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--infer-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional cap per episode; -1 means full episode.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt override. Empty means auto-read from meta/tasks.jsonl.",
    )
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--num-action-chunks", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument(
        "--precision",
        type=str,
        default="null",
        choices=["null", "bf16", "fp16", "fp32"],
    )
    parser.add_argument("--ckpt-path", type=str, default="null")
    parser.add_argument("--state-dict-key", type=str, default="")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Moving-average window for per-frame MAE trend.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result/isaaclab_openpi/episode_trend",
    )
    parser.add_argument("--run-name", type=str, default="")
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


def _read_video_frames(video_path: pathlib.Path) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    try:
        frames = imageio.v3.imread(str(video_path))
        if frames.ndim == 3:
            frames = frames[None, ...]
        return np.asarray(frames)
    except Exception:  # noqa: BLE001
        reader = imageio.get_reader(str(video_path))
        frames_list = [np.asarray(frame) for frame in reader]
        reader.close()
        if not frames_list:
            raise RuntimeError(f"Video is empty: {video_path}")
        return np.stack(frames_list, axis=0)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x
    window = min(window, x.size)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def _discover_prompt(dataset_root: pathlib.Path, override: str) -> str:
    if override:
        return override
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with tasks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in ("task", "task_description", "prompt", "instruction"):
                    if key in obj and isinstance(obj[key], str) and obj[key]:
                        return obj[key]
                for value in obj.values():
                    if isinstance(value, str) and value:
                        return value
                break
    return (
        "Pick up the red cube and place it on top of the blue cube, "
        "then pick up the green cube and place it on top of the red cube."
    )


def _parse_episode_ids(episode_ids_arg: str) -> list[int]:
    if not episode_ids_arg.strip():
        return []
    out = []
    for part in episode_ids_arg.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _discover_available_episode_ids(data_dir: pathlib.Path) -> list[int]:
    ids: list[int] = []
    for p in sorted(data_dir.glob("episode_*.parquet")):
        stem = p.stem
        try:
            ids.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return ids


def _select_episode_ids(args: argparse.Namespace, available_ids: list[int]) -> list[int]:
    requested = _parse_episode_ids(args.episode_ids)
    if requested:
        missing = [eid for eid in requested if eid not in set(available_ids)]
        if missing:
            raise ValueError(f"Requested episode ids not found: {missing}")
        return requested

    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be > 0.")
    if args.num_episodes > len(available_ids):
        raise ValueError(
            f"--num-episodes={args.num_episodes} exceeds available episodes={len(available_ids)}."
        )
    rng = np.random.default_rng(args.random_seed)
    chosen = rng.choice(np.asarray(available_ids), size=args.num_episodes, replace=False)
    return sorted([int(x) for x in chosen.tolist()])


def _episode_paths(dataset_root: pathlib.Path, episode_id: int) -> dict[str, pathlib.Path]:
    epi_name = f"episode_{episode_id:06d}"
    return {
        "parquet": dataset_root / "data" / "chunk-000" / f"{epi_name}.parquet",
        "table_video": dataset_root
        / "videos"
        / "chunk-000"
        / "observation.images.table"
        / f"{epi_name}.mp4",
        "wrist_video": dataset_root
        / "videos"
        / "chunk-000"
        / "observation.images.wrist"
        / f"{epi_name}.mp4",
    }


def _load_episode_arrays(
    dataset_root: pathlib.Path,
    episode_id: int,
    max_frames: int,
) -> dict[str, np.ndarray]:
    paths = _episode_paths(dataset_root, episode_id)
    if not paths["parquet"].exists():
        raise FileNotFoundError(f"Parquet not found: {paths['parquet']}")
    df = pd.read_parquet(paths["parquet"])
    if "observation.state" not in df.columns or "action" not in df.columns:
        raise KeyError(
            f"Required columns not found in {paths['parquet']}. "
            f"Existing columns: {list(df.columns)}"
        )

    states = np.stack(df["observation.state"].to_numpy(), axis=0).astype(np.float32)
    actions = np.stack(df["action"].to_numpy(), axis=0).astype(np.float32)
    table_frames = _read_video_frames(paths["table_video"]).astype(np.uint8)
    wrist_frames = _read_video_frames(paths["wrist_video"]).astype(np.uint8)

    length = min(states.shape[0], actions.shape[0], table_frames.shape[0], wrist_frames.shape[0])
    if max_frames > 0:
        length = min(length, max_frames)
    if length <= 0:
        raise RuntimeError(f"Episode {episode_id} has no valid frames.")

    return {
        "states": states[:length],
        "actions": actions[:length],
        "table_frames": table_frames[:length],
        "wrist_frames": wrist_frames[:length],
    }


def _predict_episode_first_step_actions(
    model: Any,
    states: np.ndarray,
    table_frames: np.ndarray,
    wrist_frames: np.ndarray,
    prompt: str,
    infer_batch_size: int,
    action_dim: int,
) -> np.ndarray:
    total = states.shape[0]
    preds: list[np.ndarray] = []
    for start in range(0, total, infer_batch_size):
        end = min(start + infer_batch_size, total)
        # OpenPI action model expects tensor inputs in env_obs.
        env_obs = {
            "main_images": torch.from_numpy(
                np.ascontiguousarray(table_frames[start:end])
            ),
            "wrist_images": torch.from_numpy(
                np.ascontiguousarray(wrist_frames[start:end])
            ),
            "states": torch.from_numpy(np.ascontiguousarray(states[start:end])),
            "task_descriptions": [prompt] * (end - start),
        }
        with torch.no_grad():
            actions, _ = model.predict_action_batch(
                env_obs, mode="eval", compute_values=False
            )
        # actions shape: [B, action_chunk, action_dim]
        first_step = np.asarray(actions[:, 0, :action_dim], dtype=np.float32)
        preds.append(first_step)
    return np.concatenate(preds, axis=0)


def _plot_episode_overview(
    episode_id: int,
    true_actions: np.ndarray,
    pred_actions: np.ndarray,
    out_path: pathlib.Path,
    smooth_window: int,
) -> dict[str, Any]:
    err = pred_actions - true_actions
    abs_err = np.abs(err)
    per_frame_mae = np.mean(abs_err, axis=1)
    per_frame_mae_smooth = _moving_average(per_frame_mae, smooth_window)
    per_dim_mae = np.mean(abs_err, axis=0)
    per_dim_rmse = np.sqrt(np.mean(np.square(err), axis=0))

    vmin = float(min(true_actions.min(), pred_actions.min()))
    vmax = float(max(true_actions.max(), pred_actions.max()))
    emax = float(abs_err.max()) if abs_err.size > 0 else 1.0

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=False)
    x = np.arange(true_actions.shape[0])

    axes[0].plot(x, per_frame_mae, color="#5e60ce", alpha=0.5, linewidth=1.2, label="MAE")
    axes[0].plot(
        x,
        per_frame_mae_smooth,
        color="#3a0ca3",
        linewidth=2.0,
        label=f"MAE smooth(w={smooth_window})",
    )
    axes[0].set_title(
        f"Episode {episode_id}: frame-wise fitting trend | "
        f"mean MAE={np.mean(per_frame_mae):.4f}, mean RMSE={np.sqrt(np.mean(np.square(err))):.4f}"
    )
    axes[0].set_xlabel("Frame index")
    axes[0].set_ylabel("MAE")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    im1 = axes[1].imshow(
        true_actions.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    axes[1].set_title("Ground-truth action (dim x frame)")
    axes[1].set_ylabel("Action dim")
    fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.01)

    im2 = axes[2].imshow(
        pred_actions.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    axes[2].set_title("Predicted action (first step of chunk)")
    axes[2].set_ylabel("Action dim")
    fig.colorbar(im2, ax=axes[2], fraction=0.025, pad=0.01)

    im3 = axes[3].imshow(
        abs_err.T, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=max(emax, 1e-6)
    )
    axes[3].set_title("Absolute error |pred-gt| (dim x frame)")
    axes[3].set_xlabel("Frame index")
    axes[3].set_ylabel("Action dim")
    fig.colorbar(im3, ax=axes[3], fraction=0.025, pad=0.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "episode_id": episode_id,
        "num_frames": int(true_actions.shape[0]),
        "dim": int(true_actions.shape[1]),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "per_dim_mae": [float(x) for x in per_dim_mae.tolist()],
        "per_dim_rmse": [float(x) for x in per_dim_rmse.tolist()],
        "figure": out_path.name,
    }


def _plot_episode_summary(
    per_episode: list[dict[str, Any]],
    out_path: pathlib.Path,
) -> None:
    if not per_episode:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    ids = [str(x["episode_id"]) for x in per_episode]
    maes = [x["mae"] for x in per_episode]
    rmses = [x["rmse"] for x in per_episode]
    x = np.arange(len(ids))
    width = 0.35
    axes[0].bar(x - width / 2, maes, width=width, label="MAE", color="#577590")
    axes[0].bar(x + width / 2, rmses, width=width, label="RMSE", color="#f3722c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ids)
    axes[0].set_xlabel("Episode id")
    axes[0].set_title("Episode-level overall error")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(loc="upper right")

    dim = len(per_episode[0]["per_dim_mae"])
    dim_x = np.arange(dim)
    for item in per_episode:
        axes[1].plot(
            dim_x,
            item["per_dim_mae"],
            marker="o",
            linewidth=1.5,
            label=f'epi {item["episode_id"]}',
        )
    axes[1].set_xticks(dim_x)
    axes[1].set_xlabel("Action dim")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Per-dim MAE by episode")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"isaaclab_episode_trend_{timestamp}"
    output_dir = pathlib.Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = pathlib.Path(args.dataset_home) / args.repo_id
    data_dir = dataset_root / "data" / "chunk-000"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset data dir not found: {data_dir}")

    available_episode_ids = _discover_available_episode_ids(data_dir)
    if not available_episode_ids:
        raise RuntimeError(f"No episode parquet found under: {data_dir}")

    selected_episode_ids = _select_episode_ids(args, available_episode_ids)
    prompt = _discover_prompt(dataset_root, args.prompt)

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

    episode_results: list[dict[str, Any]] = []
    pbar = tqdm(selected_episode_ids, desc="Episode trend eval", dynamic_ncols=True)
    for episode_id in pbar:
        arrays = _load_episode_arrays(dataset_root, episode_id, max_frames=args.max_frames)
        true_actions = arrays["actions"][:, : args.action_dim]
        pred_actions = _predict_episode_first_step_actions(
            model=model,
            states=arrays["states"],
            table_frames=arrays["table_frames"],
            wrist_frames=arrays["wrist_frames"],
            prompt=prompt,
            infer_batch_size=args.infer_batch_size,
            action_dim=args.action_dim,
        )
        out_path = output_dir / f"episode_{episode_id:06d}_trend.png"
        episode_metrics = _plot_episode_overview(
            episode_id=episode_id,
            true_actions=true_actions,
            pred_actions=pred_actions,
            out_path=out_path,
            smooth_window=args.smooth_window,
        )
        episode_results.append(episode_metrics)
        pbar.set_postfix(
            {
                "episode": episode_id,
                "mae": f'{episode_metrics["mae"]:.4f}',
                "rmse": f'{episode_metrics["rmse"]:.4f}',
            }
        )
    pbar.close()

    _plot_episode_summary(
        episode_results,
        out_path=output_dir / "episodes_summary.png",
    )

    summary = {
        "run_name": run_name,
        "timestamp": timestamp,
        "model_path": args.model_path,
        "dataset_home": args.dataset_home,
        "repo_id": args.repo_id,
        "config_name": args.config_name,
        "num_episodes_requested": args.num_episodes,
        "selected_episode_ids": selected_episode_ids,
        "sampling_mode": "manual" if args.episode_ids else "random",
        "random_seed": args.random_seed if not args.episode_ids else None,
        "infer_batch_size": args.infer_batch_size,
        "max_frames": args.max_frames,
        "prompt": prompt,
        "overall_mean_mae": float(np.mean([x["mae"] for x in episode_results])),
        "overall_mean_rmse": float(np.mean([x["rmse"] for x in episode_results])),
        "episodes": episode_results,
        "summary_figure": "episodes_summary.png",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[done] output_dir: {output_dir}")
    print(f"[done] selected_episode_ids: {selected_episode_ids}")
    print(f"[done] overall_mean_mae: {summary['overall_mean_mae']:.6f}")
    print(f"[done] overall_mean_rmse: {summary['overall_mean_rmse']:.6f}")
    print(f"[done] summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
