#!/usr/bin/env python3
"""Compute OpenPI norm_stats.json using RLinf's OpenPI config registry."""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.data_loader as data_loader
import openpi.transforms as transforms
from openpi.training.config import AssetsConfig
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute norm_stats.json for an RLinf OpenPI config."
    )
    parser.add_argument(
        "--config-name",
        required=True,
        help="RLinf OpenPI config name, e.g. pi0_dexmal_aloha.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Checkpoint/model directory used as the OpenPI assets root.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repo id or local dataset root to read from.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Alias of repo-id for local datasets. Used when repo-id is omitted.",
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help="Directory name under model-path where norm_stats.json will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used while scanning the dataset.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames used to estimate stats.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers for torch dataset loading.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to <model-path>/<asset-id>.",
    )
    parser.add_argument(
        "--force-torch-loader",
        action="store_true",
        help="Disable the Dexmal parquet fast path and use the generic OpenPI data loader.",
    )
    return parser.parse_args()


def create_torch_dataloader(data_config, action_horizon, batch_size, model_config, num_workers, max_frames=None):
    dataset = data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    torch_loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return torch_loader, num_batches


def _resolve_local_dataset_root(repo_id: str | Path) -> Path | None:
    root = Path(str(repo_id)).expanduser()
    if root.exists():
        return root.resolve()
    return None


def _arrow_array_to_numpy(array) -> np.ndarray:
    import pyarrow as pa

    array = array.combine_chunks()
    if pa.types.is_floating(array.type) or pa.types.is_integer(array.type) or pa.types.is_boolean(array.type):
        return array.to_numpy(zero_copy_only=False)
    if pa.types.is_fixed_size_list(array.type):
        values = array.values.to_numpy(zero_copy_only=False)
        return values.reshape(len(array), array.type.list_size)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        return np.asarray(array.to_pylist())
    raise TypeError(f"Unsupported arrow type for fast stats path: {array.type}")


def _iter_dexmal_episode_blocks(dataset_root: Path):
    import pyarrow.parquet as pq

    for parquet_path in sorted((dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=["episode_index", "observation.state", "action"],
        )
        episode_index = _arrow_array_to_numpy(table["episode_index"]).astype(np.int64)
        states = _arrow_array_to_numpy(table["observation.state"]).astype(np.float32)
        actions = _arrow_array_to_numpy(table["action"]).astype(np.float32)

        if len(episode_index) == 0:
            continue

        change_points = np.flatnonzero(np.diff(episode_index)) + 1
        starts = np.concatenate(([0], change_points))
        ends = np.concatenate((change_points, [len(episode_index)]))

        for start, end in zip(starts, ends, strict=True):
            yield int(episode_index[start]), states[start:end], actions[start:end], parquet_path


def _compute_dexmal_fast_stats(
    dataset_root: Path,
    action_horizon: int,
    max_frames: int | None,
) -> dict[str, normalize.NormStats]:
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    delta_mask = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)
    processed_frames = 0

    episodes = list(_iter_dexmal_episode_blocks(dataset_root))
    progress_total = sum(len(states) for _, states, _, _ in episodes)
    if max_frames is not None:
        progress_total = min(progress_total, max_frames)

    with tqdm.tqdm(total=progress_total, desc="Computing stats (fast path)") as pbar:
        for episode_index, states, actions, parquet_path in episodes:
            del episode_index, parquet_path
            if max_frames is not None and processed_frames >= max_frames:
                break

            if max_frames is not None:
                remaining = max_frames - processed_frames
                if remaining <= 0:
                    break
                states = states[:remaining]
                actions = actions[:remaining]

            num_frames = len(states)
            if num_frames == 0:
                continue

            stats["state"].update(states)

            indices = np.arange(num_frames)[:, None] + np.arange(action_horizon)[None, :]
            indices = np.clip(indices, 0, num_frames - 1)
            action_chunks = actions[indices].copy()
            action_chunks[..., delta_mask] -= states[:, None, delta_mask]
            stats["actions"].update(action_chunks)

            processed_frames += num_frames
            pbar.update(num_frames)

    return {key: stat.get_statistics() for key, stat in stats.items()}


def _should_use_dexmal_fast_path(
    args: argparse.Namespace,
    repo_id: str,
    config_name: str,
) -> tuple[bool, Path | None]:
    if args.force_torch_loader:
        return False, None
    if config_name != "pi0_dexmal_aloha":
        return False, None
    dataset_root = _resolve_local_dataset_root(repo_id)
    if dataset_root is None:
        return False, None
    if not (dataset_root / "data").exists():
        return False, None
    return True, dataset_root


def main() -> None:
    args = parse_args()

    repo_id = args.repo_id or args.data_path
    if repo_id is None:
        raise ValueError("Either --repo-id or --data-path must be provided.")

    asset_id = args.asset_id or Path(str(repo_id)).name
    data_kwargs = {
        "repo_id": repo_id,
        "assets": AssetsConfig(
            assets_dir=args.model_path,
            asset_id=asset_id,
        ),
    }
    config = get_openpi_config(
        args.config_name,
        model_path=args.model_path,
        data_kwargs=data_kwargs,
        batch_size=args.batch_size,
    )
    config = dataclasses.replace(config, num_workers=args.num_workers)
    data_config = config.data.create(config.assets_dirs, config.model)
    use_fast_path, dataset_root = _should_use_dexmal_fast_path(args, repo_id, args.config_name)

    if use_fast_path:
        print(f"[fast-path] Using parquet-only stats path for {args.config_name}")
        print(f"[fast-path] dataset_root={dataset_root}")
        norm_stats = _compute_dexmal_fast_stats(
            dataset_root,
            config.model.action_horizon,
            args.max_frames,
        )
    else:
        torch_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            args.max_frames,
        )

        stats = {key: normalize.RunningStats() for key in ("state", "actions")}
        for batch in tqdm.tqdm(torch_loader, total=num_batches, desc="Computing stats"):
            for key in ("state", "actions"):
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stat.get_statistics() for key, stat in stats.items()}

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to determine the output directory.")
        output_dir = Path(args.model_path) / data_config.asset_id

    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(output_dir, norm_stats)
    print(f"Writing stats to: {output_dir}")
    print(f"repo_id={data_config.repo_id}")
    print(f"asset_id={data_config.asset_id}")


if __name__ == "__main__":
    main()
