#!/usr/bin/env python3

# Copyright 2025 The RLinf Authors.
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

"""Prepare an eval-ready OpenPI checkpoint directory from an RLinf .pt checkpoint.

This utility converts ``full_weights.pt`` into ``model.safetensors`` and copies any
``norm_stats.json`` files from the checkpoint directory into the output directory.

Example:
    python toolkits/prepare_openpi_eval_dir.py \
        --ckpt-input /path/to/global_step_26000/actor/model_state_dict \
        --output-dir /path/to/global_step_26000/actor/eval_model
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Mapping
from typing import Any

import torch
from safetensors.torch import save_file


def _is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False
    if not all(isinstance(k, str) for k in obj.keys()):
        return False
    tensor_count = sum(torch.is_tensor(v) for v in obj.values())
    return tensor_count > 0 and tensor_count == len(obj)


def _get_nested_value(root: Mapping[str, Any], path: str) -> Any:
    cur: Any = root
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            raise KeyError(f"Path {path!r} not found in checkpoint.")
        cur = cur[key]
    return cur


def _extract_state_dict(checkpoint: Any, state_dict_key: str | None) -> Mapping[str, Any]:
    if state_dict_key:
        value = _get_nested_value(checkpoint, state_dict_key)
        if not _is_state_dict_like(value):
            raise ValueError(
                f"Value at --state-dict-key {state_dict_key!r} is not a tensor state_dict."
            )
        return value

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
        "Cannot infer state_dict from checkpoint. "
        f"Top-level keys: [{top_keys}]. Please specify --state-dict-key."
    )


def _normalize_state_dict(
    state_dict: Mapping[str, Any], strip_prefixes: list[str]
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        new_key = key
        for prefix in strip_prefixes:
            if prefix and new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]

        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        output[new_key] = tensor

    if not output:
        raise ValueError("No tensor entries found in extracted state_dict.")
    return output


def _resolve_ckpt_file(path: str) -> str:
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        for candidate in ("full_weights.pt", "full_weigths.pt"):
            candidate_path = os.path.join(path, candidate)
            if os.path.isfile(candidate_path):
                return candidate_path

        for root, _, files in os.walk(path):
            for candidate in ("full_weights.pt", "full_weigths.pt"):
                if candidate in files:
                    return os.path.join(root, candidate)

    raise FileNotFoundError(f"Could not resolve checkpoint file from: {path}")


def _copy_norm_stats(norm_source_dir: str, output_dir: str) -> list[str]:
    copied: list[str] = []
    for root, _, files in os.walk(norm_source_dir):
        if "norm_stats.json" not in files:
            continue
        src_path = os.path.join(root, "norm_stats.json")
        rel_dir = os.path.relpath(root, norm_source_dir)
        dst_dir = output_dir if rel_dir == "." else os.path.join(output_dir, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, "norm_stats.json")
        shutil.copy2(src_path, dst_path)
        copied.append(dst_path)
    return copied


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an eval-ready OpenPI checkpoint directory."
    )
    parser.add_argument(
        "--ckpt-input",
        type=str,
        required=True,
        help="Path to full_weights.pt or a directory containing it.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write model.safetensors and norm stats into.",
    )
    parser.add_argument(
        "--norm-source-dir",
        type=str,
        default=None,
        help="Directory to search for norm_stats.json. Defaults to ckpt-input dir.",
    )
    parser.add_argument(
        "--state-dict-key",
        type=str,
        default=None,
        help="Optional dot path to state_dict inside the checkpoint.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Strip prefix from every key. Can be specified multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if model.safetensors already exists.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    ckpt_file = _resolve_ckpt_file(args.ckpt_input)
    ckpt_dir = args.ckpt_input if os.path.isdir(args.ckpt_input) else os.path.dirname(ckpt_file)
    norm_source_dir = args.norm_source_dir or ckpt_dir

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "model.safetensors")
    if os.path.exists(output_file) and not args.overwrite:
        raise FileExistsError(
            f"{output_file} already exists. Use --overwrite to replace it."
        )

    print(f"[1/4] Loading checkpoint: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    print("[2/4] Extracting and normalizing state_dict")
    state_dict = _extract_state_dict(checkpoint, args.state_dict_key)
    state_dict = _normalize_state_dict(state_dict, args.strip_prefix)

    print(f"[3/4] Saving safetensors: {output_file}")
    save_file(state_dict, output_file, metadata={"format": "pt"})

    print(f"[4/4] Copying norm stats from: {norm_source_dir}")
    copied_paths = _copy_norm_stats(norm_source_dir, args.output_dir)
    if copied_paths:
        print("Copied norm stats:")
        for path in copied_paths:
            print(f"  - {path}")
    else:
        print("No norm_stats.json found under the norm source directory.")

    print("Done.")


if __name__ == "__main__":
    main()
