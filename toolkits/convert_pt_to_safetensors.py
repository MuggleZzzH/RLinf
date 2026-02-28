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

"""Convert a PyTorch checkpoint file (.pt/.pth) to safetensors.

Examples:
    python toolkits/convert_pt_to_safetensors.py \
        --input-pt /path/to/full_weights.pt \
        --output-dir /path/to/output

    python toolkits/convert_pt_to_safetensors.py \
        --input-pt /path/to/full_weights.pt \
        --output-dir /path/to/output \
        --single-file \
        --output-name model \
        --strip-prefix module.
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping
from typing import Any

import torch
from safetensors.torch import save_file

from rlinf.utils.ckpt_convertor.fsdp_convertor.utils import (
    save_state_dict_sharded_safetensors,
)


def _parse_size_to_bytes(size_str: str) -> int:
    """Parse size string like '4GB' or '4096MB' to bytes."""
    pattern = r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$"
    match = re.fullmatch(pattern, size_str)
    if match is None:
        raise ValueError(
            f"Invalid size string: {size_str!r}. Examples: 4GB, 4096MB, 1048576"
        )

    value = float(match.group(1))
    unit = match.group(2).strip().upper()

    multipliers = {
        "": 1,
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "KIB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "MIB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "GIB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
        "TIB": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported size unit: {unit!r}")

    return int(value * multipliers[unit])


def _is_state_dict_like(obj: Any) -> bool:
    """Return True when object looks like a torch state dict."""
    if not isinstance(obj, Mapping):
        return False
    if not obj:
        return False
    if not all(isinstance(k, str) for k in obj.keys()):
        return False
    tensor_count = sum(torch.is_tensor(v) for v in obj.values())
    if tensor_count == 0:
        return False
    return tensor_count == len(obj)


def _get_nested_value(root: Mapping[str, Any], path: str) -> Any:
    """Get nested value from mapping by dot-separated key path."""
    cur: Any = root
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            raise KeyError(f"Path {path!r} not found in checkpoint.")
        cur = cur[key]
    return cur


def _extract_state_dict(checkpoint: Any, state_dict_key: str | None) -> Mapping[str, Any]:
    """Extract model state_dict from various checkpoint layouts."""
    if state_dict_key:
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"--state-dict-key is set, but checkpoint is {type(checkpoint)}."
            )
        state = _get_nested_value(checkpoint, state_dict_key)
        if not _is_state_dict_like(state):
            raise ValueError(
                f"Value at --state-dict-key {state_dict_key!r} is not a mapping."
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
        "Cannot infer state_dict from checkpoint. "
        f"Top-level keys: [{top_keys}]. "
        "Please specify --state-dict-key."
    )


def _normalize_state_dict(
    state_dict: Mapping[str, Any],
    strip_prefixes: list[str],
    target_dtype: torch.dtype | None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Normalize keys/tensors and keep tensor entries only."""
    output: dict[str, torch.Tensor] = {}
    skipped = 0

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            skipped += 1
            continue

        new_key = key
        for prefix in strip_prefixes:
            if prefix and new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]

        tensor = value.detach()
        if target_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=target_dtype)
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        output[new_key] = tensor

    if not output:
        raise ValueError("No tensor entries found in extracted state_dict.")
    return output, skipped


def _resolve_dtype(dtype_name: str) -> torch.dtype | None:
    """Map CLI dtype string to torch dtype."""
    mapping = {
        "auto": None,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return mapping[dtype_name]


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Convert .pt/.pth checkpoint to safetensors."
    )
    parser.add_argument(
        "--input-pt",
        type=str,
        required=True,
        help="Path to input checkpoint (.pt/.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write safetensors files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="model",
        help="Output base name (default: model).",
    )
    parser.add_argument(
        "--state-dict-key",
        type=str,
        default=None,
        help="Optional dot path to state dict inside checkpoint.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="Strip prefix from every key. Can be specified multiple times.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="Optional cast for floating tensors before save.",
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Save one file <output-name>.safetensors instead of sharded output.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="4GB",
        help="Max shard size for sharded save (e.g. 4GB, 4096MB).",
    )
    parser.add_argument(
        "--map-location",
        type=str,
        default="cpu",
        help="torch.load map_location (default: cpu).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] Loading checkpoint: {args.input_pt}")
    checkpoint = torch.load(
        args.input_pt,
        map_location=args.map_location,
        weights_only=False,
    )

    print("[2/4] Extracting state_dict")
    state_dict = _extract_state_dict(checkpoint, args.state_dict_key)

    target_dtype = _resolve_dtype(args.dtype)
    normalized_state_dict, skipped_non_tensor = _normalize_state_dict(
        state_dict=state_dict,
        strip_prefixes=args.strip_prefix,
        target_dtype=target_dtype,
    )
    print(
        "[3/4] Prepared tensors: "
        f"{len(normalized_state_dict)} keys, "
        f"skipped non-tensor entries: {skipped_non_tensor}"
    )

    if args.single_file:
        output_file = args.output_name
        if not output_file.endswith(".safetensors"):
            output_file = f"{output_file}.safetensors"
        output_path = os.path.join(args.output_dir, output_file)
        save_file(normalized_state_dict, output_path, metadata={"format": "pt"})
        print(f"[4/4] Saved single safetensors: {output_path}")
    else:
        base_name = args.output_name
        if base_name.endswith(".safetensors"):
            base_name = base_name[: -len(".safetensors")]
        max_shard_size = _parse_size_to_bytes(args.max_shard_size)
        shard_count, total_size = save_state_dict_sharded_safetensors(
            state_dict=normalized_state_dict,
            out_dir=args.output_dir,
            base_name=base_name,
            max_shard_size=max_shard_size,
        )
        print(
            "[4/4] Saved sharded safetensors: "
            f"{shard_count} shard(s), total_size={total_size} bytes, "
            f"index={os.path.join(args.output_dir, f'{base_name}.safetensors.index.json')}"
        )


if __name__ == "__main__":
    main()
