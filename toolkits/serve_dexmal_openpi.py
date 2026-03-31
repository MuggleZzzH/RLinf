#!/usr/bin/env python3
"""Serve Dexmal OpenPI checkpoints over websocket for real-robot deployment.

This script is intentionally thin:
- It reuses RLinf's Dexmal OpenPI config registry.
- It supports both RLinf FSDP checkpoints and exported safetensors checkpoints.
- It reuses OpenPI's websocket protocol so the robot can use openpi-client directly.
- It also supports lightweight mock policies for end-to-end deployment testing before
  a checkpoint is ready.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import openpi.policies.policy as _policy
from openpi.models import model as _model
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training.config import AssetsConfig
import openpi.transforms as _transforms
import safetensors
import torch

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)


class _WrappedTorchPolicyModel(torch.nn.Module):
    """Adapts RLinf's model.sample_actions output to OpenPI Policy expectations."""

    def __init__(self, model: OpenPi0ForRLActionPrediction):
        super().__init__()
        self.model = model

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def eval(self):
        self.model.eval()
        return self

    def sample_actions(self, rng_or_device, observation: _model.Observation, **kwargs):
        del rng_or_device
        result = self.model.sample_actions(observation, **kwargs)
        if isinstance(result, dict):
            return result["actions"]
        return result


class _MockPolicy:
    def __init__(self, mode: str, action_horizon: int, action_dim: int):
        self._mode = mode
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._metadata = {
            "policy_type": "mock",
            "mock_mode": mode,
            "action_horizon": action_horizon,
            "action_dim": action_dim,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict, *, noise=None) -> dict:
        del noise
        state = obs.get("state")
        if state is None:
            raise ValueError('Mock policy requires "state" in observation.')
        state = torch.as_tensor(state).detach().cpu().numpy().astype("float32")
        if state.shape[-1] != self._action_dim:
            raise ValueError(
                f"Mock policy expected state dim {self._action_dim}, got {state.shape[-1]}"
            )

        if self._mode == "hold-state":
            action = state
        else:
            action = torch.zeros_like(torch.as_tensor(state)).detach().cpu().numpy().astype("float32")

        action_chunk = action[None, :].repeat(self._action_horizon, axis=0)
        return {
            "actions": action_chunk,
            "policy_timing": {"infer_ms": 0.0},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Dexmal OpenPI policy over websocket.")
    parser.add_argument(
        "--config-name",
        default="pi0_dexmal_aloha",
        help="RLinf OpenPI config name.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint/model directory. Supports RLinf FSDP checkpoints and safetensors checkpoints.",
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Directory containing <asset-id>/norm_stats.json. Defaults to checkpoint-dir.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Dataset repo id or local dataset root. Used to set the data config and default asset id.",
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help="Asset directory name that contains norm_stats.json. Defaults to the repo-id basename.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the websocket server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the websocket server to.",
    )
    parser.add_argument(
        "--default-prompt",
        default=None,
        help="Fallback prompt if the client does not provide one.",
    )
    parser.add_argument(
        "--pytorch-device",
        default=None,
        help='PyTorch device, e.g. "cuda", "cuda:0", or "cpu". Defaults to auto-detect.',
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON string or JSON file path to expose via websocket server metadata.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Wrap the policy with OpenPI's PolicyRecorder.",
    )
    parser.add_argument(
        "--use-repack-inputs",
        action="store_true",
        help="Apply dataset repack transforms on server input. Leave disabled when the client already sends canonical OpenPI inputs.",
    )
    parser.add_argument(
        "--mock-policy",
        choices=("zeros", "hold-state"),
        default=None,
        help="Serve a mock policy instead of loading a checkpoint. Useful before training finishes.",
    )
    parser.add_argument(
        "--mock-action-horizon",
        type=int,
        default=50,
        help="Action horizon used by mock policies.",
    )
    parser.add_argument(
        "--mock-action-dim",
        type=int,
        default=14,
        help="Action dimension used by mock policies.",
    )
    return parser.parse_args()


def _maybe_load_metadata(metadata_json: str | None) -> dict[str, Any]:
    if metadata_json is None:
        return {}
    candidate = Path(metadata_json).expanduser()
    if candidate.exists():
        return json.loads(candidate.read_text())
    return json.loads(metadata_json)


def _resolve_asset_id(repo_id: str | None, asset_id: str | None) -> str | None:
    if asset_id is not None:
        return asset_id
    if repo_id is None:
        return None
    return Path(str(repo_id)).name


def _create_train_config(args: argparse.Namespace):
    if args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required unless --mock-policy is used.")

    asset_id = _resolve_asset_id(args.repo_id, args.asset_id)
    assets_dir = args.assets_dir or args.checkpoint_dir
    data_kwargs: dict[str, Any] = {
        "assets": AssetsConfig(assets_dir=assets_dir, asset_id=asset_id),
    }
    if args.repo_id is not None:
        data_kwargs["repo_id"] = args.repo_id

    return get_openpi_config(
        args.config_name,
        model_path=args.checkpoint_dir,
        data_kwargs=data_kwargs,
    )


def _load_checkpoint_model(
    train_config,
    checkpoint_dir: str,
) -> OpenPi0ForRLActionPrediction:
    checkpoint_dir = str(Path(checkpoint_dir).expanduser().resolve())
    actor_model_config = OpenPi0Config(**train_config.model.__dict__)
    model = OpenPi0ForRLActionPrediction(actor_model_config)

    full_weights_path = os.path.join(checkpoint_dir, "model_state_dict", "full_weights.pt")
    actor_full_weights_path = os.path.join(checkpoint_dir, "actor", "model_state_dict", "full_weights.pt")
    safetensor_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not safetensor_paths:
        default_safetensor = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(default_safetensor):
            safetensor_paths = [default_safetensor]

    logging.info("Loading checkpoint from %s", checkpoint_dir)
    if os.path.exists(full_weights_path):
        state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    elif safetensor_paths:
        for weight_path in safetensor_paths:
            safetensors.torch.load_model(model, weight_path, strict=False)
    else:
        raise FileNotFoundError(
            f"No supported checkpoint weights found under {checkpoint_dir}. "
            "Expected model_state_dict/full_weights.pt, actor/model_state_dict/full_weights.pt, or *.safetensors."
        )

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return model


def _create_checkpoint_policy(args: argparse.Namespace):
    train_config = _create_train_config(args)
    model = _load_checkpoint_model(train_config, args.checkpoint_dir)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    assets_root = Path(args.assets_dir or args.checkpoint_dir).expanduser().resolve()
    norm_stats = _checkpoints.load_norm_stats(assets_root, data_config.asset_id)

    wrapped_model = _WrappedTorchPolicyModel(model)
    repack_transforms = data_config.repack_transforms if args.use_repack_inputs else _transforms.Group()
    metadata = _maybe_load_metadata(args.metadata_json)

    if args.repo_id is not None:
        metadata.setdefault("repo_id", args.repo_id)
    metadata.setdefault("config_name", args.config_name)
    metadata.setdefault("asset_id", data_config.asset_id)
    metadata.setdefault("checkpoint_dir", str(Path(args.checkpoint_dir).expanduser().resolve()))

    policy = _policy.Policy(
        wrapped_model,
        transforms=[
            *repack_transforms.inputs,
            _transforms.InjectDefaultPrompt(args.default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs={
            "mode": "eval",
            "compute_values": False,
        },
        metadata=metadata,
        is_pytorch=True,
        pytorch_device=args.pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return policy


def main() -> None:
    args = parse_args()

    if args.mock_policy is not None:
        policy = _MockPolicy(
            mode=args.mock_policy,
            action_horizon=args.mock_action_horizon,
            action_dim=args.mock_action_dim,
        )
    else:
        policy = _create_checkpoint_policy(args)

    metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logging.info("Serving Dexmal OpenPI policy on ws://%s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
