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

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.common.wrappers import (
    DualQuat2EulerWrapper,
    DualRelativeFrame,
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
    KeyboardRunningModeWrapper,
    MasterTakeoverIntervention,
    apply_single_arm_wrappers,
)
from rlinf.envs.realworld.xsquare.tasks.button_env import (
    ButtonEnv as ButtonEnv,
)
from rlinf.envs.realworld.xsquare.turtle2_env import (
    Turtle2Env,
    Turtle2RobotConfig,
)


def _resolve_deploy_action_mode(
    override_cfg: dict[str, Any],
    env_cfg: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Resolve deploy action mode from env-level and override-level config."""
    wrapper_cfg = dict(env_cfg)
    env_action_mode = wrapper_cfg.get("action_mode", None)
    override_action_mode = override_cfg.get("action_mode", None)

    if env_action_mode is not None and override_action_mode is not None:
        if env_action_mode != override_action_mode:
            raise ValueError(
                "Turtle2 deploy action_mode is configured in both env config "
                "and override_cfg with different values: "
                f"{env_action_mode!r} != {override_action_mode!r}."
            )
        action_mode = env_action_mode
    elif env_action_mode is not None:
        action_mode = env_action_mode
    elif override_action_mode is not None:
        action_mode = override_action_mode
    else:
        action_mode = "relative_pose"

    if action_mode not in {"relative_pose", "absolute_pose"}:
        raise ValueError(
            f"Unsupported Turtle2 deploy action_mode={action_mode!r}. "
            "Expected one of {'relative_pose', 'absolute_pose'}."
        )

    override_cfg["action_mode"] = action_mode
    wrapper_cfg["action_mode"] = action_mode
    return action_mode, wrapper_cfg


def _apply_keyboard_reward(env: gym.Env, mode: str | None) -> gym.Env:
    config = env.get_wrapper_attr("config")
    if config.is_dummy or not mode:
        return env
    if mode == "multi_stage":
        return KeyboardRewardDoneMultiStageWrapper(env)
    if mode == "single_stage":
        return KeyboardRewardDoneWrapper(env)
    return env


def create_button_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = ButtonEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return apply_single_arm_wrappers(env, env_cfg)


def create_turtle2_deploy_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    override_cfg = dict(override_cfg)
    action_mode, wrapper_cfg = _resolve_deploy_action_mode(override_cfg, env_cfg)
    wrapper_cfg.setdefault("no_gripper", False)
    if wrapper_cfg.get("use_spacemouse", False) or wrapper_cfg.get("use_gello", False):
        raise ValueError(
            "Turtle2 deploy does not support teleop wrappers. "
            "Set use_spacemouse=False and use_gello=False."
        )
    use_master_takeover = bool(wrapper_cfg.get("use_master_takeover", False))
    if use_master_takeover and action_mode != "absolute_pose":
        raise ValueError(
            "use_master_takeover=True requires action_mode='absolute_pose'."
        )
    if use_master_takeover and str(
        override_cfg.get("pose_control_backend", "smooth")
    ).lower() != "hybrid":
        raise ValueError(
            "use_master_takeover=True requires "
            "override_cfg.pose_control_backend='hybrid'."
        )
    override_cfg.setdefault("use_arm_ids", [0, 1])
    override_cfg.setdefault("use_camera_ids", [0, 1, 2])
    override_cfg.setdefault("enforce_gripper_close", False)
    override_cfg.setdefault("enable_task_reward", False)
    override_cfg.setdefault("task_description", env_cfg.get("task_description", ""))
    override_cfg["action_mode"] = action_mode
    config = Turtle2RobotConfig(**override_cfg)
    env = Turtle2Env(
        config=config,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    if action_mode == "relative_pose" and wrapper_cfg.get("use_relative_frame", True):
        env = DualRelativeFrame(env)
    if use_master_takeover:
        env = MasterTakeoverIntervention(
            env, config=wrapper_cfg.get("master_takeover", None)
        )
        keyboard_running_mode_cfg = wrapper_cfg.get("keyboard_running_mode", None)
        if keyboard_running_mode_cfg is not None:
            env = KeyboardRunningModeWrapper(env, config=keyboard_running_mode_cfg)
    env = _apply_keyboard_reward(env, wrapper_cfg.get("keyboard_reward_wrapper", None))
    env = DualQuat2EulerWrapper(env)
    return env


register(
    id="ButtonEnv-v1",
    entry_point="rlinf.envs.realworld.xsquare.tasks:create_button_env",
)

register(
    id="Turtle2DeployEnv-v1",
    entry_point="rlinf.envs.realworld.xsquare.tasks:create_turtle2_deploy_env",
)
