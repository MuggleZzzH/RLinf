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
    apply_single_arm_wrappers,
)
from rlinf.envs.realworld.xsquare.tasks.button_env import (
    ButtonEnv as ButtonEnv,
)
from rlinf.envs.realworld.xsquare.turtle2_env import (
    Turtle2Env,
    Turtle2RobotConfig,
)


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
    """Build ``Turtle2DeployEnv-v1`` with the deploy wrapper chain.

    Validation is intentionally minimal:

    * ``override_cfg.action_mode`` must be ``relative_pose`` or
      ``absolute_pose`` (a top-level ``env_cfg.action_mode`` is *not* read —
      configs should set the value under ``override_cfg``).
    * ``override_cfg.use_arm_ids`` must be ``[0, 1]`` (deploy is dual-arm only;
      single-arm deploy is out of scope for this entry).

    Wrapper chain:

    * ``relative_pose``: wraps with the shared :class:`DualRelativeFrame`
      (EE→base action transform, reset-relative obs) and
      :class:`DualQuat2EulerWrapper` (quat→euler on ``tcp_pose``). The
      ``state.gripper`` key is passed through untouched by both wrappers.
    * ``absolute_pose``: only :class:`DualQuat2EulerWrapper` is applied —
      absolute pose targets are already in the base frame, so no
      ``DualRelativeFrame`` step is required. The wrapper is an observation
      wrapper and does not modify actions.
    """
    # Deploy-specific defaults are injected here rather than in
    # ``Turtle2RobotConfig`` so that ButtonEnv (which also constructs the same
    # config class) is unaffected. User overrides win via the trailing
    # ``**override_cfg`` spread.
    cfg_kwargs: dict[str, Any] = {
        "use_arm_ids": [0, 1],
        "use_camera_ids": [0, 1, 2],
        "enforce_gripper_close": False,
        "enable_gripper_penalty": False,
        "use_dense_reward": False,
        "expose_gripper_obs": True,
        "enable_task_reward": False,
        **override_cfg,
    }
    cfg = Turtle2RobotConfig(**cfg_kwargs)
    if cfg.action_mode not in {"relative_pose", "absolute_pose"}:
        raise ValueError(
            f"Unsupported Turtle2 deploy action_mode={cfg.action_mode!r}. "
            "Expected one of {'relative_pose', 'absolute_pose'}."
        )
    if list(cfg.use_arm_ids) != [0, 1]:
        raise ValueError(
            "Turtle2DeployEnv-v1 requires use_arm_ids=[0, 1]; "
            f"got {list(cfg.use_arm_ids)!r}."
        )

    env = Turtle2Env(
        config=cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    if cfg.action_mode == "relative_pose":
        env = DualRelativeFrame(env)
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
