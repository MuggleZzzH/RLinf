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

"""Generic Turtle2 deploy environment.

This module defines :class:`Turtle2DeployEnv`, a thin subclass of
:class:`Turtle2Env` that serves the ``Turtle2DeployEnv-v1`` gym id. It is
intended for policy-only rollout / evaluation on Turtle2 hardware.

Local obs/action contract (Turtle2 deploy only — *not* a repo-wide convention):

* ``obs['state']['tcp_pose']``: ``(14,)`` dual-arm ``xyz + quat`` in the
  Turtle2 base frame. Matches the existing :class:`Turtle2Env` format so the
  shared :class:`DualRelativeFrame` / :class:`DualQuat2EulerWrapper` apply
  unchanged.
* ``obs['state']['gripper']``: ``(2,)`` dual-arm gripper width, pass-through
  (no wrapper touches it).
* Action: ``(14,)``, two arms of ``[dx, dy, dz, drx, dry, drz, dg]``
  (``relative_pose`` mode, EE-frame for the 6D motion) or
  ``[x, y, z, rx, ry, rz, g]`` (``absolute_pose`` mode, base-frame).

``relative_pose`` reuses the existing :meth:`Turtle2Env.step` path; the EE→base
transform is performed by an outer :class:`DualRelativeFrame` wrapper applied
in the task factory. ``absolute_pose`` is a deploy-only control path that
issues per-arm absolute targets directly to the smooth controller and is
implemented entirely in :meth:`Turtle2DeployEnv._step_absolute_pose`.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.xsquare.turtle2_env import Turtle2Env, Turtle2RobotConfig
from rlinf.scheduler import Turtle2HWInfo, WorkerInfo

NUM_ARMS = 2
POSE_DIM = 6
ACTION_DIM_PER_ARM = POSE_DIM + 1  # xyz + rpy + gripper


@dataclass
class Turtle2DeployConfig(Turtle2RobotConfig):
    """Config for :class:`Turtle2DeployEnv`.

    Adds the deploy-specific ``action_mode`` selector and ``task_description``
    string. Deploy defaults force dual-arm operation and disable training-only
    behaviour (gripper-close enforcement, dense reward).
    """

    action_mode: Literal["relative_pose", "absolute_pose"] = "relative_pose"
    task_description: str = ""

    # Deploy-only default overrides.
    use_arm_ids: list[int] = field(default_factory=lambda: [0, 1])
    use_camera_ids: list[int] = field(default_factory=lambda: [0, 1, 2])
    enforce_gripper_close: bool = False
    enable_gripper_penalty: bool = False
    use_dense_reward: bool = False

    def __post_init__(self) -> None:
        # YAML-supplied lists need to be promoted to ndarrays because
        # ``Turtle2Env._init_action_obs_spaces`` slices them as numpy arrays
        # (e.g. ``ee_pose_limit_min[0, :3]``). This is a minimal type
        # conversion — no shape or value validation is performed here.
        for name in (
            "target_ee_pose",
            "reset_ee_pose",
            "ee_pose_limit_min",
            "ee_pose_limit_max",
        ):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))


class Turtle2DeployEnv(Turtle2Env):
    """Turtle2 environment for policy-only deployment.

    Subclasses :class:`Turtle2Env` to (1) expose gripper as a separate
    observation key, (2) add an ``absolute_pose`` action mode for policies
    that emit per-arm absolute pose targets, and (3) short-circuit the task
    reward (deploy does not compute task-success rewards).

    The ``relative_pose`` mode delegates to the parent ``step`` and is
    semantically identical to existing Turtle2 training behaviour. The
    ``DualRelativeFrame`` / ``DualQuat2EulerWrapper`` chain is applied
    externally by the task factory.
    """

    def __init__(
        self,
        config: Turtle2DeployConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[Turtle2HWInfo],
        env_idx: int,
    ) -> None:
        super().__init__(
            config=config,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
        )
        # Extend the parent observation space with the deploy-only gripper
        # channel. We update both ``observation_space`` (the public dict the
        # wrappers see) and ``_base_observation_space`` (used to sample dummy
        # observations) so dummy-mode sampling stays consistent.
        gripper_space = gym.spaces.Box(
            low=self.config.gripper_width_limit_min,
            high=self.config.gripper_width_limit_max,
            shape=(len(self.config.use_arm_ids),),
            dtype=np.float64,
        )
        self.observation_space["state"]["gripper"] = gripper_space
        self._base_observation_space = copy.deepcopy(self.observation_space)

        # In absolute_pose mode the action is a per-arm base-frame absolute
        # pose target, so the action space bounds are the ee pose limits
        # plus the gripper width limits rather than the [-1, 1] delta range
        # used by relative_pose.
        if self.config.action_mode == "absolute_pose":
            self.action_space = self._build_absolute_action_space()

    # ------------------------------------------------------------------
    # Action / observation
    # ------------------------------------------------------------------

    def _build_absolute_action_space(self) -> gym.spaces.Box:
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for arm_id in self.config.use_arm_ids:
            lows.append(
                np.concatenate(
                    [
                        self.config.ee_pose_limit_min[arm_id],
                        np.array([self.config.gripper_width_limit_min]),
                    ]
                )
            )
            highs.append(
                np.concatenate(
                    [
                        self.config.ee_pose_limit_max[arm_id],
                        np.array([self.config.gripper_width_limit_max]),
                    ]
                )
            )
        return gym.spaces.Box(
            low=np.concatenate(lows).astype(np.float32),
            high=np.concatenate(highs).astype(np.float32),
            dtype=np.float32,
        )

    def _get_observation(self) -> dict[str, dict[str, np.ndarray]]:
        """Return the parent observation augmented with a ``gripper`` key.

        ``tcp_pose`` is left untouched (still ``(14,) xyz+quat``) so the
        existing ``DualRelativeFrame`` / ``DualQuat2EulerWrapper`` chain
        applies without modification.
        """
        obs = super()._get_observation()
        if not self.config.is_dummy:
            grippers: list[float] = []
            if 0 in self.config.use_arm_ids:
                grippers.append(float(self._turtle2_state.follow1_pos[6]))
            if 1 in self.config.use_arm_ids:
                grippers.append(float(self._turtle2_state.follow2_pos[6]))
            obs["state"]["gripper"] = np.array(grippers, dtype=np.float64)
        else:
            # Parent ``_get_observation`` for dummy mode samples the full
            # observation space, which (after ``__init__``) already includes
            # the gripper key, so nothing extra is needed here.
            pass
        return obs

    # ------------------------------------------------------------------
    # Step dispatch
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        if self.config.action_mode == "relative_pose":
            return super().step(action)
        if self.config.action_mode == "absolute_pose":
            return self._step_absolute_pose(action)
        # Validation in the factory should make this unreachable.
        raise ValueError(
            f"Unsupported action_mode={self.config.action_mode!r}. "
            "Expected one of {'relative_pose', 'absolute_pose'}."
        )

    def _step_absolute_pose(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """Execute one absolute-pose step.

        Args:
            action: ``(14,)`` array, per-arm
                ``[x, y, z, roll, pitch, yaw, gripper]`` in the Turtle2 base
                frame.

        Returns:
            Standard gymnasium ``(obs, reward, terminated, truncated, info)``.
            Reward is always ``0.0`` (deploy env does not compute task
            rewards); ``terminated`` is always ``False``; ``truncated``
            triggers on ``max_num_steps``.
        """
        expected_shape = (len(self.config.use_arm_ids) * ACTION_DIM_PER_ARM,)
        assert action.shape == expected_shape, (
            f"Action shape must be {expected_shape}, but got {action.shape}."
        )

        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.reshape(-1, ACTION_DIM_PER_ARM)

        next_positions = {
            0: self._turtle2_state.follow1_pos.copy(),
            1: self._turtle2_state.follow2_pos.copy(),
        }
        for row, arm_id in zip(action, self.config.use_arm_ids, strict=False):
            next_positions[arm_id][:POSE_DIM] = row[:POSE_DIM]
            next_positions[arm_id][POSE_DIM] = row[POSE_DIM]

        next_position = self._clip_position_to_safety_box(
            np.stack([next_positions[0], next_positions[1]])
        )

        if not self.config.is_dummy:
            self._controller.move_arm(
                next_position[0].tolist(), next_position[1].tolist()
            ).wait()
        else:
            self._turtle2_state.follow1_pos = next_position[0].copy()
            self._turtle2_state.follow2_pos = next_position[1].copy()

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._turtle2_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()
        reward = self._calc_step_reward(observation)
        terminated = False
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Reward / metadata
    # ------------------------------------------------------------------

    def _calc_step_reward(self, observation: dict[str, np.ndarray]) -> float:
        # Deploy never evaluates task-success reward. Keeping this trivial
        # avoids inheriting the parent's reward-threshold / dense-reward
        # branches, which require task-specific configuration that deploy
        # configs are not expected to supply.
        return 0.0

    @property
    def task_description(self) -> str:
        return self.config.task_description
