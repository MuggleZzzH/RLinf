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

import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.xsquare.x1_env import X1Env, X1RobotConfig


@dataclass
class X1DeployEnvConfig(X1RobotConfig):
    use_camera_ids: list[int] = field(default_factory=lambda: [0, 1, 2])
    use_arm_ids: list[int] = field(default_factory=lambda: [0, 1])
    enforce_gripper_close: bool = False
    task_description: str = ""

    def __post_init__(self):
        self.use_arm_ids = list(self.use_arm_ids)
        if self.use_arm_ids != [0, 1]:
            raise ValueError(
                "X1DeployEnv currently supports only dual-arm s2s deployment with "
                "use_arm_ids=[0, 1]."
            )
        self.target_ee_pose = np.asarray(self.target_ee_pose, dtype=np.float64).reshape(
            2, 6
        )
        self.reset_ee_pose = np.asarray(self.reset_ee_pose, dtype=np.float64).reshape(
            2, 6
        )
        self.ee_pose_limit_min = np.asarray(
            self.ee_pose_limit_min, dtype=np.float64
        ).reshape(2, 6)
        self.ee_pose_limit_max = np.asarray(
            self.ee_pose_limit_max, dtype=np.float64
        ).reshape(2, 6)
        self.reward_threshold = np.asarray(
            self.reward_threshold, dtype=np.float64
        ).reshape(2, 6)
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)


class X1DeployEnv(X1Env):
    CONFIG_CLS = X1DeployEnvConfig

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = self.CONFIG_CLS(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return self.config.task_description

    def _init_action_obs_spaces(self):
        super()._init_action_obs_spaces()
        self._relative_pose_action_space = self.action_space
        action_low = []
        action_high = []
        for arm_id in self.config.use_arm_ids:
            action_low.append(
                np.concatenate(
                    [
                        self.config.ee_pose_limit_min[arm_id],
                        np.array([self.config.gripper_width_limit_min]),
                    ]
                )
            )
            action_high.append(
                np.concatenate(
                    [
                        self.config.ee_pose_limit_max[arm_id],
                        np.array([self.config.gripper_width_limit_max]),
                    ]
                )
            )
        self._absolute_pose_action_space = gym.spaces.Box(
            low=np.concatenate(action_low).astype(np.float32),
            high=np.concatenate(action_high).astype(np.float32),
            dtype=np.float32,
        )

    def get_absolute_pose_action_space(self) -> gym.spaces.Box:
        return self._absolute_pose_action_space

    def get_relative_pose_action_space(self) -> gym.spaces.Box:
        return self._relative_pose_action_space

    def step_absolute_pose(self, action: np.ndarray):
        assert action.shape == (len(self.config.use_arm_ids) * 7,), (
            f"Action shape must be {(len(self.config.use_arm_ids) * 7,)}, but got {action.shape}."
        )

        start_time = time.time()
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()

        action = np.clip(
            raw_action,
            self._absolute_pose_action_space.low,
            self._absolute_pose_action_space.high,
        ).astype(np.float32, copy=False)
        action = action.reshape(-1, 7)
        next_positions = {
            0: self._x1_state.follow1_pos.copy(),
            1: self._x1_state.follow2_pos.copy(),
        }
        for action_row, arm_id in zip(action, self.config.use_arm_ids, strict=False):
            next_positions[arm_id][:6] = action_row[:6]
            if self.config.enforce_gripper_close:
                next_positions[arm_id][6] = self.config.gripper_width_limit_min
            else:
                next_positions[arm_id][6] = action_row[6]

        next_position = self._clip_position_to_safety_box(
            np.stack([next_positions[0], next_positions[1]])
        )
        next_position = next_position.astype(np.float32, copy=False)
        executed_action = next_position.reshape(-1).copy()
        clip_delta = np.abs(executed_action - raw_action)
        clip_delta_max = float(np.max(clip_delta)) if clip_delta.size else 0.0
        action_clipped = bool(clip_delta_max > 1e-6)
        next_position1 = next_position[0]
        next_position2 = next_position[1]
        if self.config.debug_gripper_control or self.config.debug_pose_control:
            now = time.time()
            last_log_time = getattr(self, "_last_pose_debug_log_time", 0.0)
            if now - last_log_time >= 1.0:
                self._last_pose_debug_log_time = now
                self._logger.info(
                    "X1 absolute pose target: raw_left=%s raw_right=%s "
                    "executed_left=%s executed_right=%s current_left=%s "
                    "current_right=%s clipped=%s clip_delta_max=%.6f",
                    np.array2string(raw_action[:7], precision=4),
                    np.array2string(raw_action[7:], precision=4),
                    np.array2string(executed_action[:7], precision=4),
                    np.array2string(executed_action[7:], precision=4),
                    np.array2string(self._x1_state.follow1_pos, precision=4),
                    np.array2string(self._x1_state.follow2_pos, precision=4),
                    action_clipped,
                    clip_delta_max,
                )
                if self.config.debug_gripper_control:
                    self._logger.info(
                        "X1 absolute pose gripper target: action=(%.4f, %.4f) "
                        "target=(%.4f, %.4f) current=(%.4f, %.4f)",
                        float(action[0, 6]),
                        float(action[-1, 6]),
                        float(next_position1[6]),
                        float(next_position2[6]),
                        float(self._x1_state.follow1_pos[6]),
                        float(self._x1_state.follow2_pos[6]),
                    )

        if not self.config.is_dummy:
            self._controller.move_arm(
                next_position1.tolist(), next_position2.tolist()
            ).wait()
        else:
            self._x1_state.follow1_pos = next_position1.copy()
            self._x1_state.follow2_pos = next_position2.copy()

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._x1_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()
        reward = self._calc_step_reward(observation)
        terminated = False
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "raw_action": raw_action,
            "executed_action": executed_action,
            "action_clipped": action_clipped,
            "clip_delta_max": clip_delta_max,
        }
        return observation, reward, terminated, truncated, info

    def step_relative_pose(self, action: np.ndarray):
        return super().step(action)

    def step(self, action: np.ndarray):
        return self.step_relative_pose(action)

    def _calc_step_reward(self, observation) -> float:
        return 0.0
