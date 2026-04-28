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
        self.joint_limit_min = np.asarray(self.joint_limit_min, dtype=np.float64)
        self.joint_limit_max = np.asarray(self.joint_limit_max, dtype=np.float64)
        if self.joint_limit_min.shape == (7,):
            self.joint_limit_min = np.stack([self.joint_limit_min] * 2)
        if self.joint_limit_max.shape == (7,):
            self.joint_limit_max = np.stack([self.joint_limit_max] * 2)
        self.joint_limit_min = self.joint_limit_min.reshape(2, 7)
        self.joint_limit_max = self.joint_limit_max.reshape(2, 7)
        if self.joint_limit_min.shape != (2, 7):
            raise ValueError(
                "joint_limit_min must be shape (2, 7) or (7,), "
                f"got {self.joint_limit_min.shape}."
            )
        if self.joint_limit_max.shape != (2, 7):
            raise ValueError(
                "joint_limit_max must be shape (2, 7) or (7,), "
                f"got {self.joint_limit_max.shape}."
            )
        if np.any(self.joint_limit_min > self.joint_limit_max):
            raise ValueError("joint_limit_min must be <= joint_limit_max.")
        self.max_joint_delta = float(self.max_joint_delta)


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
        joint_low = []
        joint_high = []
        for arm_id in self.config.use_arm_ids:
            joint_low.append(self.config.joint_limit_min[arm_id])
            joint_high.append(self.config.joint_limit_max[arm_id])
        self._joint_action_space = gym.spaces.Box(
            low=np.concatenate(joint_low).astype(np.float32),
            high=np.concatenate(joint_high).astype(np.float32),
            dtype=np.float32,
        )

    def get_absolute_pose_action_space(self) -> gym.spaces.Box:
        return self._absolute_pose_action_space

    def get_joint_action_space(self) -> gym.spaces.Box:
        return self._joint_action_space

    def get_relative_pose_action_space(self) -> gym.spaces.Box:
        return self._relative_pose_action_space

    def _current_absolute_pose_action(self) -> np.ndarray:
        return np.stack(
            [self._x1_state.follow1_pos, self._x1_state.follow2_pos]
        ).astype(np.float32, copy=False).reshape(-1).copy()

    def _last_published_absolute_pose_action(self) -> np.ndarray:
        last_action = getattr(self, "_last_published_action", None)
        if last_action is None:
            return self._current_absolute_pose_action()
        return np.asarray(last_action, dtype=np.float32).reshape(-1).copy()

    def _direct_pose_rejection_reason(
        self, raw_action: np.ndarray, expected_shape: tuple[int, ...]
    ) -> str | None:
        if raw_action.shape != expected_shape:
            return f"invalid_shape:{raw_action.shape}"
        if not np.all(np.isfinite(raw_action)):
            return "non_finite"
        low = self._absolute_pose_action_space.low
        high = self._absolute_pose_action_space.high
        if np.any(raw_action < low) or np.any(raw_action > high):
            return "outside_absolute_pose_action_space"
        if self.config.enforce_gripper_close:
            grip_min = float(self.config.gripper_width_limit_min)
            gripper_values = raw_action.reshape(-1, 7)[:, 6]
            if np.any(np.abs(gripper_values - grip_min) > 1e-6):
                return "enforce_gripper_close"
        return None

    def _limited_direct_pose_action(
        self, raw_action: np.ndarray, last_published_action: np.ndarray
    ) -> np.ndarray:
        target = np.asarray(raw_action, dtype=np.float32).reshape(-1, 7)
        if not self.config.direct_pose_limiter_enabled:
            return target.reshape(-1).copy()

        current = np.asarray(last_published_action, dtype=np.float32).reshape(-1, 7)
        xyz_delta = target[:, :3] - current[:, :3]
        rpy_delta = self._shortest_angle_delta(current[:, 3:6], target[:, 3:6])
        gripper_delta = target[:, 6] - current[:, 6]

        limited = current.copy()
        limited[:, :3] = current[:, :3] + np.clip(
            xyz_delta,
            -self.config.direct_max_xyz_step,
            self.config.direct_max_xyz_step,
        )
        limited[:, 3:6] = self._normalize_angles(
            current[:, 3:6]
            + np.clip(
                rpy_delta,
                -self.config.direct_max_rpy_step,
                self.config.direct_max_rpy_step,
            )
        )
        limited[:, 6] = current[:, 6] + np.clip(
            gripper_delta,
            -self.config.direct_max_gripper_step,
            self.config.direct_max_gripper_step,
        )
        return limited.astype(np.float32, copy=False).reshape(-1).copy()

    def step_absolute_pose(self, action: np.ndarray):
        start_time = time.time()
        expected_shape = (len(self.config.use_arm_ids) * 7,)
        raw_action = np.asarray(action, dtype=np.float32)
        pose_control_backend = str(self.config.pose_control_backend)

        if pose_control_backend == "direct":
            rejection_reason = self._direct_pose_rejection_reason(
                raw_action,
                expected_shape,
            )
            last_published_action = self._last_published_absolute_pose_action()
            if rejection_reason is None:
                executed_action = self._limited_direct_pose_action(
                    raw_action,
                    last_published_action,
                )
                action_rejected = False
                next_position = executed_action.reshape(-1, 7)
                next_position1 = next_position[0]
                next_position2 = next_position[1]
                if not self.config.is_dummy:
                    self._controller.move_arm(
                        next_position1.tolist(), next_position2.tolist()
                    ).wait()
                else:
                    self._x1_state.follow1_pos = next_position1.copy()
                    self._x1_state.follow2_pos = next_position2.copy()
                self._last_published_action = executed_action.copy()
                last_published_action = executed_action.copy()
            else:
                executed_action = last_published_action.copy()
                action_rejected = True

            if self.config.debug_gripper_control or self.config.debug_pose_control:
                now = time.time()
                last_log_time = getattr(self, "_last_pose_debug_log_time", 0.0)
                if now - last_log_time >= 1.0:
                    self._last_pose_debug_log_time = now
                    self._logger.info(
                        "X1 direct pose target: raw=%s executed=%s "
                        "rejected=%s reason=%s current_left=%s current_right=%s",
                        np.array2string(raw_action.reshape(-1), precision=4),
                        np.array2string(executed_action, precision=4),
                        action_rejected,
                        rejection_reason,
                        np.array2string(self._x1_state.follow1_pos, precision=4),
                        np.array2string(self._x1_state.follow2_pos, precision=4),
                    )

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
                "raw_action": raw_action.reshape(-1).copy(),
                "executed_action": executed_action,
                "action_clipped": False,
                "clip_delta_max": 0.0,
                "direct_pose_limiter_enabled": self.config.direct_pose_limiter_enabled,
                "direct_pose_limited": bool(
                    np.max(np.abs(executed_action - raw_action.reshape(-1))) > 1e-6
                )
                if not action_rejected
                else False,
                "direct_pose_delta_max": float(
                    np.max(np.abs(executed_action - raw_action.reshape(-1)))
                )
                if not action_rejected
                else 0.0,
                "action_rejected": action_rejected,
                "rejection_reason": rejection_reason,
                "last_published_action": last_published_action,
                "pose_control_backend": pose_control_backend,
            }
            return observation, reward, terminated, truncated, info

        assert raw_action.shape == expected_shape, (
            f"Action shape must be {expected_shape}, but got {raw_action.shape}."
        )
        raw_action = raw_action.reshape(-1).copy()
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
            "action_rejected": False,
            "rejection_reason": None,
            "last_published_action": executed_action.copy(),
            "pose_control_backend": pose_control_backend,
        }
        self._last_published_action = executed_action.copy()
        return observation, reward, terminated, truncated, info

    def step_joint(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        expected_shape = (len(self.config.use_arm_ids) * 7,)
        assert action.shape == expected_shape, (
            f"Joint action shape must be {expected_shape}, but got {action.shape}."
        )

        start_time = time.time()
        raw_joint_action = action.reshape(-1).copy()
        joint_target = np.clip(
            raw_joint_action,
            self._joint_action_space.low,
            self._joint_action_space.high,
        ).astype(np.float32, copy=False)

        current_joints = np.stack(
            [self._x1_state.follow1_joints, self._x1_state.follow2_joints]
        ).astype(np.float32)
        current_selected = np.concatenate(
            [current_joints[arm_id] for arm_id in self.config.use_arm_ids]
        )
        if np.isfinite(self.config.max_joint_delta):
            max_delta = float(self.config.max_joint_delta)
            joint_target = np.clip(
                joint_target,
                current_selected - max_delta,
                current_selected + max_delta,
            ).astype(np.float32, copy=False)

        if self.config.enforce_gripper_close:
            joint_target = joint_target.copy()
            for row_idx in range(len(self.config.use_arm_ids)):
                joint_target[row_idx * 7 + 6] = self.config.gripper_width_limit_min

        executed_joint_action = joint_target.reshape(-1).copy()
        clip_delta = np.abs(executed_joint_action - raw_joint_action)
        clip_delta_max = float(np.max(clip_delta)) if clip_delta.size else 0.0
        joint_action_clipped = bool(clip_delta_max > 1e-6)

        next_joints = current_joints.copy()
        target_rows = executed_joint_action.reshape(-1, 7)
        for action_row, arm_id in zip(
            target_rows, self.config.use_arm_ids, strict=False
        ):
            next_joints[arm_id] = action_row

        if not self.config.is_dummy:
            self._controller.move_joint(
                next_joints[0].tolist(), next_joints[1].tolist()
            ).wait()
        else:
            self._x1_state.follow1_joints = next_joints[0].copy()
            self._x1_state.follow2_joints = next_joints[1].copy()

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
            "raw_joint_action": raw_joint_action,
            "executed_joint_action": executed_joint_action,
            "joint_action_clipped": joint_action_clipped,
            "joint_clip_delta_max": clip_delta_max,
            "takeover_control_mode": "joint",
        }
        return observation, reward, terminated, truncated, info

    def step_relative_pose(self, action: np.ndarray):
        return super().step(action)

    def step(self, action: np.ndarray):
        return self.step_relative_pose(action)

    def _calc_step_reward(self, observation) -> float:
        return 0.0
