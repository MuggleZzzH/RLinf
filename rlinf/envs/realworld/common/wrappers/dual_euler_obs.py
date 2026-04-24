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

"""Quaternion-to-Euler observation wrapper for dual-arm pose observations."""

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from scipy.spatial.transform import Rotation as R


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """Convert dual-arm ``xyz + quat`` TCP pose to euler pose with grippers.

    This wrapper is robot-agnostic: it only requires the observation to contain
    ``state.tcp_pose`` laid out as two ``xyz + quat`` pose vectors.  If the
    wrapped env exposes ``get_gripper_widths()``, its two gripper channels are
    inserted after each arm pose to produce
    ``[left_xyz, left_rpy, left_gripper, right_xyz, right_rpy, right_gripper]``.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(14,)
        )

    def _get_gripper_widths(self, dtype) -> np.ndarray:
        try:
            gripper_widths = self.get_wrapper_attr("get_gripper_widths")()
        except AttributeError:
            gripper_widths = np.zeros(2, dtype=dtype)

        gripper_widths = np.asarray(gripper_widths, dtype=dtype).reshape(-1)
        if gripper_widths.shape != (2,):
            raise ValueError(
                "get_gripper_widths() must return two values for a dual-arm env, "
                f"got shape {gripper_widths.shape}."
            )
        return gripper_widths

    def observation(self, observation: dict) -> dict:
        """Convert dual-arm quaternion TCP pose to euler+gripper in-place."""
        tcp_pose = observation["state"]["tcp_pose"]
        left = tcp_pose[:7]
        right = tcp_pose[7:]
        gripper_widths = self._get_gripper_widths(tcp_pose.dtype)
        left_euler = np.concatenate(
            [
                left[:3],
                R.from_quat(left[3:].copy()).as_euler("xyz"),
                gripper_widths[:1],
            ]
        )
        right_euler = np.concatenate(
            [
                right[:3],
                R.from_quat(right[3:].copy()).as_euler("xyz"),
                gripper_widths[1:],
            ]
        )
        observation["state"]["tcp_pose"] = np.concatenate([left_euler, right_euler])
        return observation
