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

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.takeover import (
    X2RobotTakeoverTCPConfig,
    X2RobotTakeoverTCPServer,
)
from rlinf.envs.realworld.common.takeover.x2robot_protocol import (
    make_ros_running_mode_getter,
)


class MasterTakeoverIntervention(gym.ActionWrapper):
    """Override absolute dual-arm policy actions with X2Robot master poses."""

    def __init__(
        self,
        env: gym.Env,
        config: Mapping[str, Any] | None = None,
        adapter: X2RobotTakeoverTCPServer | None = None,
    ):
        super().__init__(env)

        joint_snapshot = np.asarray(
            self.get_wrapper_attr("get_joint_snapshot")(), dtype=np.float32
        )
        if joint_snapshot.shape != (2, 7):
            raise ValueError(
                "Master takeover requires dual-arm joint snapshots with shape "
                f"(2, 7), got {joint_snapshot.shape}."
            )

        if adapter is None:
            tcp_cfg = X2RobotTakeoverTCPConfig.from_dict(config)
            adapter = X2RobotTakeoverTCPServer(
                config=tcp_cfg,
                running_mode_getter=make_ros_running_mode_getter(tcp_cfg),
                joint_snapshot_getter=self.get_wrapper_attr("get_joint_snapshot"),
                logger=logging.getLogger(__name__),
            )
        self.adapter = adapter
        self.adapter.start()

        self._chunk_active = False
        self._hold_until_chunk_end = False
        self._was_takeover_active = self.adapter.is_takeover_active()

    def begin_action_chunk(self) -> None:
        self._chunk_active = True
        self._hold_until_chunk_end = False

    def end_action_chunk(self) -> None:
        self._chunk_active = False
        self._hold_until_chunk_end = False

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool, bool]:
        self.adapter.poll()
        takeover_active = self.adapter.is_takeover_active()
        if self._chunk_active and self._was_takeover_active and not takeover_active:
            self._hold_until_chunk_end = True
        self._was_takeover_active = takeover_active

        expert_action = self.adapter.get_takeover_action()
        if expert_action is not None:
            return expert_action.astype(np.float32, copy=False), True, False, False

        if takeover_active:
            return self._hold_action(), False, False, True

        if self._hold_until_chunk_end:
            return self._hold_action(), False, True, False

        return action, False, False, False

    def step(self, action):
        new_action, replaced, chunk_holding, sync_holding = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        self.adapter.sync_control_plane()
        info["executed_action"] = new_action
        info["intervene_flag"] = np.asarray([replaced], dtype=bool)
        if replaced:
            info["intervene_action"] = new_action
        info["takeover_active"] = self.adapter.is_takeover_active()
        info["takeover_chunk_hold"] = chunk_holding
        info["takeover_sync_hold"] = sync_holding
        info["master_takeover_connected"] = self.adapter.is_connected()
        return obs, rew, done, truncated, info

    def close(self):
        self.adapter.close()
        return self.env.close()

    def _hold_action(self) -> np.ndarray:
        pose_snapshot = np.asarray(
            self.get_wrapper_attr("get_arm_pose_snapshot")(), dtype=np.float32
        )
        if pose_snapshot.shape != (2, 7):
            raise ValueError(
                "Master takeover chunk-boundary recovery requires dual-arm pose "
                f"snapshots with shape (2, 7), got {pose_snapshot.shape}."
            )
        return pose_snapshot.reshape(-1).astype(np.float32, copy=False)
