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
    """Override policy actions with remote master takeover actions."""

    def __init__(
        self,
        env,
        config: dict | None = None,
        adapter: X2RobotTakeoverTCPServer | None = None,
    ):
        super().__init__(env)

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

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        self.adapter.poll()
        expert_action = self.adapter.get_takeover_action()
        if expert_action is None:
            return action, False
        return expert_action.astype(np.float32, copy=False), True

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["takeover_active"] = self.adapter.is_takeover_active()
        info["master_takeover_connected"] = self.adapter.is_connected()
        return obs, rew, done, truncated, info

    def close(self):
        self.adapter.close()
        return self.env.close()
