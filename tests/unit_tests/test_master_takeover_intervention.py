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

import socket

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.takeover.x2robot_protocol import (
    MSG_POSE,
    recv_frame,
    send_frame,
)
from rlinf.envs.realworld.common.wrappers.master_takeover_intervention import (
    MasterTakeoverIntervention,
)


class _DummyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-np.ones(14, dtype=np.float32),
            high=np.ones(14, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.ones(1, dtype=np.float32),
            high=np.ones(1, dtype=np.float32),
            dtype=np.float32,
        )
        self.last_action = None

    def step(self, action):
        self.last_action = action.copy()
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def close(self):
        return None


class _FakeAdapter:
    def __init__(self, expert_action=None):
        self._expert_action = expert_action
        self._started = False
        self._closed = False

    def start(self):
        self._started = True

    def close(self):
        self._closed = True

    def poll(self):
        return None

    def get_takeover_action(self):
        return self._expert_action

    def is_takeover_active(self):
        return self._expert_action is not None

    def is_connected(self):
        return True


def test_send_recv_frame_roundtrip_pose():
    left = [float(idx) for idx in range(7)]
    right = [float(idx + 10) for idx in range(7)]
    sock_a, sock_b = socket.socketpair()
    try:
        send_frame(
            sock_a,
            {
                "header": {"msg_type": MSG_POSE, "seq": 1, "timestamp_us": 123},
                "payload": {"pose_left": left, "pose_right": right},
            },
        )
        frame = recv_frame(sock_b)
    finally:
        sock_a.close()
        sock_b.close()

    assert frame["header"]["msg_type"] == MSG_POSE
    assert frame["payload"]["pose_left"] == left
    assert frame["payload"]["pose_right"] == right


def test_master_takeover_wrapper_overrides_policy_action():
    expert_action = np.arange(14, dtype=np.float32)
    env = MasterTakeoverIntervention(
        _DummyEnv(),
        adapter=_FakeAdapter(expert_action=expert_action),
    )

    _, _, _, _, info = env.step(np.zeros(14, dtype=np.float32))

    assert np.allclose(env.env.last_action, expert_action)
    assert np.allclose(info["intervene_action"], expert_action)
    assert info["takeover_active"] is True
    assert info["master_takeover_connected"] is True


def test_master_takeover_wrapper_keeps_policy_action_without_expert():
    env = MasterTakeoverIntervention(
        _DummyEnv(),
        adapter=_FakeAdapter(expert_action=None),
    )
    policy_action = np.ones(14, dtype=np.float32) * 0.5

    _, _, _, _, info = env.step(policy_action)

    assert np.allclose(env.env.last_action, policy_action)
    assert "intervene_action" not in info
    assert info["takeover_active"] is False
