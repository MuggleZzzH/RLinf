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

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.realworld.common.wrappers.master_takeover_intervention import (
    MasterTakeoverIntervention,
)


class FakeTakeoverAdapter:
    def __init__(self, states, initial_active=False):
        self.states = list(states)
        self.initial_active = initial_active
        self.index = -1
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def poll(self):
        self.index += 1

    def is_takeover_active(self):
        if self.index < 0:
            return self.initial_active
        return bool(self.states[min(self.index, len(self.states) - 1)][0])

    def is_connected(self):
        return True

    def get_takeover_action(self):
        if self.index < 0:
            return None
        action = self.states[min(self.index, len(self.states) - 1)][1]
        return None if action is None else np.asarray(action, dtype=np.float32)


class DummyTakeoverEnv(gym.Env):
    def __init__(self, joint_shape=(2, 7)):
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, shape=(14,), dtype=np.float32
        )
        self.last_action = None
        self.joint_snapshot = np.zeros(joint_shape, dtype=np.float32)
        self.pose_snapshot = np.arange(14, dtype=np.float32).reshape(2, 7)

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return np.zeros(14, dtype=np.float32), 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return np.zeros(14, dtype=np.float32), {}

    def get_joint_snapshot(self):
        return self.joint_snapshot

    def get_arm_pose_snapshot(self):
        return self.pose_snapshot


def test_master_takeover_mode_1_passes_policy_action():
    env = DummyTakeoverEnv()
    adapter = FakeTakeoverAdapter(states=[(False, None)])
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    action = np.ones(14, dtype=np.float32)
    _, _, _, _, info = wrapped.step(action)

    np.testing.assert_array_equal(env.last_action, action)
    np.testing.assert_array_equal(info["executed_action"], action)
    assert info["intervene_flag"].shape == (1,)
    assert not info["intervene_flag"].item()
    assert "intervene_action" not in info


def test_master_takeover_mode_2_without_fresh_pose_passes_policy_action():
    env = DummyTakeoverEnv()
    adapter = FakeTakeoverAdapter(states=[(True, None)])
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    action = np.ones(14, dtype=np.float32)
    _, _, _, _, info = wrapped.step(action)

    np.testing.assert_array_equal(env.last_action, action)
    assert info["takeover_active"]
    assert not info["intervene_flag"].item()
    assert "intervene_action" not in info


def test_master_takeover_mode_2_fresh_pose_overrides_policy_action():
    env = DummyTakeoverEnv()
    expert = np.arange(14, dtype=np.float32)
    adapter = FakeTakeoverAdapter(states=[(True, expert)])
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    policy = np.ones(14, dtype=np.float32)
    _, _, _, _, info = wrapped.step(policy)

    np.testing.assert_array_equal(env.last_action, expert)
    np.testing.assert_array_equal(info["executed_action"], expert)
    np.testing.assert_array_equal(info["intervene_action"], expert)
    assert info["intervene_flag"].item()


def test_master_takeover_holds_after_takeover_until_chunk_boundary():
    env = DummyTakeoverEnv()
    expert = np.full(14, 3.0, dtype=np.float32)
    adapter = FakeTakeoverAdapter(
        states=[
            (True, expert),
            (False, None),
            (False, None),
        ]
    )
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    wrapped.begin_action_chunk()
    wrapped.step(np.ones(14, dtype=np.float32))
    _, _, _, _, hold_info = wrapped.step(np.full(14, 2.0, dtype=np.float32))

    expected_hold = env.pose_snapshot.reshape(-1)
    np.testing.assert_array_equal(env.last_action, expected_hold)
    assert hold_info["takeover_chunk_hold"]
    assert not hold_info["intervene_flag"].item()

    wrapped.end_action_chunk()
    policy_after_boundary = np.full(14, 4.0, dtype=np.float32)
    wrapped.step(policy_after_boundary)
    np.testing.assert_array_equal(env.last_action, policy_after_boundary)


def test_master_takeover_rejects_bad_joint_snapshot_shape():
    with pytest.raises(ValueError, match=r"shape \(2, 7\)"):
        MasterTakeoverIntervention(
            DummyTakeoverEnv(joint_shape=(1, 7)),
            adapter=FakeTakeoverAdapter(states=[]),
        )
