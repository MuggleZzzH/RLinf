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

from rlinf.envs.realworld.common.wrappers.keyboard_running_mode_wrapper import (
    KeyboardRunningModeWrapper,
)


class FakeKeyboardListener:
    def __init__(self, keys):
        self.keys = list(keys)
        self.index = 0

    def get_key(self):
        if self.index >= len(self.keys):
            return None
        key = self.keys[self.index]
        self.index += 1
        return key


class DummyEnv(gym.Env):
    def __init__(self, events=None):
        self.events = events
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        if self.events is not None:
            self.events.append("env_step")
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}


def test_keyboard_running_mode_sets_ros_param_before_inner_step():
    events = []
    writes = []

    wrapped = KeyboardRunningModeWrapper(
        DummyEnv(events),
        config={"enabled": True},
        listener=FakeKeyboardListener(["2"]),
        param_setter=lambda name, value: (
            events.append("set_param"),
            writes.append((name, value)),
        ),
        time_fn=lambda: 10.0,
    )

    wrapped.step(np.zeros(1, dtype=np.float32))

    assert events == ["set_param", "env_step"]
    assert writes == [("/running_mode", 2)]


def test_keyboard_running_mode_maps_normal_and_takeover_keys():
    writes = []
    times = iter([1.0, 2.0])
    wrapped = KeyboardRunningModeWrapper(
        DummyEnv(),
        config={
            "running_mode_param": "/running_mode",
            "normal_key": "1",
            "takeover_key": "2",
            "debounce_s": 0.3,
        },
        listener=FakeKeyboardListener(["1", None, "2"]),
        param_setter=lambda name, value: writes.append((name, value)),
        time_fn=lambda: next(times),
    )

    wrapped.step(np.zeros(1, dtype=np.float32))
    wrapped.step(np.zeros(1, dtype=np.float32))
    wrapped.step(np.zeros(1, dtype=np.float32))

    assert writes == [("/running_mode", 1), ("/running_mode", 2)]


def test_keyboard_running_mode_debounces_repeated_key():
    writes = []
    wrapped = KeyboardRunningModeWrapper(
        DummyEnv(),
        config={"debounce_s": 0.3},
        listener=FakeKeyboardListener(["2", "2", None, "2"]),
        param_setter=lambda name, value: writes.append((name, value)),
        time_fn=lambda: 1.0,
    )

    for _ in range(4):
        wrapped.step(np.zeros(1, dtype=np.float32))

    assert writes == [("/running_mode", 2)]


def test_keyboard_running_mode_ignores_unmapped_keys():
    writes = []
    wrapped = KeyboardRunningModeWrapper(
        DummyEnv(),
        listener=FakeKeyboardListener(["a", "q"]),
        param_setter=lambda name, value: writes.append((name, value)),
        time_fn=lambda: 1.0,
    )

    wrapped.step(np.zeros(1, dtype=np.float32))
    wrapped.step(np.zeros(1, dtype=np.float32))

    assert writes == []
