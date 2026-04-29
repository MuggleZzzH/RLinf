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

import pickle
import types

import gymnasium as gym
import numpy as np

from rlinf.envs.wrappers.collect_episode import CollectEpisode


class OneStepCollectionEnv(gym.Env):
    def __init__(self, intervened=False, success=True):
        self.intervened = intervened
        self.success = success
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(-10.0, 10.0, shape=(1,), dtype=np.float32)}
        )

    def reset(self, *, seed=None, options=None):
        return {"state": np.zeros((1, 1), dtype=np.float32)}, {}

    def step(self, action):
        info = {}
        if self.intervened:
            info["intervene_flag"] = np.asarray([True], dtype=bool)
            info["executed_action"] = np.asarray([7.0], dtype=np.float32)
        info["success"] = self.success
        return {"state": np.ones((1, 1), dtype=np.float32)}, 0.0, True, False, info


class ExplicitFalseInterveneEnv:
    def step(self, action):
        obs = {
            "state": {"tcp_pose": np.zeros((1, 1), dtype=np.float32)},
            "frames": {"main": np.zeros((1, 4, 4, 3), dtype=np.uint8)},
        }
        infos = {
            "intervene_flag": np.asarray([False], dtype=bool),
            "intervene_action": [np.asarray([9.0], dtype=np.float32)],
        }
        return (
            obs,
            np.asarray([0.0], dtype=np.float32),
            np.asarray([False], dtype=bool),
            np.asarray([False], dtype=bool),
            infos,
        )


def test_realworld_step_respects_explicit_false_intervene_flag():
    from rlinf.envs.realworld.realworld_env import RealWorldEnv

    env = RealWorldEnv.__new__(RealWorldEnv)
    env.cfg = types.SimpleNamespace(max_episode_steps=10)
    env.num_envs = 1
    env.env = ExplicitFalseInterveneEnv()
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env.manual_episode_control_only = False
    env.ignore_terminations = False
    env.auto_reset = False
    env.main_image_key = "main"
    env.task_descriptions = ["test"]
    env._init_metrics()

    _, _, _, _, infos = env.step(np.zeros((1, 1), dtype=np.float32), auto_reset=False)

    assert not bool(infos["intervene_flag"].item())
    assert not bool(infos["episode"]["intervened_once"].item())


def test_collect_episode_only_intervened_skips_non_intervened(tmp_path):
    wrapped = CollectEpisode(
        OneStepCollectionEnv(intervened=False),
        save_dir=str(tmp_path),
        only_intervened=True,
    )
    wrapped.reset()
    wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    wrapped.close()

    assert list(tmp_path.glob("*.pkl")) == []


def test_collect_episode_records_full_intervened_episode_with_executed_action(tmp_path):
    wrapped = CollectEpisode(
        OneStepCollectionEnv(intervened=True),
        save_dir=str(tmp_path),
        only_intervened=True,
    )
    wrapped.reset()
    wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    wrapped.close()

    [path] = list(tmp_path.glob("*.pkl"))
    with path.open("rb") as f:
        episode = pickle.load(f)

    assert episode["intervened"]
    np.testing.assert_array_equal(episode["actions"][0], np.asarray([7.0]))


def test_collect_episode_success_filter_is_independent_from_intervention(tmp_path):
    success_dir = tmp_path / "success"
    success_wrapped = CollectEpisode(
        OneStepCollectionEnv(intervened=False, success=True),
        save_dir=str(success_dir),
        only_success=True,
        only_intervened=False,
    )
    success_wrapped.reset()
    success_wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    success_wrapped.close()

    [success_path] = list(success_dir.glob("*.pkl"))
    with success_path.open("rb") as f:
        success_episode = pickle.load(f)
    assert success_episode["success"] is True
    assert success_episode["intervened"] is False

    failed_dir = tmp_path / "failed"
    failed_wrapped = CollectEpisode(
        OneStepCollectionEnv(intervened=True, success=False),
        save_dir=str(failed_dir),
        only_success=True,
        only_intervened=False,
    )
    failed_wrapped.reset()
    failed_wrapped.step(np.asarray([[1.0]], dtype=np.float32))
    failed_wrapped.close()

    assert list(failed_dir.glob("*.pkl")) == []
