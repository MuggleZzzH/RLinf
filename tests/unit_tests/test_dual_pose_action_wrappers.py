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

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.realworld.common.wrappers import apply_dual_pose_action_wrappers
from rlinf.envs.realworld.franka.utils import construct_adjoint_matrix
from rlinf.models.embodiment.openpi.fold_towel_obs import process_fold_towel_s2s_obs


class DummyDualPoseEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(14,), dtype=np.float32
                        )
                    }
                )
            }
        )
        self.absolute_action_space = gym.spaces.Box(
            low=np.full((14,), -2.0, dtype=np.float32),
            high=np.full((14,), 2.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.gripper_widths = np.array([0.25, 0.75], dtype=np.float32)
        self.last_mode = None
        self.last_action = None

    @property
    def config(self):
        class Config:
            is_dummy = True

        return Config()

    def get_absolute_pose_action_space(self):
        return self.absolute_action_space

    def get_relative_pose_action_space(self):
        return self.action_space

    def get_gripper_widths(self):
        return self.gripper_widths

    def get_joint_snapshot(self):
        return np.zeros((2, 7), dtype=np.float32)

    def get_arm_pose_snapshot(self):
        return np.zeros((2, 7), dtype=np.float32)

    def step_absolute_pose(self, action):
        self.last_mode = "absolute_pose"
        self.last_action = np.asarray(action)
        return self._obs(), 0.0, False, False, {}

    def step_relative_pose(self, action):
        self.last_mode = "relative_pose"
        self.last_action = np.asarray(action)
        return self._obs(), 0.0, False, False, {}

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def _obs(self):
        # Two identity quaternions, laid out as [xyz, quat] per arm.
        return {
            "state": {
                "tcp_pose": np.array(
                    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0] * 2,
                    dtype=np.float32,
                )
            }
        }


def test_dual_pose_builder_absolute_mode_routes_and_converts_euler_obs():
    env = DummyDualPoseEnv()
    wrapped = apply_dual_pose_action_wrappers(env, {"action_mode": "absolute_pose"})

    action = np.full((14,), 1.5, dtype=np.float32)
    obs, reward, terminated, truncated, info = wrapped.step(action)

    assert wrapped.action_space.low[0] == -2.0
    assert env.last_mode == "absolute_pose"
    np.testing.assert_array_equal(env.last_action, action)
    np.testing.assert_allclose(
        obs["state"]["tcp_pose"],
        np.array(
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.25, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.75],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info == {}


def test_dual_pose_builder_relative_mode_applies_relative_frame_by_default():
    env = DummyDualPoseEnv()
    wrapped = apply_dual_pose_action_wrappers(env, {"action_mode": "relative_pose"})

    # reset() initialises the adjoint matrices inside DualRelativeFrame.
    reset_obs, _ = wrapped.reset()
    np.testing.assert_allclose(
        reset_obs["state"]["tcp_pose"],
        np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
            dtype=np.float32,
        ),
        atol=1e-6,
    )

    action = np.full((14,), 0.5, dtype=np.float32)
    obs, *_ = wrapped.step(action)

    assert wrapped.action_space.low[0] == -1.0
    assert env.last_mode == "relative_pose"
    expected_action = action.copy()
    reset_pose = env._obs()["state"]["tcp_pose"]
    expected_action[:6] = construct_adjoint_matrix(reset_pose[:7]) @ action[:6]
    expected_action[7:13] = construct_adjoint_matrix(reset_pose[7:]) @ action[7:13]
    np.testing.assert_allclose(env.last_action, expected_action)
    assert not np.allclose(env.last_action, action)
    assert obs["state"]["tcp_pose"].shape == (14,)
    assert obs["state"]["tcp_pose"][6] == pytest.approx(0.25)
    assert obs["state"]["tcp_pose"][13] == pytest.approx(0.75)


def test_dual_pose_builder_relative_mode_no_relative_frame():
    env = DummyDualPoseEnv()
    wrapped = apply_dual_pose_action_wrappers(
        env, {"action_mode": "relative_pose", "use_relative_frame": False}
    )

    action = np.full((14,), 0.5, dtype=np.float32)
    obs, *_ = wrapped.step(action)

    assert env.last_mode == "relative_pose"
    np.testing.assert_array_equal(env.last_action, action)
    assert obs["state"]["tcp_pose"].shape == (14,)


def test_dual_pose_builder_rejects_unknown_action_mode():
    with pytest.raises(ValueError, match="Unsupported action_mode"):
        apply_dual_pose_action_wrappers(DummyDualPoseEnv(), {"action_mode": "joint"})


def test_dual_pose_builder_rejects_relative_master_takeover():
    with pytest.raises(ValueError, match="requires action_mode='absolute_pose'"):
        apply_dual_pose_action_wrappers(
            DummyDualPoseEnv(),
            {"action_mode": "relative_pose", "use_master_takeover": True},
        )


def test_fold_towel_obs_processor_maps_clean_euler_obs():
    main_images = np.zeros((2, 128, 128, 3), dtype=np.uint8)
    extra_view_images = np.stack(
        [
            np.ones((2, 128, 128, 3), dtype=np.uint8),
            np.full((2, 128, 128, 3), 2, dtype=np.uint8),
        ],
        axis=1,
    )
    states = np.arange(28, dtype=np.float32).reshape(2, 14)
    task_descriptions = ["fold the towel", "fold the towel"]

    processed = process_fold_towel_s2s_obs(
        {
            "main_images": main_images,
            "extra_view_images": extra_view_images,
            "states": states,
            "task_descriptions": task_descriptions,
        },
    )

    assert set(processed) == {"images", "state", "prompt"}
    np.testing.assert_array_equal(
        processed["images"]["left_wrist_view"], extra_view_images[:, 0]
    )
    np.testing.assert_array_equal(processed["images"]["face_view"], main_images)
    np.testing.assert_array_equal(
        processed["images"]["right_wrist_view"], extra_view_images[:, 1]
    )
    np.testing.assert_array_equal(processed["state"], states)
    assert processed["prompt"] == task_descriptions
