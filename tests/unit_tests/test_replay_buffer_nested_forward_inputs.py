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

import torch

from rlinf.data.embodied_io_struct import (
    EmbodiedRolloutResult,
    convert_trajectories_to_batch,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def test_replay_buffer_preserves_nested_forward_inputs_for_dagger_s2s():
    rollout = EmbodiedRolloutResult(max_episode_length=2)
    for _ in range(2):
        rollout.actions.append(torch.zeros(1, 420))
        rollout.intervene_flags.append(torch.ones(1, 420, dtype=torch.bool))
        rollout.rewards.append(torch.zeros(1))
        rollout.forward_inputs.append(
            {
                "images": {
                    "left_wrist_view": torch.ones(1, 4, 4, 3),
                    "face_view": torch.full((1, 4, 4, 3), 2),
                    "right_wrist_view": torch.full((1, 4, 4, 3), 3),
                },
                "state": torch.arange(14, dtype=torch.float32).reshape(1, 14),
                "action": torch.zeros(1, 420),
                "model_action": torch.ones(1, 420),
            }
        )

    trajectory = rollout.to_trajectory()
    assert set(trajectory.forward_inputs["images"]) == {
        "left_wrist_view",
        "face_view",
        "right_wrist_view",
    }
    assert trajectory.forward_inputs["images"]["left_wrist_view"].shape == (
        2,
        1,
        4,
        4,
        3,
    )

    extracted = trajectory.extract_intervene_traj(mode="all")
    assert extracted is not None
    assert extracted[0].forward_inputs["images"]["left_wrist_view"].shape == (
        2,
        1,
        4,
        4,
        3,
    )

    buffer = TrajectoryReplayBuffer(
        seed=0,
        enable_cache=True,
        cache_size=1,
        sample_window_size=1,
        auto_save=False,
    )

    buffer.add_trajectories(extracted)
    batch = buffer.sample(num_chunks=1)

    forward_inputs = batch["forward_inputs"]
    assert set(forward_inputs["images"]) == {
        "left_wrist_view",
        "face_view",
        "right_wrist_view",
    }
    assert forward_inputs["images"]["left_wrist_view"].shape == (1, 4, 4, 3)
    assert forward_inputs["images"]["face_view"].shape == (1, 4, 4, 3)
    assert forward_inputs["images"]["right_wrist_view"].shape == (1, 4, 4, 3)
    assert forward_inputs["state"].shape == (1, 14)
    assert forward_inputs["action"].shape == (1, 420)
    assert forward_inputs["model_action"].shape == (1, 420)


def test_convert_trajectories_to_batch_preserves_nested_forward_inputs():
    rollout = EmbodiedRolloutResult(max_episode_length=1)
    rollout.actions.append(torch.zeros(1, 420))
    rollout.intervene_flags.append(torch.ones(1, 420, dtype=torch.bool))
    rollout.rewards.append(torch.zeros(1))
    rollout.forward_inputs.append(
        {
            "images": {
                "left_wrist_view": torch.ones(1, 4, 4, 3),
            },
            "state": torch.arange(14, dtype=torch.float32).reshape(1, 14),
        }
    )
    trajectory = rollout.to_trajectory()

    batch = convert_trajectories_to_batch([trajectory, trajectory])
    assert batch["forward_inputs"]["images"]["left_wrist_view"].shape == (
        1,
        2,
        4,
        4,
        3,
    )
    assert batch["forward_inputs"]["state"].shape == (1, 2, 14)


def test_extract_intervene_traj_all_drops_partial_chunk():
    rollout = EmbodiedRolloutResult(max_episode_length=1)
    rollout.actions.append(torch.zeros(1, 420))
    flags = torch.zeros(1, 420, dtype=torch.bool)
    flags[:, :14] = True
    rollout.intervene_flags.append(flags)
    rollout.rewards.append(torch.zeros(1))
    rollout.forward_inputs.append(
        {
            "images": {
                "left_wrist_view": torch.ones(1, 4, 4, 3),
            },
            "state": torch.arange(14, dtype=torch.float32).reshape(1, 14),
        }
    )
    trajectory = rollout.to_trajectory()

    assert trajectory.extract_intervene_traj(mode="all") is None
    assert trajectory.extract_intervene_traj(mode="any") is not None
