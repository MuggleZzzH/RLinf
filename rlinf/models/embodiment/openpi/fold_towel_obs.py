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

"""Observation processing for the fold_towel OpenPI s2s path."""


def process_fold_towel_s2s_obs(env_obs: dict) -> dict:
    """Map clean realworld env observations to X2Robot/OpenPI s2s inputs."""
    extra_view_images = env_obs.get("extra_view_images")
    if extra_view_images is None or extra_view_images.shape[1] < 2:
        raise ValueError(
            "fold_towel_s2s requires extra_view_images with left/right wrist views."
        )

    states = env_obs["states"]
    if states.shape[-1] != 14:
        raise ValueError(
            "fold_towel_s2s requires 14D Euler+gripper states, "
            f"got shape {states.shape}."
        )

    return {
        "images": {
            "left_wrist_view": extra_view_images[:, 0],
            "face_view": env_obs["main_images"],
            "right_wrist_view": extra_view_images[:, 1],
        },
        "state": states,
        "prompt": env_obs["task_descriptions"],
    }
