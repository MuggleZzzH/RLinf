# Copyright 2025 The RLinf Authors.
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

import os

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv


class IsaaclabStackCubeEnv(IsaaclabBaseEnv):
    """Thin IsaacLab stack-cube wrapper aligned with the GR00T IsaacLab flow."""

    ENV_ID_ALIASES = (
        "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0",
        "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
    )

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )
        self.main_cam_cfg = self._build_main_cam_cfg(cfg.init_params)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    @staticmethod
    def _dedupe_keys(keys):
        deduped = []
        for key in keys:
            if key and key not in deduped:
                deduped.append(key)
        return deduped

    @classmethod
    def _build_main_cam_cfg(cls, init_params):
        main_cam_cfg = cls._cfg_get(init_params, "main_cam")
        if main_cam_cfg is None:
            table_cam_cfg = cls._cfg_get(init_params, "table_cam")
            return {
                "height": table_cam_cfg.height,
                "width": table_cam_cfg.width,
                "scene_keys": ["table_cam", "front_cam"],
                "obs_keys": ["table_cam", "front_cam"],
            }

        return {
            "height": main_cam_cfg.height,
            "width": main_cam_cfg.width,
            "scene_keys": cls._dedupe_keys(
                [
                    cls._cfg_get(main_cam_cfg, "scene_key"),
                    cls._cfg_get(main_cam_cfg, "fallback_scene_key"),
                    "table_cam",
                    "front_cam",
                ]
            ),
            "obs_keys": cls._dedupe_keys(
                [
                    cls._cfg_get(main_cam_cfg, "obs_key"),
                    cls._cfg_get(main_cam_cfg, "fallback_obs_key"),
                    "table_cam",
                    "front_cam",
                ]
            ),
        }

    @staticmethod
    def _get_first_attr(obj, keys):
        for key in keys:
            if hasattr(obj, key):
                return getattr(obj, key)
        raise AttributeError(f"None of the candidate camera keys exist on scene: {keys}")

    @staticmethod
    def _get_first_item(mapping, keys):
        for key in keys:
            if key in mapping:
                return mapping[key]
        raise KeyError(
            f"None of the candidate camera keys exist in policy obs: {keys}; "
            f"available={sorted(mapping.keys())}"
        )

    def _resolve_env_cfg(self, load_cfg_from_registry):
        candidate_ids = [self.isaaclab_env_id] + [
            env_id for env_id in self.ENV_ID_ALIASES if env_id != self.isaaclab_env_id
        ]
        last_error = None
        for env_id in candidate_ids:
            try:
                env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
                return env_id, env_cfg
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            f"Unable to resolve IsaacLab task id from aliases: {candidate_ids}"
        ) from last_error

    def _make_env_function(self):
        """Build the IsaacLab stack-cube task with camera overrides."""

        def make_env_isaaclab():
            # Force headless mode even if the host shell exported DISPLAY.
            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher
            from isaaclab_tasks.utils import load_cfg_from_registry

            sim_app = AppLauncher(headless=True, enable_cameras=True).app
            resolved_env_id, isaac_env_cfg = self._resolve_env_cfg(
                load_cfg_from_registry
            )

            isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs
            isaac_env_cfg.scene.wrist_cam.height = self.cfg.init_params.wrist_cam.height
            isaac_env_cfg.scene.wrist_cam.width = self.cfg.init_params.wrist_cam.width
            main_cam = self._get_first_attr(
                isaac_env_cfg.scene, self.main_cam_cfg["scene_keys"]
            )
            main_cam.height = self.main_cam_cfg["height"]
            main_cam.width = self.main_cam_cfg["width"]

            env = gym.make(
                resolved_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def step(self, actions=None, auto_reset=True):
        raw_obs, step_reward, terminations, truncations, _ = self.env.step(actions)

        step_reward = step_reward.clone()
        terminations = terminations.clone()
        truncations = truncations.clone()

        if self.video_cfg.save_video:
            self.images.append(self.add_image(raw_obs))

        obs = self._wrap_obs(raw_obs)
        self._elapsed_steps += 1

        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations
        dones = terminations | truncations
        success_flags = terminations & (step_reward > 0)

        infos = self._record_metrics(
            step_reward=step_reward,
            terminations=terminations,
            infos={},
            success_flags=success_flags,
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = success_flags
            terminations[:] = False

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            step_reward,
            terminations,
            truncations,
            infos,
        )

    def _wrap_obs(self, obs):
        instruction = [self.task_description] * self.num_envs
        wrist_image = obs["policy"]["wrist_cam"]
        main_image = self._get_first_item(obs["policy"], self.main_cam_cfg["obs_keys"])
        quat = obs["policy"]["eef_quat"][
            :, [1, 2, 3, 0]
        ]  # IsaacLab stores quaternion as wxyz.
        states = torch.concatenate(
            [
                obs["policy"]["eef_pos"],
                quat2axisangle_torch(quat),
                obs["policy"]["gripper_pos"],
            ],
            dim=1,
        )

        env_obs = {
            "main_images": main_image,
            "task_descriptions": instruction,
            "states": states,
            "wrist_images": wrist_image,
        }
        return env_obs

    def add_image(self, obs):
        return self._get_first_item(obs["policy"], self.main_cam_cfg["obs_keys"])[
            0
        ].cpu().numpy()
