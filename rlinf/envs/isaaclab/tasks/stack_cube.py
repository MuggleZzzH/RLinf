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

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv


class IsaaclabStackCubeEnv(IsaaclabBaseEnv):
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
        reward_cfg = getattr(cfg, "reward_cfg", None)
        reward_terms_cfg = getattr(reward_cfg, "rewards", None)
        self.enable_dense_reward = bool(
            getattr(reward_cfg, "enable_dense_reward", False)
        )
        self.enable_stage_gating = bool(getattr(reward_cfg, "enable_stage_gating", True))
        self.drop_height_threshold = float(
            getattr(reward_cfg, "drop_height_threshold", -0.05)
        )
        self.reward_grasp_red = float(getattr(reward_terms_cfg, "grasp_red", 0.10))
        self.reward_stack_red_blue = float(
            getattr(reward_terms_cfg, "stack_red_blue", 0.25)
        )
        self.reward_grasp_green = float(getattr(reward_terms_cfg, "grasp_green", 0.10))
        self.reward_success = float(getattr(reward_terms_cfg, "success", 1.00))
        self.reward_fail_drop = float(getattr(reward_terms_cfg, "fail_drop", -0.30))

    def _make_env_function(self):
        """
        function for make isaaclab
        """

        def make_env_isaaclab():
            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app
            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            isaac_env_cfg.scene.num_envs = (
                self.cfg.init_params.num_envs
            )  # default 4096 ant_env_spaces.pkl

            isaac_env_cfg.scene.wrist_cam.height = self.cfg.init_params.wrist_cam.height
            isaac_env_cfg.scene.wrist_cam.width = self.cfg.init_params.wrist_cam.width
            isaac_env_cfg.scene.table_cam.height = self.cfg.init_params.table_cam.height
            isaac_env_cfg.scene.table_cam.width = self.cfg.init_params.table_cam.width

            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def _init_metrics(self):
        super()._init_metrics()
        self._stage_grasp_red_done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stage_stack_red_blue_done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stage_grasp_green_done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stage_success_done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stage_fail_drop_done = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _reset_metrics(self, env_idx=None):
        super()._reset_metrics(env_idx)
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self._stage_grasp_red_done[mask] = False
            self._stage_stack_red_blue_done[mask] = False
            self._stage_grasp_green_done[mask] = False
            self._stage_success_done[mask] = False
            self._stage_fail_drop_done[mask] = False
        else:
            self._stage_grasp_red_done[:] = False
            self._stage_stack_red_blue_done[:] = False
            self._stage_grasp_green_done[:] = False
            self._stage_success_done[:] = False
            self._stage_fail_drop_done[:] = False

    def _to_bool_tensor(self, value):
        if value is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self.device)
        value = value.to(self.device)
        if value.ndim > 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.dtype != torch.bool:
            value = value > 0.5
        return value

    def _get_stage_signals(self, raw_obs):
        subtask_terms = raw_obs.get("subtask_terms", {})
        return {
            "grasp_red": self._to_bool_tensor(subtask_terms.get("grasp_1")),
            "stack_red_blue": self._to_bool_tensor(subtask_terms.get("stack_1")),
            "grasp_green": self._to_bool_tensor(subtask_terms.get("grasp_2")),
        }

    def _get_success_fail_signals(self, raw_obs, terminations):
        cube_positions = raw_obs.get("policy", {}).get("cube_positions")
        if cube_positions is None:
            dropped = torch.zeros_like(terminations, dtype=torch.bool)
        else:
            cube_positions = cube_positions.to(self.device)
            cube_z = cube_positions[:, 2::3]
            dropped = (cube_z < self.drop_height_threshold).any(dim=1)
        success_terminated = terminations & (~dropped)
        fail_drop = terminations & dropped
        return success_terminated, fail_drop

    def step(self, actions=None, auto_reset=True):
        raw_obs, _, terminations, truncations, infos = self.env.step(actions)
        terminations = terminations.clone()
        truncations = truncations.clone()
        success_terminated, fail_drop = self._get_success_fail_signals(
            raw_obs, terminations
        )

        reward_scale = float(self.cfg.reward_coef)
        reward_terms = {
            "reward/grasp_red": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "reward/stack_red_blue": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "reward/grasp_green": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "reward/success": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "reward/fail_drop": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
        }
        step_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        if self.enable_dense_reward:
            stage_signals = self._get_stage_signals(raw_obs)
            if self.enable_stage_gating:
                new_grasp_red = stage_signals["grasp_red"] & (~self._stage_grasp_red_done)
                new_stack_red_blue = (
                    stage_signals["stack_red_blue"]
                    & self._stage_grasp_red_done
                    & (~self._stage_stack_red_blue_done)
                )
                new_grasp_green = (
                    stage_signals["grasp_green"]
                    & self._stage_stack_red_blue_done
                    & (~self._stage_grasp_green_done)
                )
            else:
                new_grasp_red = stage_signals["grasp_red"] & (~self._stage_grasp_red_done)
                new_stack_red_blue = (
                    stage_signals["stack_red_blue"] & (~self._stage_stack_red_blue_done)
                )
                new_grasp_green = (
                    stage_signals["grasp_green"] & (~self._stage_grasp_green_done)
                )

            new_success = success_terminated & (~self._stage_success_done)
            new_fail_drop = fail_drop & (~self._stage_fail_drop_done)

            reward_terms["reward/grasp_red"] = (
                reward_scale * self.reward_grasp_red * new_grasp_red.float()
            )
            reward_terms["reward/stack_red_blue"] = (
                reward_scale * self.reward_stack_red_blue * new_stack_red_blue.float()
            )
            reward_terms["reward/grasp_green"] = (
                reward_scale * self.reward_grasp_green * new_grasp_green.float()
            )
            reward_terms["reward/success"] = (
                reward_scale * self.reward_success * new_success.float()
            )
            reward_terms["reward/fail_drop"] = (
                reward_scale * self.reward_fail_drop * new_fail_drop.float()
            )

            step_reward = (
                reward_terms["reward/grasp_red"]
                + reward_terms["reward/stack_red_blue"]
                + reward_terms["reward/grasp_green"]
                + reward_terms["reward/success"]
                + reward_terms["reward/fail_drop"]
            )

            self._stage_grasp_red_done |= stage_signals["grasp_red"]
            self._stage_stack_red_blue_done |= stage_signals["stack_red_blue"]
            self._stage_grasp_green_done |= stage_signals["grasp_green"]
            self._stage_success_done |= success_terminated
            self._stage_fail_drop_done |= fail_drop
        else:
            # Sparse baseline only rewards true task success (not every termination).
            reward_terms["reward/success"] = reward_scale * success_terminated.float()
            step_reward = reward_terms["reward/success"]

        if self.video_cfg.save_video:
            self.images.append(self.add_image(raw_obs))

        obs = self._wrap_obs(raw_obs)

        self._elapsed_steps += 1

        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations

        dones = terminations | truncations

        infos = self._record_metrics(
            step_reward=step_reward,
            terminations=terminations,
            infos={},
            success_flags=success_terminated,
            reward_terms=reward_terms,
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = success_terminated
            terminations[:] = False

        _auto_reset = auto_reset and self.auto_reset  # always False
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
        table_image = obs["policy"]["table_cam"]
        quat = obs["policy"]["eef_quat"][:, [1, 2, 3, 0]]
        states = torch.concatenate(
            [
                obs["policy"]["eef_pos"],
                quat2axisangle_torch(quat),
                obs["policy"]["gripper_pos"],
            ],
            dim=1,
        )

        env_obs = {
            "main_images": table_image,
            "task_descriptions": instruction,
            "states": states,
            "wrist_images": wrist_image,
        }
        return env_obs

    def add_image(self, obs):
        img = obs["policy"]["table_cam"][0].cpu().numpy()
        return img
