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

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict


def test_x1_fold_towel_dagger_config_composes(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "examples" / "embodiment" / "config"
    monkeypatch.setenv("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="realworld_x1_fold_towel_dagger_openpi")
    finally:
        GlobalHydra.instance().clear()

    assert cfg.runner.only_eval is False
    assert cfg.algorithm.loss_type == "embodied_dagger"
    assert cfg.algorithm.dagger.only_save_expert is True
    assert cfg.rollout.collect_transitions is True
    assert "expert_model" not in cfg.rollout
    assert cfg.actor.model.openpi.config_name == "fold_towel_s2s"
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 30
    assert cfg.env.train.use_master_takeover is True
    assert cfg.env.train.master_takeover.port == 8766
    assert cfg.env.train.master_takeover.control_mode == "pose"
    assert cfg.env.train.master_takeover.max_pose_age_s == 0.25
    assert cfg.env.train.master_takeover.max_joint_age_s == 0.25
    assert cfg.env.train.data_collection.save_dir == "../results/x1_dagger_rollouts"
    assert cfg.env.train.data_collection.export_format == "lerobot"
    assert cfg.env.train.data_collection.only_success is True
    assert cfg.env.train.data_collection.only_intervened is False
    assert cfg.env.eval.use_master_takeover is False


def test_x1_fold_towel_takeover_collect_config_composes(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "examples" / "embodiment" / "config"
    monkeypatch.setenv("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="realworld_x1_fold_towel_takeover_collect_openpi")
    finally:
        GlobalHydra.instance().clear()

    assert cfg.runner.only_eval is True
    assert cfg.rollout.collect_transitions is False
    assert cfg.env.eval.use_master_takeover is True
    assert cfg.env.eval.action_mode == "absolute_pose"
    assert cfg.env.eval.master_takeover.port == 8766
    assert cfg.env.eval.master_takeover.control_mode == "pose"
    assert cfg.env.eval.master_takeover.max_pose_age_s == 0.25
    assert cfg.env.eval.master_takeover.max_joint_age_s == 0.25
    assert cfg.env.eval.keyboard_reward_wrapper == "single_stage"
    assert cfg.env.eval.data_collection.enabled is True
    assert cfg.env.eval.data_collection.export_format == "lerobot"
    assert cfg.env.eval.data_collection.only_success is True
    assert cfg.env.eval.data_collection.only_intervened is False
    assert cfg.actor.model.openpi.config_name == "fold_towel_s2s"
    assert cfg.actor.model.action_dim == 14
    assert cfg.actor.model.num_action_chunks == 30


def test_x1_fold_towel_takeover_collect_joint_config_composes(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "examples" / "embodiment" / "config"
    monkeypatch.setenv("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="realworld_x1_fold_towel_takeover_collect_joint")
    finally:
        GlobalHydra.instance().clear()

    assert cfg.runner.only_eval is True
    assert cfg.algorithm.loss_type == "embodied_sac"
    assert cfg.rollout.collect_transitions is False
    assert cfg.env.eval.use_master_takeover is True
    assert cfg.env.eval.master_takeover.control_mode == "joint"
    assert cfg.env.eval.master_takeover.max_joint_age_s == 0.25
    assert cfg.env.eval.data_collection.export_format == "pickle"
    assert cfg.env.eval.data_collection.only_success is False
    assert cfg.env.eval.data_collection.only_intervened is False
    assert cfg.env.eval.override_cfg.follower_joint_cmd_left_topic == "/follow_joint_control_1"
    assert cfg.env.eval.override_cfg.follower_joint_cmd_right_topic == "/follow_joint_control_2"
    assert cfg.env.eval.override_cfg.max_joint_delta == 0.2
    assert cfg.actor.model.openpi.config_name == "fold_towel_s2s"
    assert cfg.actor.model.action_dim == 14


def test_embodied_dagger_rejects_joint_master_takeover(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "examples" / "embodiment" / "config"
    monkeypatch.setenv("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            cfg = compose(config_name="realworld_x1_fold_towel_dagger_openpi")
    finally:
        GlobalHydra.instance().clear()

    with open_dict(cfg):
        cfg.env.train.master_takeover.control_mode = "joint"

    from rlinf.config import validate_embodied_cfg

    with pytest.raises(ValueError, match="embodied_dagger"):
        validate_embodied_cfg(cfg)


def test_x1_deploy_config_rejects_single_arm_use_arm_ids():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from rlinf.envs.realworld.xsquare.tasks.deploy_env import X1DeployEnvConfig

    with pytest.raises(ValueError, match=r"use_arm_ids=\[0, 1\]"):
        X1DeployEnvConfig(use_arm_ids=[1])


def test_x1_absolute_pose_step_returns_post_clip_executed_action():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from rlinf.envs.realworld.xsquare.tasks.deploy_env import X1DeployEnv

    env = X1DeployEnv(
        {
            "is_dummy": True,
            "use_arm_ids": [0, 1],
            "use_camera_ids": [2],
            "enforce_gripper_close": False,
            "ee_pose_limit_min": [[-0.1] * 6, [-0.2] * 6],
            "ee_pose_limit_max": [[0.1] * 6, [0.2] * 6],
            "gripper_width_limit_min": 0.0,
            "gripper_width_limit_max": 1.0,
        }
    )

    action = np.array([0.3] * 6 + [2.0] + [-0.3] * 6 + [-1.0], dtype=np.float32)
    _, _, _, _, info = env.step_absolute_pose(action)

    assert info["action_clipped"] is True
    assert info["clip_delta_max"] > 0.0
    np.testing.assert_array_equal(info["raw_action"], action)
    np.testing.assert_allclose(info["executed_action"][:6], np.full(6, 0.1))
    assert info["executed_action"][6] == 1.0
    np.testing.assert_allclose(info["executed_action"][7:13], np.full(6, -0.2))
    assert info["executed_action"][13] == 0.0


def test_x1_joint_step_clips_limits_and_delta():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from rlinf.envs.realworld.xsquare.tasks.deploy_env import X1DeployEnv

    env = X1DeployEnv(
        {
            "is_dummy": True,
            "use_arm_ids": [0, 1],
            "use_camera_ids": [2],
            "enforce_gripper_close": False,
            "joint_limit_min": [[-1.0] * 7, [-2.0] * 7],
            "joint_limit_max": [[1.0] * 7, [2.0] * 7],
            "max_joint_delta": 0.25,
        }
    )
    env._x1_state.follow1_joints = np.zeros(7, dtype=np.float32)
    env._x1_state.follow2_joints = np.ones(7, dtype=np.float32)

    action = np.array([2.0] * 7 + [-2.0] * 7, dtype=np.float32)
    _, _, _, _, info = env.step_joint(action)

    assert info["joint_action_clipped"] is True
    assert info["joint_clip_delta_max"] > 0.0
    np.testing.assert_array_equal(info["raw_joint_action"], action)
    np.testing.assert_allclose(info["executed_joint_action"][:7], np.full(7, 0.25))
    np.testing.assert_allclose(info["executed_joint_action"][7:], np.full(7, 0.75))
    np.testing.assert_allclose(env._x1_state.follow1_joints, np.full(7, 0.25))
    np.testing.assert_allclose(env._x1_state.follow2_joints, np.full(7, 0.75))
