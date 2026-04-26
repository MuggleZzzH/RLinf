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

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


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
    assert cfg.env.train.data_collection.save_dir == "../results/x1_dagger_rollouts"
    assert cfg.env.train.data_collection.export_format == "lerobot"
    assert cfg.env.train.data_collection.only_success is True
    assert cfg.env.train.data_collection.only_intervened is False
    assert cfg.env.eval.use_master_takeover is False


def test_x1_deploy_config_rejects_single_arm_use_arm_ids():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from rlinf.envs.realworld.xsquare.tasks.deploy_env import X1DeployEnvConfig

    with pytest.raises(ValueError, match=r"use_arm_ids=\[0, 1\]"):
        X1DeployEnvConfig(use_arm_ids=[1])
