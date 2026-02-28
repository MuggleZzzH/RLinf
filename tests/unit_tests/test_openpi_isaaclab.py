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

import numpy as np
import pytest

pytest.importorskip("openpi")

from openpi.models import model as _model

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.isaaclab_policy import (
    IsaaclabInputs,
    IsaaclabOutputs,
)


def test_get_openpi_config_pi0_isaaclab_registered():
    cfg = get_openpi_config("pi0_isaaclab")
    assert cfg.name == "pi0_isaaclab"
    assert cfg.model.action_horizon == 10
    assert cfg.data.__class__.__name__ == "LeRobotIsaaclabDataConfig"


@pytest.mark.parametrize("state_dim", [7, 8])
def test_isaaclab_inputs_supports_state_7_or_8(state_dim: int):
    transform = IsaaclabInputs(model_type=_model.ModelType.PI0)
    output = transform(
        {
            "observation/image": np.random.rand(3, 84, 84).astype(np.float32),
            "observation/wrist_image": np.random.rand(84, 84, 3).astype(np.float32),
            "observation/state": np.zeros((state_dim,), dtype=np.float32),
            "actions": np.zeros((10, 7), dtype=np.float32),
            "prompt": "stack cubes",
        }
    )

    assert output["state"].shape == (state_dim,)
    assert output["image"]["base_0_rgb"].shape == (84, 84, 3)
    assert output["image"]["base_0_rgb"].dtype == np.uint8
    assert output["image"]["left_wrist_0_rgb"].shape == (84, 84, 3)
    assert bool(output["image_mask"]["left_wrist_0_rgb"]) is True
    assert bool(output["image_mask"]["right_wrist_0_rgb"]) is False
    assert output["actions"].shape == (10, 7)


def test_isaaclab_inputs_handles_missing_wrist_image():
    transform = IsaaclabInputs(model_type=_model.ModelType.PI0)
    output = transform(
        {
            "observation/image": np.random.randint(
                0, 255, size=(84, 84, 3), dtype=np.uint8
            ),
            "observation/state": np.zeros((7,), dtype=np.float32),
            "prompt": "stack cubes",
        }
    )

    assert np.array_equal(
        output["image"]["left_wrist_0_rgb"],
        np.zeros_like(output["image"]["base_0_rgb"]),
    )
    assert bool(output["image_mask"]["left_wrist_0_rgb"]) is False


def test_isaaclab_outputs_trims_to_7d_actions():
    transform = IsaaclabOutputs()
    output = transform({"actions": np.random.rand(10, 32).astype(np.float32)})
    assert output["actions"].shape == (10, 7)
