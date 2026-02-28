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

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class IsaaclabInputs(transforms.DataTransformFn):
    """Convert IsaacLab observations to OpenPI model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        if (
            "observation/wrist_image" in data
            and data["observation/wrist_image"] is not None
        ):
            wrist_image = _parse_image(data["observation/wrist_image"])
            wrist_mask = np.True_
        else:
            wrist_image = np.zeros_like(base_image)
            wrist_mask = np.False_

        state = np.asarray(data["observation/state"])
        assert state.ndim == 1 and state.shape[0] in (7, 8), (
            f"Expected state shape (7,) or (8,), got {state.shape}"
        )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": wrist_mask,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class IsaaclabOutputs(transforms.DataTransformFn):
    """Convert OpenPI outputs to IsaacLab action format."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
