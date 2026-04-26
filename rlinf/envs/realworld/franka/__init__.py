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

from importlib import import_module
from typing import Any

_LAZY_ATTRS = {
    "FrankaEnv": ("rlinf.envs.realworld.franka.franka_env", "FrankaEnv"),
    "FrankaRobotConfig": (
        "rlinf.envs.realworld.franka.franka_env",
        "FrankaRobotConfig",
    ),
    "FrankaRobotState": (
        "rlinf.envs.realworld.franka.franka_robot_state",
        "FrankaRobotState",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = ["FrankaEnv", "FrankaRobotState", "FrankaRobotConfig"]
