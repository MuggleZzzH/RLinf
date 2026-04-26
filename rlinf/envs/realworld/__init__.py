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

from .realworld_env import RealWorldEnv

_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "DualFrankaEnv": ("rlinf.envs.realworld.franka.dual_franka_env", "DualFrankaEnv"),
    "DualFrankaRobotConfig": (
        "rlinf.envs.realworld.franka.dual_franka_env",
        "DualFrankaRobotConfig",
    ),
    "DOSW1Config": ("rlinf.envs.realworld.dosw1", "DOSW1Config"),
    "DOSW1Env": ("rlinf.envs.realworld.dosw1", "DOSW1Env"),
    "dosw1_tasks": ("rlinf.envs.realworld.dosw1.tasks", None),
    "FrankaEnv": ("rlinf.envs.realworld.franka", "FrankaEnv"),
    "FrankaRobotConfig": ("rlinf.envs.realworld.franka", "FrankaRobotConfig"),
    "FrankaRobotState": ("rlinf.envs.realworld.franka", "FrankaRobotState"),
    "franka_tasks": ("rlinf.envs.realworld.franka.tasks", None),
    "X1Env": ("rlinf.envs.realworld.xsquare", "X1Env"),
    "X1RobotConfig": ("rlinf.envs.realworld.xsquare", "X1RobotConfig"),
    "X1RobotState": ("rlinf.envs.realworld.xsquare", "X1RobotState"),
    "xsquare_tasks": ("rlinf.envs.realworld.xsquare.tasks", None),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "DualFrankaEnv",
    "DualFrankaRobotConfig",
    "DOSW1Config",
    "DOSW1Env",
    "dosw1_tasks",
    "FrankaEnv",
    "FrankaRobotConfig",
    "FrankaRobotState",
    "franka_tasks",
    "X1Env",
    "X1RobotConfig",
    "X1RobotState",
    "xsquare_tasks",
    "RealWorldEnv",
]
