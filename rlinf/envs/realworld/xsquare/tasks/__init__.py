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

from gymnasium.envs.registration import register

from rlinf.envs.realworld.xsquare.tasks.button_env import (
    ButtonEnv as ButtonEnv,
)
from rlinf.envs.realworld.xsquare.tasks.deploy_env import (
    Turtle2DeployEnv as Turtle2DeployEnv,
)

register(
    id="ButtonEnv-v1",
    entry_point="rlinf.envs.realworld.xsquare.tasks:ButtonEnv",
)

register(
    id="Turtle2DeployEnv-v1",
    entry_point="rlinf.envs.realworld.xsquare.tasks:Turtle2DeployEnv",
)
