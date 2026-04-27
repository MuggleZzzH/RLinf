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

from __future__ import annotations

import sys
import types

import numpy as np


class FakeArmsController:
    def __init__(self):
        self.commands = []

    def arms_control(self, left, right):
        self.commands.append((list(left), list(right)))


def _install_controller_import_stubs(monkeypatch):
    rospy = types.ModuleType("rospy")
    rospy.loginfo_throttle = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object
    monkeypatch.setitem(sys.modules, "cv_bridge", cv_bridge)

    turtle2_basic = types.ModuleType("turtle2_basic")
    turtle2_controller_pkg = types.ModuleType("turtle2_basic.turtle2_controller")
    turtle2_controller_mod = types.ModuleType(
        "turtle2_basic.turtle2_controller.Turtle2Controller"
    )
    turtle2_controller_mod.Turtle2Controller = object
    monkeypatch.setitem(sys.modules, "turtle2_basic", turtle2_basic)
    monkeypatch.setitem(
        sys.modules, "turtle2_basic.turtle2_controller", turtle2_controller_pkg
    )
    monkeypatch.setitem(
        sys.modules,
        "turtle2_basic.turtle2_controller.Turtle2Controller",
        turtle2_controller_mod,
    )


def _load_controller_class(monkeypatch):
    _install_controller_import_stubs(monkeypatch)
    sys.modules.pop("rlinf.envs.realworld.xsquare.x1_smooth_controller", None)
    from rlinf.envs.realworld.xsquare.x1_smooth_controller import (
        X1SmoothController,
    )

    return X1SmoothController


def _make_controller(
    controller_cls,
    left_current,
    right_current,
    left_target,
    right_target,
    gripper_tolerance=0.05,
):
    controller = controller_cls.__new__(controller_cls)
    controller.controller = FakeArmsController()
    controller._state = types.SimpleNamespace(
        follow1_pos=np.asarray(left_current, dtype=np.float32),
        follow2_pos=np.asarray(right_current, dtype=np.float32),
    )
    controller.left_arm_target = list(left_target)
    controller.right_arm_target = list(right_target)
    controller.last_expected_xyz1 = None
    controller.last_expected_xyz2 = None
    controller.last_expected_rpy1 = None
    controller.last_expected_rpy2 = None
    controller.xyz_target_tolerance = 0.002
    controller.rpy_target_tolerance = 0.005
    controller.gripper_target_tolerance = gripper_tolerance
    controller.debug_pose_control = False
    controller.debug_gripper_control = False
    controller.xyz_speed = 0.5
    controller.rpy_speed = 1.5
    controller.freq = 50
    return controller


def test_smooth_controller_publishes_gripper_only_target(monkeypatch):
    controller_cls = _load_controller_class(monkeypatch)
    controller = _make_controller(
        controller_cls,
        left_current=[0, 0, 0, 0, 0, 0, 0.0],
        right_current=[0, 0, 0, 0, 0, 0, 0.0],
        left_target=[0, 0, 0, 0, 0, 0, 1.0],
        right_target=[0, 0, 0, 0, 0, 0, 2.0],
    )

    controller.smooth_action_callback(None)

    assert len(controller.controller.commands) == 1
    left_cmd, right_cmd = controller.controller.commands[0]
    np.testing.assert_allclose(left_cmd[:6], np.zeros(6))
    np.testing.assert_allclose(right_cmd[:6], np.zeros(6))
    assert left_cmd[6] == 1.0
    assert right_cmd[6] == 2.0


def test_smooth_controller_does_not_publish_when_pose_and_gripper_reached(
    monkeypatch,
):
    controller_cls = _load_controller_class(monkeypatch)
    controller = _make_controller(
        controller_cls,
        left_current=[0, 0, 0, 0, 0, 0, 1.0],
        right_current=[0, 0, 0, 0, 0, 0, 2.0],
        left_target=[0, 0, 0, 0, 0, 0, 1.01],
        right_target=[0, 0, 0, 0, 0, 0, 2.01],
    )

    controller.smooth_action_callback(None)

    assert controller.controller.commands == []


def test_smooth_controller_pose_tracking_keeps_target_gripper(monkeypatch):
    controller_cls = _load_controller_class(monkeypatch)
    controller = _make_controller(
        controller_cls,
        left_current=[0, 0, 0, 0, 0, 0, 0.0],
        right_current=[0, 0, 0, 0, 0, 0, 0.0],
        left_target=[0.2, 0, 0, 0, 0, 0, 1.0],
        right_target=[0.2, 0, 0, 0, 0, 0, 2.0],
    )

    controller.smooth_action_callback(None)

    assert len(controller.controller.commands) == 1
    left_cmd, right_cmd = controller.controller.commands[0]
    assert 0.0 < left_cmd[0] <= 0.2
    assert 0.0 < right_cmd[0] <= 0.2
    assert left_cmd[6] == 1.0
    assert right_cmd[6] == 2.0


def test_smooth_controller_uses_shortest_rpy_delta(monkeypatch):
    controller_cls = _load_controller_class(monkeypatch)

    delta = controller_cls._shortest_angle_delta(
        np.array([-3.13], dtype=np.float32),
        np.array([3.13], dtype=np.float32),
    )

    assert delta[0] < 0
    assert abs(delta[0]) < 0.1


def test_smooth_controller_circular_midpoint_does_not_jump_to_zero(monkeypatch):
    controller_cls = _load_controller_class(monkeypatch)

    midpoint = controller_cls._circular_midpoint(
        np.array([-3.13], dtype=np.float32),
        np.array([3.13], dtype=np.float32),
    )

    assert abs(abs(midpoint[0]) - np.pi) < 0.05
