"""
The hanging arm with its gripper held closed, for trajectory tracking.

Deliberately leaner than `hanging_robot.Scene`: there are no objects, no
table and no place target in this scene, so none of that bookkeeping exists
here. What remains is the seven pressure lines, the tool point, and the
stepping.

    arm      pressure[0:6]  A1 A2 A3 B1 B2 B3  -> 30 segment actuators
    gripper  one line, pinned at GRIP_CLOSED_BAR for the whole experiment

The gripper pressure still goes through the mapper, so the look-ahead
rollouts inside robot.JacobianCorrector reproduce the closed fist and the
Jacobian sees the arm as it really is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent     # the simulator root
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gripper as G  # noqa: E402
from build_track_scene import GRIP_CLOSED_BAR, MOUNT_HEIGHT  # noqa: E402
from hanging_robot import HangingGripperMapper, balanced_pressure  # noqa: E402
from robot import NUM_ACTUATORS  # noqa: E402
from sim.mujoco_env import stabilize_model  # noqa: E402

SCENE_PATH = HERE / "scene_tracking.xml"


class TrackScene:
    """Model, data, mapper and the handles the tracking scripts need."""

    def __init__(self, control_dt=0.01, scene_path=SCENE_PATH,
                 grip=GRIP_CLOSED_BAR):
        if not Path(scene_path).exists():
            raise FileNotFoundError(
                f"{scene_path} not found - run build_track_scene.py")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        expected = NUM_ACTUATORS + len(G.joint_names())
        if self.model.nu != expected:
            raise RuntimeError(f"expected {expected} actuators, "
                               f"found {self.model.nu}")
        stabilize_model(self.model)

        self.data = mujoco.MjData(self.model)
        self.mapper = HangingGripperMapper(self.model)
        self.grip = float(grip)

        # `tool_tip`, not `tcp`: with the fist closed the fingertips meet
        # 18.2 mm below the grasp TCP, and it is the visible tip that should
        # follow the reference. See build_track_scene.TOOL_TIP_OFFSET.
        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                        "tool_tip")
        if self.tcp_id < 0:
            raise RuntimeError("scene is missing the tool_tip site - "
                               "re-run build_track_scene.py")

        self.substeps = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.control_dt = self.substeps * self.model.opt.timestep

    # -- state ---------------------------------------------------------------

    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()

    # -- actuation -----------------------------------------------------------

    def apply(self, arm_pressure, gripper_pressure=None):
        """The gripper defaults to the pinned closed pressure."""

        self.mapper.gripper_pressure = (self.grip if gripper_pressure is None
                                        else float(gripper_pressure))
        self.data.ctrl[:] = self.mapper.convert(self.model, arm_pressure)

    def step(self):
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def reset(self, arm_pressure, settle_s=1.5):
        mujoco.mj_resetData(self.model, self.data)
        self.apply(arm_pressure)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(settle_s / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)


__all__ = ["TrackScene", "balanced_pressure", "SCENE_PATH",
           "GRIP_CLOSED_BAR", "MOUNT_HEIGHT"]
