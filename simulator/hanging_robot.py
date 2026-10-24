"""
The hanging two-module arm with its soft gripper: loading, actuation, handles.

Seven pressure lines in total:

    arm      pressure[0:6]  A1 A2 A3 B1 B2 B3  -> 30 segment actuators
    gripper  one line, 0..gripper.P_MAX bar    -> 12 finger hinge actuators

The arm part reuses robot.TwoModulePressureMapper unchanged. The gripper
pressure is held on the mapper itself (`mapper.gripper_pressure`), so every
call to `mapper.convert(model, arm_pressure)` - including the look-ahead
rollouts inside robot.JacobianCorrector - reproduces the current grip. The
Jacobian therefore sees the arm as it really is: holding an object or not.

Four objects live in the scene (cube, apple, banana, ball). One is ACTIVE -
on the table - and the others are parked on the floor off-camera;
Scene.set_layout() chooses which, and where.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent     # the simulator root
# This layout is flat: configs/, sim/, viz/ and every simulator module live
# side by side under simulator/, so one entry on sys.path resolves them all.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gripper as G  # noqa: E402
from build_hanging_scene import OBJECTS, PARKING  # noqa: E402
from robot import NUM_ACTUATORS, NUM_CHAMBERS, TwoModulePressureMapper  # noqa: E402
from sim.mujoco_env import stabilize_model  # noqa: E402

SCENE_PATH = HERE / "scene_hanging_gripper.xml"

from build_hanging_scene import OBJECT_START as OBJECT_START_XY  # noqa: E402


class HangingGripperMapper(TwoModulePressureMapper):
    """6 arm pressures -> 30 arm setpoints, plus the gripper line -> 12."""

    def __init__(self, model):
        super().__init__()
        self.gripper_pressure = 0.0
        self.gripper_actuators = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                              f"act_{name}")
            for name in G.joint_names()])
        if (self.gripper_actuators < 0).any():
            raise RuntimeError("scene is missing gripper actuators")

    def convert(self, model, pressure):
        ctrl = super().convert(model, pressure)
        ctrl[self.gripper_actuators] = G.finger_targets(self.gripper_pressure)
        return ctrl


class Scene:
    """Model, data, mapper and the handles every task script needs."""

    # Which object bodies this scene is expected to contain, in the order the
    # scene builder laid them out (the parking slot is derived from that
    # order). A subclass whose XML holds more objects overrides it; see
    # continual_scene.ContinualScene.
    OBJECT_SPECS = OBJECTS

    def __init__(self, control_dt=0.01, scene_path=SCENE_PATH):
        if not Path(scene_path).exists():
            raise FileNotFoundError(
                f"{scene_path} not found - run build_hanging_scene.py")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        expected = NUM_ACTUATORS + len(G.joint_names())
        if self.model.nu != expected:
            raise RuntimeError(f"expected {expected} actuators, "
                               f"found {self.model.nu}")
        stabilize_model(self.model)

        self.data = mujoco.MjData(self.model)
        self.mapper = HangingGripperMapper(self.model)

        name = lambda kind, n: mujoco.mj_name2id(self.model, kind, n)  # noqa
        self.tcp_id = name(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.place_site = name(mujoco.mjtObj.mjOBJ_SITE, "place_target")
        self.table_geom = name(mujoco.mjtObj.mjOBJ_GEOM, "table")

        self.objects = {}
        for k, obj in enumerate(self.OBJECT_SPECS):
            body = name(mujoco.mjtObj.mjOBJ_BODY, f"obj_{obj}")
            if body < 0:
                raise RuntimeError(
                    f"{Path(scene_path).name} has no body obj_{obj} - the "
                    f"scene and {type(self).__name__}.OBJECT_SPECS disagree; "
                    "rebuild the scene")
            self.objects[obj] = {
                "body": body,
                "geoms": {g for g in range(self.model.ngeom)
                          if self.model.geom_bodyid[g] == body
                          and self.model.geom_contype[g]},
                "qposadr": int(self.model.jnt_qposadr[
                    self.model.body_jntadr[body]]),
                "parking": (PARKING[0] + 0.15 * k, PARKING[1]),
                **self.OBJECT_SPECS[obj],
            }
        self.active = "cube"
        self.finger_geoms = {
            g for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                or "").startswith("finger")}

        self.substeps = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.control_dt = self.substeps * self.model.opt.timestep

        self.table_top = float(
            self.model.geom_pos[self.table_geom][2]
            + self.model.geom_size[self.table_geom][2])

    # -- state ---------------------------------------------------------------

    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()

    @property
    def obj(self):
        """Spec + handles of the active object."""
        return self.objects[self.active]

    def object_pos(self):
        """Centre of mass of the active object. For the symmetric objects this
        is the body frame; for the curved banana it sits ~6 mm inside the
        curve, and gripping there - not at the middle of the arc - is what
        keeps the banana level in the hand."""
        return self.data.xipos[self.obj["body"]].copy()

    def object_upright(self):
        """Cosine between the object's z axis and world z (1 = as placed)."""
        return float(self.data.xmat[self.obj["body"]].reshape(3, 3)[2, 2])

    def rest_z(self):
        """Height of the active object's centre of mass when it lies on the
        table (measured after the reset settles it)."""
        return self._rest_com_z

    def place_target(self):
        return self.data.site_xpos[self.place_site].copy()

    def finger_contacts(self):
        """Number of finger geoms currently touching the active object."""
        mine = self.obj["geoms"]
        touching = set()
        for c in self.data.contact[:self.data.ncon]:
            if c.geom1 in mine and c.geom2 in self.finger_geoms:
                touching.add(c.geom2)
            elif c.geom2 in mine and c.geom1 in self.finger_geoms:
                touching.add(c.geom1)
        return len(touching)

    def object_on_table(self):
        """True while the active object is touching the table."""
        mine = self.obj["geoms"]
        for c in self.data.contact[:self.data.ncon]:
            if (c.geom1 in mine and c.geom2 == self.table_geom) or \
                    (c.geom2 in mine and c.geom1 == self.table_geom):
                return True
        return False

    # -- actuation -----------------------------------------------------------

    def apply(self, arm_pressure, gripper_pressure):
        self.mapper.gripper_pressure = float(gripper_pressure)
        self.data.ctrl[:] = self.mapper.convert(self.model, arm_pressure)

    def step(self):
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def set_layout(self, obj=None, object_xy=None, place_xy=None):
        """Choose the active object and where it starts on the table, and/or
        move the place target. The other objects go to the floor, off-camera.
        Takes effect at the next reset."""

        if obj is not None:
            if obj not in self.objects:
                raise ValueError(f"unknown object {obj!r}; "
                                 f"choose from {list(self.objects)}")
            self.active = obj
        xy = OBJECT_START_XY if object_xy is None else object_xy

        for name, o in self.objects.items():
            adr = o["qposadr"]
            if name == self.active:
                self.model.qpos0[adr:adr + 3] = [*xy, self.table_top
                                                 + o["rest"] + 1e-4]
            else:
                self.model.qpos0[adr:adr + 3] = [*o["parking"],
                                                 o["rest"] + 1e-4]
            self.model.qpos0[adr + 3:adr + 7] = [1, 0, 0, 0]

        if place_xy is not None:
            self.model.site_pos[self.place_site][:2] = place_xy

    def reset(self, arm_pressure, gripper_pressure=0.0, settle_s=1.5):
        mujoco.mj_resetData(self.model, self.data)
        self.apply(arm_pressure, gripper_pressure)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(int(settle_s / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self._rest_com_z = float(self.object_pos()[2])


def balanced_pressure(level):
    """Arm pressure that extends both modules straight down (no bend):
    each chamber scaled by 1/strain-gain so the three bend terms cancel."""

    from configs.parameters import STRAIN_GAIN

    comp = STRAIN_GAIN.min() / np.asarray(STRAIN_GAIN, dtype=np.float64)
    return np.tile(level * comp, NUM_CHAMBERS // 3)
