"""
The hanging arm + soft gripper with a bowl sunk into the table.

`BowlScene` is `hanging_robot.Scene` with three additions:

  * the bowl is a movable static body, so `set_layout()` can put it anywhere
    inside the table's hole;
  * `object_in_bowl()` replaces the old "resting within 15 mm of a marker"
    test - the object counts as delivered once it is below the rim and
    inside the rim radius;
  * the table is four slabs around the hole rather than one box, so
    `object_on_table()` checks all of them.

Everything else - the 7 pressure lines, the mapper, the objects - is
inherited unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent     # the simulator root
# This layout is flat: configs/, sim/, viz/ and every simulator module live
# side by side under simulator/, so one entry on sys.path resolves them all.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_bowl_scene as BS  # noqa: E402
from hanging_robot import Scene, balanced_pressure  # noqa: E402

SCENE_PATH = HERE / "scene_bowl_drop.xml"

# re-exported so the task scripts do not all import build_bowl_scene
BOWL_RIM_Z = BS.BOWL_RIM_Z
BOWL_FLOOR_Z = BS.BOWL_FLOOR_Z
BOWL_R_RIM = BS.BOWL_R_RIM
BOWL_DEPTH = BS.BOWL_DEPTH


class BowlScene(Scene):
    def __init__(self, control_dt=0.01, scene_path=SCENE_PATH):
        super().__init__(control_dt, scene_path)

        self.bowl_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                           "bowl")
        if self.bowl_body < 0:
            raise RuntimeError("scene is missing the bowl body")
        self.bowl_geoms = {g for g in range(self.model.ngeom)
                           if self.model.geom_bodyid[g] == self.bowl_body
                           and self.model.geom_contype[g]}

        self.table_geoms = set()
        for name in ("table", "table_right", "table_front", "table_back"):
            g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if g >= 0:
                self.table_geoms.add(g)

    # -- the bowl ------------------------------------------------------------

    def bowl_centre(self):
        """World position of the centre of the bowl's rim."""
        return self.data.xpos[self.bowl_body].copy()

    def object_in_bowl(self):
        """True once the active object is inside the bowl.

        Inside means below the rim plane and within the rim radius - the
        object has fallen in, rather than come to rest perched on the rim or
        on the table beside it.
        """

        centre = self.bowl_centre()
        pos = self.object_pos()
        radial = float(np.linalg.norm(pos[:2] - centre[:2]))
        return bool(radial < BOWL_R_RIM and pos[2] < centre[2] - 0.002)

    def bowl_offset_mm(self):
        """Horizontal distance from the object to the bowl axis [mm] - the
        drop's accuracy, comparable with the old placement error."""

        centre = self.bowl_centre()
        return float(np.linalg.norm(self.object_pos()[:2] - centre[:2]) * 1000)

    def object_settled(self, speed=0.02):
        """True while the active object is nearly stationary [m/s]."""

        adr = self.model.body_dofadr[self.obj["body"]]
        return bool(np.linalg.norm(self.data.qvel[adr:adr + 3]) < speed)

    # -- overrides -----------------------------------------------------------

    def object_on_table(self):
        mine = self.obj["geoms"]
        for c in self.data.contact[:self.data.ncon]:
            if (c.geom1 in mine and c.geom2 in self.table_geoms) or \
                    (c.geom2 in mine and c.geom1 in self.table_geoms):
                return True
        return False

    def set_layout(self, obj=None, object_xy=None, bowl_xy=None,
                   place_xy=None):
        """As Scene.set_layout, but `bowl_xy` moves the whole bowl.

        `place_xy` is accepted as an alias so code written against the
        pick-and-place scene keeps working.
        """

        bowl_xy = bowl_xy if bowl_xy is not None else place_xy

        if obj is not None:
            if obj not in self.objects:
                raise ValueError(f"unknown object {obj!r}; "
                                 f"choose from {list(self.objects)}")
            self.active = obj
        xy = BS.default_object_xy() if object_xy is None else object_xy

        for name, o in self.objects.items():
            adr = o["qposadr"]
            if name == self.active:
                self.model.qpos0[adr:adr + 3] = [*xy, self.table_top
                                                 + o["rest"] + 1e-4]
            else:
                self.model.qpos0[adr:adr + 3] = [*o["parking"],
                                                 o["rest"] + 1e-4]
            self.model.qpos0[adr + 3:adr + 7] = [1, 0, 0, 0]

        if bowl_xy is not None:
            self.model.body_pos[self.bowl_body][:2] = bowl_xy
        # the place_target site rides on the bowl body, so nothing else moves


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

def sample_layout(rng):
    """Where the object starts and where the bowl sits, for one episode.

    The object is drawn from a sector of the reachable disc that lies on the
    left table slab; the bowl is jittered inside the hole. The two can never
    overlap, so no rejection loop is needed.
    """

    lo, hi = BS.OBJECT_RADIUS_RANGE
    a0, a1 = np.radians(BS.OBJECT_ANGLE_RANGE_DEG)

    r = float(rng.uniform(lo, hi))
    a = float(rng.uniform(a0, a1))
    object_xy = np.array([r * np.cos(a), r * np.sin(a)])

    bowl_xy = np.array(BS.HOLE_CENTRE) + rng.uniform(
        -BS.BOWL_JITTER, BS.BOWL_JITTER, size=2)

    return object_xy, bowl_xy


__all__ = ["BowlScene", "balanced_pressure", "sample_layout", "SCENE_PATH",
           "BOWL_RIM_Z", "BOWL_FLOOR_Z", "BOWL_R_RIM", "BOWL_DEPTH"]
