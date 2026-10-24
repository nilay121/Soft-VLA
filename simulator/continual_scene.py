"""
The bowl-drop scene with the can in it, for the continual experiment.

`ContinualScene` is `BowlScene` pointed at scene_continual.xml and told to
expect five objects instead of four. Nothing else differs: the same seven
pressure lines, the same bowl, the same success test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_bowl_scene as BS  # noqa: E402
import build_continual_scene as CS  # noqa: E402
import build_hanging_scene as B  # noqa: E402
from bowl_scene import (  # noqa: E402
    BOWL_DEPTH,
    BOWL_FLOOR_Z,
    BOWL_R_RIM,
    BOWL_RIM_Z,
    BowlScene,
    balanced_pressure,
)

SCENE_PATH = CS.OUTPUT


class ContinualScene(BowlScene):
    OBJECT_SPECS = CS.OBJECTS

    def __init__(self, control_dt=0.01, scene_path=SCENE_PATH):
        super().__init__(control_dt, scene_path)

    def object_pos_named(self, name):
        return self.data.xipos[self.objects[name]["body"]].copy()

    def in_bowl(self, name):
        """Is THIS object in the bowl? (`object_in_bowl` only sees the
        active one, which is not enough when three are on the table.)"""

        centre = self.bowl_centre()
        pos = self.object_pos_named(name)
        radial = float(np.linalg.norm(pos[:2] - centre[:2]))
        return bool(radial < BOWL_R_RIM and pos[2] < centre[2] - 0.002)

    def objects_in_bowl(self, names=None):
        return {n: self.in_bowl(n) for n in (names or self.objects)}

    def set_multi_layout(self, layout, bowl_xy=None, active=None):
        """Put SEVERAL objects on the table at once.

        `layout` maps object name -> (x, y). Anything not in it is parked on
        the floor, as usual. `active` is the one the success test follows -
        the one the instruction names.
        """

        if active is None:
            active = next(iter(layout))
        if active not in self.objects:
            raise ValueError(f"unknown object {active!r}")
        self.active = active

        for name, o in self.objects.items():
            adr = o["qposadr"]
            if name in layout:
                xy = layout[name]
                self.model.qpos0[adr:adr + 3] = [*xy, self.table_top
                                                 + o["rest"] + 1e-4]
            else:
                self.model.qpos0[adr:adr + 3] = [*o["parking"],
                                                 o["rest"] + 1e-4]
            self.model.qpos0[adr + 3:adr + 7] = [1, 0, 0, 0]

        if bowl_xy is not None:
            self.model.body_pos[self.bowl_body][:2] = bowl_xy


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
#
# The old sampler drew from one fixed sector for every object. That stopped
# working once the objects stopped being the same size: a 44 mm apple centred
# at the sector's inner corner hangs over the edge of the table's hole and
# falls in before the arm has moved. Clearance is therefore checked per
# object, against its own footprint.

FOOTPRINT = B.FOOTPRINT
R_RANGE = (0.032, 0.074)                 # m, reachable band on the table
ANGLE_RANGE_DEG = (115.0, 245.0)         # the left slab, widened for clutter
HOLE_EDGE_X = BS.HOLE_CENTRE[0] - BS.HOLE_HALF[0]     # where the table stops
HOLE_MARGIN = 0.008                      # m, keep this much of table under it
MIN_GAP = 0.016                          # m, edge-to-edge between neighbours
# Up to four objects (cube, apple, can + the small cup: 40/40 layouts) keep
# 16 mm. Five do not fit at 16 mm (0 of 60 layouts); 8 mm fits 97%, but the
# open fingertips (~45 mm radius) then land on neighbours and the expert's
# success falls to ~40% on the can, so the experiment uses four.
MIN_GAP_CROWDED = 0.008
# CONTINUAL_LAYOUT=spread: keep the objects as far apart as the reachable
# band allows. 40 mm edge-to-edge clears the open fingertips (~45 mm from
# the target's axis) of every neighbour; three objects fit at up to ~44 mm.
# Draws that cannot reach 40 mm relax in 2 mm steps (see
# sample_multi_layout), so each layout is as spread as that draw permits.
import os as _os
SPREAD = _os.environ.get("CONTINUAL_LAYOUT", "") == "spread"
MIN_GAP_SPREAD = 0.040


def footprint(name):
    return FOOTPRINT.get(name, 0.020)


def _ok(xy, half, placed, gap=MIN_GAP):
    r = float(np.hypot(*xy))
    if not (R_RANGE[0] <= r <= R_RANGE[1]):
        return False
    if xy[0] + half > HOLE_EDGE_X - HOLE_MARGIN:
        return False
    for other_xy, other_half in placed:
        if np.hypot(*(xy - other_xy)) < half + other_half + gap:
            return False
    return True


def sample_object_xy(rng, name, placed=(), tries=600, gap=MIN_GAP):
    """One object's start, clear of the hole and of anything already placed."""

    half = footprint(name)
    lo, hi = np.radians(ANGLE_RANGE_DEG)
    for _ in range(tries):
        r = float(rng.uniform(*R_RANGE))
        a = float(rng.uniform(lo, hi))
        xy = np.array([r * np.cos(a), r * np.sin(a)])
        if _ok(xy, half, placed, gap):
            return xy
    return None


def sample_bowl_xy(rng):
    return np.array(BS.HOLE_CENTRE) + rng.uniform(-BS.BOWL_JITTER,
                                                  BS.BOWL_JITTER, size=2)


def sample_layout(rng, obj="cube"):
    """(object_xy, bowl_xy) for one object - the single-task case."""

    xy = sample_object_xy(rng, obj)
    if xy is None:
        raise RuntimeError(f"no valid start found for {obj!r}")
    return xy, sample_bowl_xy(rng)


def sample_multi_layout(rng, names, attempts=200):
    """({name: xy}, bowl_xy) with every object on the table at once.

    Placed biggest first - a 44 mm apple has far fewer legal positions than
    an 22 mm can, so placing it last is how a layout fails to come together.

    If no layout is found at the nominal gap, the gap is relaxed in 2 mm
    steps down to MIN_GAP_CROWDED rather than failing. Only draws that used
    to raise are affected; every layout that succeeded before is unchanged.
    """

    order = sorted(names, key=footprint, reverse=True)
    nominal = MIN_GAP if len(names) <= 4 else MIN_GAP_CROWDED
    if SPREAD and len(names) <= 4:
        nominal = MIN_GAP_SPREAD
    gaps = [nominal]
    while gaps[-1] - 0.002 >= MIN_GAP_CROWDED - 1e-9:
        gaps.append(round(gaps[-1] - 0.002, 4))
    for gap in gaps:
        for _ in range(attempts):
            placed, layout = [], {}
            for name in order:
                xy = sample_object_xy(rng, name, placed, gap=gap)
                if xy is None:
                    break
                layout[name] = xy
                placed.append((xy, footprint(name)))
            if len(layout) == len(order):
                return {n: layout[n] for n in names}, sample_bowl_xy(rng)
    raise RuntimeError(f"could not place {list(names)} without overlap - "
                       "loosen MIN_GAP or the angle range")


__all__ = ["ContinualScene", "balanced_pressure", "sample_layout",
           "sample_multi_layout", "sample_object_xy", "sample_bowl_xy",
           "footprint", "SCENE_PATH", "BOWL_RIM_Z", "BOWL_FLOOR_Z",
           "BOWL_R_RIM", "BOWL_DEPTH"]
