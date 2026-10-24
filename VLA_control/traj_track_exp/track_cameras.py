"""
The two views the trajectory-tracking policy sees.

WHY TWO EXTERNAL VIEWS, AND NOT THE WRIST CAMERA
------------------------------------------------
The pick-and-place policy used an external view plus an eye-in-hand camera,
because the thing it had to find - a cube - was small, and at the moment of
the grasp only the wrist could see it. Here there is no object. What the
policy has to read out of the image is the ARM'S OWN CONFIGURATION, and a
camera bolted to the end effector cannot see the arm it is attached to: it
would stare at empty space and return almost the same picture wherever the
arm was.

So both views are external, placed 90 degrees apart around the arm. That
separation is the point. One side view collapses the axis pointing away
from it, so tool-point motion along that axis is nearly invisible; a second
view at 90 degrees measures exactly the axis the first one loses. Together
they pin down all three coordinates, which is what a 3-D trajectory needs.

Both look slightly downward at the lower half of the arm, framed so the
ceiling mount stays in shot: with fixed cameras those static landmarks are
what turn a pixel position into a world position.

`camera_check.py` measures how far the tool point actually travels in pixels
across the workspace under these poses, and whether the arm ever leaves the
frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from viz.common import free_camera  # noqa: E402

LOOKAT_Z = 0.475        # m, centred on the lower half of the arm
DISTANCE = 0.38          # m, close enough that the tool point travels
ELEVATION = -12.0        # deg, slightly above the arm looking down

VIEWS = [
    {"name": "side_a", "lookat": [0.0, 0.0, LOOKAT_Z], "distance": DISTANCE,
     "azimuth": 215.0, "elevation": ELEVATION},
    {"name": "side_b", "lookat": [0.0, 0.0, LOOKAT_Z], "distance": DISTANCE,
     "azimuth": 305.0, "elevation": ELEVATION},   # +90 deg from side_a
]


def policy_cameras(scene, model):
    """([camera, camera], [cfg, cfg]) - the two external views, in order."""

    cameras, cfgs = [], []
    for v in VIEWS:
        cfg = {k: v[k] for k in ("lookat", "distance", "azimuth", "elevation")}
        cameras.append(free_camera(model, **cfg))
        cfgs.append({"kind": "external", "name": v["name"], **cfg})
    return cameras, cfgs


__all__ = ["policy_cameras", "VIEWS"]
