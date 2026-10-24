"""
The two views the bowl-drop policy sees, and nothing else.

VIEW 1 - EXTERNAL, chosen by measurement
----------------------------------------
`camera_study.py` renders a segmentation buffer with and without the arm at
the scored phases of real episodes, so "how much of the object is the arm
hiding" is an exact number rather than a judgement. The pose below is that
sweep's winner; re-run the study and update EXTERNAL if the scene changes.

VIEW 2 - WRIST
--------------
Bolted to `gripper_palm`, where the arm meets the gripper, looking straight
down the tool axis (see build_bowl_scene.add_wrist_camera). It sees what no
fixed camera can: the object between the fingers at the moment of the
grasp, and the bowl mouth directly below at the moment of the drop. It is
defined in the XML, so it is referenced here by name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # this experiment
SIMULATOR = HERE.parent.parent / "simulator"      # ../../simulator
# HERE goes in last so it wins: the experiment's own modules must never be
# shadowed by a same-named module in the simulator.
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_bowl_scene as BS  # noqa: E402
from viz.common import free_camera  # noqa: E402

STUDY_PATH = HERE / "results" / "camera_study.json"

# --- filled in from camera_study.py (results/camera_study.json) -------------
EXTERNAL = {
    "lookat": [0.5 * (BS.HOLE_CENTRE[0] + BS.default_object_xy()[0]), 0.0,
               0.372 + 0.045],
    "distance": 0.40,
    "azimuth": 225.0,
    "elevation": -35.0,
}
# camera_study.py, 24 azimuths x 5 elevations x 3 distances over 16 arm
# poses from 4 layouts: this pose hides none of the object at any scored
# phase and resolves it and the bowl over more pixels than the other poses
# that also hide none (350 of 360 candidates tie at zero occlusion, so
# apparent size is what actually decides).

WRIST_CAMERA = "wrist_camera"


def policy_cameras(scene, model):
    """([camera, camera], [cfg, cfg]) - the external view then the wrist.

    The first is a free camera built from EXTERNAL; the second is the fixed
    XML camera on the palm, passed to the renderer by name.
    """

    external = dict(EXTERNAL)
    external["lookat"] = [external["lookat"][0], external["lookat"][1],
                          scene.table_top + 0.045]

    cameras = [free_camera(model, **external), WRIST_CAMERA]
    cfgs = [{"kind": "external", "chosen_by": "camera_study.py", **external},
            {"kind": "wrist", "camera": WRIST_CAMERA, "body": "gripper_palm",
             "fovy_deg": BS.WRIST_FOVY,
             "note": "eye-in-hand, looking down the tool axis"}]
    return cameras, cfgs


def study_winner():
    """The camera study's best pose, if it has been run."""

    if not STUDY_PATH.exists():
        return None
    return json.loads(STUDY_PATH.read_text())["best"]


__all__ = ["policy_cameras", "EXTERNAL", "WRIST_CAMERA", "study_winner"]
