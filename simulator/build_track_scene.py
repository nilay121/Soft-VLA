#!/usr/bin/env python3
"""
Generate the hanging-arm scene for TRAJECTORY TRACKING.

    python simulator/build_track_scene.py

Writes scene_tracking.xml next to this file.

The arm and the gripper are exactly the ones from build_hanging_scene.py -
same segments, joints, masses, 30 arm actuators and 12 finger actuators.
What is removed is everything the tracking task does not need:

    no table, no bowl, no objects, no place marker

so nothing occludes the arm and nothing constrains where it may go. The
gripper is held CLOSED for the whole experiment (one constant pressure on
its single line), which makes the end effector a compact fist rather than
three splayed fingers - easier to see, and it cannot catch on anything.

WHY THERE IS NO REFERENCE MARKER IN THE SCENE
---------------------------------------------
The policy is told which path to trace by being shown GOAL IMAGES: frames of
the robot at the configurations it should pass through. If the target were
also drawn in the scene as a marker, the policy could simply servo to the
marker and the goal images would carry no information - the experiment would
be measuring something else entirely.

So the reference is never part of the scene. The videos draw it at render
time through the recorder's `decorate` hook, which touches only the
human-facing camera, never the two the policy sees.

The remaining scenery - ceiling plate, beam, posts, floor - is static and is
there on purpose: with fixed cameras those are the landmarks that make a
pixel position mean a world position.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent     # the simulator root
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_hanging_scene as B  # noqa: E402  (arm + gripper construction)
import build_scene  # noqa: E402
import gripper as G  # noqa: E402

OUTPUT = HERE / "scene_tracking.xml"

MOUNT_HEIGHT = B.MOUNT_HEIGHT        # 0.80 m, same ceiling mount
FLOOR_Z = 0.0

# The gripper line is pinned here for the whole task (see the module
# docstring).
#
# NOT G.P_MAX. Each hinge tracks -26 deg per bar, four hinges per finger, so
# 1.0 bar is 104 deg of cumulative curl - the fingers sweep PAST closure and
# come back up, leaving the fingertips 21 mm ABOVE the tool point and splayed
# 36 mm out, with the PneuNet chambers fanned open on the outside of each
# bend. It looks inflated and half-open, which is what it is.
#
# Measured fingertip radius from the tool axis: 41 mm vented, 2.1 mm at
# 0.40 bar, 7 mm at 0.50, 36.5 mm at 1.00. 0.40 bar is where the tips
# actually meet - which is how gripper.TCP_OFFSET defines the tool point in
# the first place ("where the fingertips meet when closed on nothing").
#
# Pick-and-place never sees this because there the fingers stall on the cube
# long before they can over-curl.
GRIP_CLOSED_BAR = 0.40

# Where the closed fingertips actually meet, along the tool axis from the arm's
# end disc. gripper.TCP_OFFSET (0.062 m) is NOT this point: it was derived for
# the pick-and-place grasp pose, where the fingers are only partly curled
# around an object. With the fist fully closed at GRIP_CLOSED_BAR the tips
# converge 18.2 mm further down, at 0.0802 m (measured).
#
# Tracking the 0.062 m site made the arm trace the path with a point buried
# inside the gripper body, leaving the visible tip trailing below the
# reference. The scene therefore carries its own `tool_tip` site and
# TrackScene drives that instead. gripper.py and the pick-and-place scene are
# untouched.
TOOL_TIP_OFFSET = 0.0802


def fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def add_world(world):
    top = MOUNT_HEIGHT

    # ceiling plate + two-post frame: visual only, but they are the fixed
    # landmarks the cameras measure the arm against
    ET.SubElement(world, "geom", name="ceiling_plate", type="box",
                  size="0.09 0.09 0.012", pos=fmt([0, 0, top + 0.012]),
                  rgba="0.3 0.32 0.36 1", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", name="ceiling_beam", type="box",
                  size="0.42 0.035 0.02", pos=fmt([0, 0, top + 0.044]),
                  rgba="0.3 0.32 0.36 1", contype="0", conaffinity="0")
    for x in (-0.40, 0.40):
        ET.SubElement(world, "geom", type="box", size="0.02 0.035 0.43",
                      pos=fmt([x, 0, 0.5 * (top + 0.064)]),
                      rgba="0.3 0.32 0.36 1", contype="0", conaffinity="0")

    # a plain floor well below the workspace, for depth cues only
    ET.SubElement(world, "geom", name="track_floor", type="box",
                  size="0.9 0.9 0.01", pos=fmt([0, 0, FLOOR_Z - 0.01]),
                  rgba="0.35 0.37 0.40 1", contype="0", conaffinity="0")

    # a faint backdrop so the arm is never silhouetted against empty space
    ET.SubElement(world, "geom", name="backdrop", type="box",
                  size="0.01 0.9 0.6", pos=fmt([-0.55, 0, 0.45]),
                  rgba="0.46 0.48 0.52 1", contype="0", conaffinity="0")

    ET.SubElement(world, "camera", name="task_camera",
                  pos=fmt([0.62, -0.62, 0.62]),
                  xyaxes="0.707 0.707 0 -0.32 0.32 0.89")


def build():
    root = ET.fromstring(build_scene.build().split("?>", 1)[1])

    root.set("model", "i_support_2module_hanging_tracking")
    root.find("compiler").set("meshdir", "sim/meshes")

    option = root.find("option")
    option.set("cone", "elliptic")
    option.set("impratio", "10")

    world = root.find("worldbody")

    base = world.find("body[@name='robot_base']")
    base.set("pos", fmt([0, 0, MOUNT_HEIGHT]))
    base.set("quat", "0 1 0 0")                  # 180 deg about X: hang down

    seg10 = base.find(".//body[@name='segment_10']")
    seg10.find("site[@name='end_effector']").set("rgba", "1 0 0 0")
    B.add_gripper(seg10)

    # the point the controller actually drives: where the closed fist meets
    palm = seg10.find("body[@name='gripper_palm']")
    ET.SubElement(palm, "site", name="tool_tip", type="sphere", size="0.004",
                  pos=fmt([0, 0, TOOL_TIP_OFFSET]), rgba="1 0.85 0.1 1")
    palm.find("site[@name='tcp']").set("rgba", "1 0.85 0.1 0")   # hide the old one

    # nothing in this scene can be touched, so drop the gripper out of
    # collision entirely - it makes the rollouts noticeably cheaper
    for geom in seg10.iter("geom"):
        if (geom.get("name") or "").startswith(("finger", "palm")):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    add_world(world)

    act = root.find("actuator")
    for name in G.joint_names():
        ET.SubElement(act, "position", name=f"act_{name}", joint=name,
                      kp=f"{G.FINGER_KP}",
                      forcerange=fmt([-G.FINGER_FORCE_LIMIT,
                                      G.FINGER_FORCE_LIMIT]),
                      forcelimited="true")

    ET.indent(root, space="    ")
    header = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
              "<!--\n"
              "    Two-module I-Support robot HANGING from a ceiling mount\n"
              "    with a PneuNet soft gripper held CLOSED, for TRAJECTORY\n"
              "    TRACKING. No table, no objects, no target marker - the\n"
              "    reference is given to the policy as goal IMAGES, so it\n"
              "    must never be visible in the scene.\n"
              "    30 arm actuators (6 pressures) + 12 finger actuators "
              "(1 pressure).\n\n"
              "    GENERATED FILE - edit build_track_scene.py and re-run.\n"
              "-->\n")
    return header + ET.tostring(root, encoding="unicode") + "\n"


def main():
    OUTPUT.write_text(build())
    print(f"Wrote {OUTPUT}")
    print(f"  mount height {MOUNT_HEIGHT} m, floor {FLOOR_Z} m")
    print(f"  gripper held closed at {GRIP_CLOSED_BAR} bar, collision off")
    print(f"  tool point     tool_tip site at {TOOL_TIP_OFFSET * 1000:.1f} mm "
          f"(the closed fingertips), not the {G.TCP_OFFSET * 1000:.1f} mm "
          f"grasp TCP")
    print("  no table / bowl / objects / reference marker")


if __name__ == "__main__":
    main()
