#!/usr/bin/env python3
"""
Generate the hanging-arm scene for the BOWL DROP task.

    python 2_module_ISupport/ISupport_real_tasks/bowl_drop/build_bowl_scene.py

Writes scene_bowl_drop.xml next to this file.

Same arm, same gripper, same objects as ../build_hanging_scene.py - only the
goal changes. Instead of a flat green marker the object must be released
into a BOWL, so the task no longer needs a precise, guarded set-down: carry
the object over the bowl and vent the gripper.

WHY THE BOWL IS SUNK INTO THE TABLE
-----------------------------------
The hanging arm's tool point only travels between about 395 and 437 mm
(measured; see ../README.md). Carrying the object over a bowl rim needs

    TCP  >  rim + 15 mm (half the object) + 23 mm (open-fingertip drop)

so a rim more than ~25 mm above the 372 mm table top cannot be cleared at
all, and anything shallow enough to clear is a saucer rather than a bowl.
Cutting a hole in the table and hanging a 68 mm bowl in it puts the rim
6 mm above the table top, which leaves 19 mm of clearance at the top of the
reach band and still gives the object a real bowl to fall into.

The table is therefore built as four slabs around a rectangular hole. The
object starts on the left slab, the bowl hangs in the hole on the right,
and both stay inside the ~72 mm the arm can actually reach sideways.

Layout (top view, mm):

        y
        ^      +-------------------------+
        |      |        back strip       |
        |      +------+---------+--------+
        |      |      |         |        |
        |      | left | <hole>  | right  |
        |      | slab |  bowl   | slab   |
        |      |      |         |        |
        |      +------+---------+--------+
        |      |       front strip       |
        |      +-------------------------+
        +-------------------------------------> x
             object here            bowl here
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent     # the simulator root
# This layout is flat: configs/, sim/, viz/ and every simulator module live
# side by side under simulator/, so one entry on sys.path resolves them all.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_hanging_scene as B  # noqa: E402  (arm + gripper + objects)
import build_scene  # noqa: E402
import gripper as G  # noqa: E402

OUTPUT = HERE / "scene_bowl_drop.xml"

MOUNT_HEIGHT = B.MOUNT_HEIGHT        # 0.80 m
TABLE_TOP = B.TABLE_TOP              # 0.372 m
TABLE_HALF = (0.22, 0.22)
TABLE_THICK = 0.015

# --- the hole in the table, and the bowl that hangs in it -------------------

HOLE_CENTRE = (0.055, 0.000)         # m, on the +x side, still within reach
HOLE_HALF = (0.058, 0.058)           # m, big enough for the bowl to move in

BOWL_RIM_ABOVE_TABLE = 0.006         # m, how far the rim stands proud
BOWL_DEPTH = 0.068                   # m, rim to inner floor
BOWL_R_RIM = 0.045                   # m, inner radius at the rim
BOWL_R_BASE = 0.032                  # m, inner radius at the floor
BOWL_WALL = 0.004                    # m, wall thickness
# The floor is much thicker than the wall. The object arrives having fallen
# the full depth of the bowl at ~1.2 m/s; against the original 4 mm plate it
# penetrated far enough that the solver pushed it back out with MORE energy
# than it came in with - one deployment episode had the cube launched to
# 629 mm, well above the arm's 437 mm ceiling. Thickness is the whole fix,
# and it grows DOWNWARD so the inner surface the cameras see is unchanged.
#
# The contact stays SOFT (MuJoCo's critically damped default, as everywhere
# else in this scene). Stiffening it was tried and was actively harmful: a
# stiff bowl rebounds the object back out over the rim, and the policy's
# delivery rate fell from 18/20 to 11/20 with no other change.
BOWL_FLOOR_THICK = 0.012             # m
BOWL_SEGMENTS = 24                   # boxes making up the flared wall
BOWL_JITTER = 0.006                  # m, how far the bowl may be moved

BOWL_RIM_Z = TABLE_TOP + BOWL_RIM_ABOVE_TABLE          # 0.378 m
BOWL_FLOOR_Z = BOWL_RIM_Z - BOWL_DEPTH                 # 0.310 m
BOWL_OUTER_R = BOWL_R_RIM + BOWL_WALL                  # 0.049 m

BOWL_COLOUR = "0.93 0.94 0.91 1"     # off-white ceramic, reads against the
BOWL_RIM_COLOUR = "0.36 0.55 0.62 1"  # brown table and the blue cube

# Where the object may start: a sector of the reachable disc that lies on the
# left slab. Kept clear of the hole edge by the object's own half-width.
OBJECT_RADIUS_RANGE = (0.032, 0.070)
OBJECT_ANGLE_RANGE_DEG = (130.0, 230.0)


def fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def _quat_from_matrix(R):
    """[w x y z] from a 3x3 rotation matrix (MuJoCo's quaternion order)."""

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def add_bowl(world):
    """A flared, open-topped bowl as a static body.

    The body frame sits at the CENTRE OF THE RIM, so moving the body in x/y
    moves the whole bowl and `bowl_centre` reports where to aim. Local z=0 is
    the rim plane and the inner floor is at z = -BOWL_DEPTH.

    The wall is BOWL_SEGMENTS boxes around a circle, each tilted outward by
    the flare angle so the bowl opens towards the top and funnels a slightly
    off-centre object in rather than bouncing it out.
    """

    body = ET.SubElement(world, "body", name="bowl",
                         pos=fmt([*HOLE_CENTRE, BOWL_RIM_Z]))

    # inner floor: a disc just wide enough to carry the wall feet. Its TOP
    # surface stays at -BOWL_DEPTH whatever the thickness, so the bowl looks
    # the same from inside (see BOWL_FLOOR_THICK).
    ET.SubElement(body, "geom", name="bowl_floor", type="cylinder",
                  size=fmt([BOWL_R_BASE + BOWL_WALL,
                            0.5 * BOWL_FLOOR_THICK]),
                  pos=fmt([0, 0, -BOWL_DEPTH - 0.5 * BOWL_FLOOR_THICK]),
                  rgba=BOWL_COLOUR, contype="1", conaffinity="1", condim="4",
                  friction=B.FRICTION, solref=B.SOFT_SOLREF,
                  solimp=B.SOFT_SOLIMP, density="0", mass="0.05")

    flare = float(np.arctan2(BOWL_R_RIM - BOWL_R_BASE, BOWL_DEPTH))
    length = BOWL_DEPTH / np.cos(flare)           # slanted wall length
    r_mid = 0.5 * (BOWL_R_BASE + BOWL_R_RIM)
    chord = 2.0 * r_mid * np.sin(np.pi / BOWL_SEGMENTS)

    for i in range(BOWL_SEGMENTS):
        a = 2.0 * np.pi * i / BOWL_SEGMENTS
        # local frame: x outward (through the wall), y tangential, z up along
        # the slanted wall
        z_axis = np.array([np.sin(flare), 0.0, np.cos(flare)])
        y_axis = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(y_axis, z_axis)
        Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                       [np.sin(a), np.cos(a), 0.0],
                       [0.0, 0.0, 1.0]])
        R = Rz @ np.column_stack([x_axis, y_axis, z_axis])

        radius = r_mid + 0.5 * BOWL_WALL
        pos = [radius * np.cos(a), radius * np.sin(a), -0.5 * BOWL_DEPTH]

        ET.SubElement(body, "geom", name=f"bowl_wall{i}", type="box",
                      size=fmt([0.5 * BOWL_WALL, 0.55 * chord,
                                0.5 * length]),
                      pos=fmt(pos), quat=fmt(_quat_from_matrix(R)),
                      rgba=BOWL_COLOUR, contype="1", conaffinity="1",
                      condim="4", friction=B.FRICTION,
                      solref=B.SOFT_SOLREF, solimp=B.SOFT_SOLIMP,
                      density="0", mass="0.004")

    # a coloured rim ring, visual only, so the bowl mouth is unmistakable
    # from every camera angle
    for i in range(BOWL_SEGMENTS):
        a = 2.0 * np.pi * i / BOWL_SEGMENTS
        radius = BOWL_R_RIM + 0.5 * BOWL_WALL
        ET.SubElement(body, "geom", type="box",
                      size=fmt([0.5 * BOWL_WALL * 1.6,
                                0.55 * 2.0 * radius * np.sin(np.pi / BOWL_SEGMENTS),
                                0.0018]),
                      pos=fmt([radius * np.cos(a), radius * np.sin(a), 0.0]),
                      euler=f"0 0 {np.degrees(a):.6g}",
                      rgba=BOWL_RIM_COLOUR, contype="0", conaffinity="0",
                      density="0")

    # aim point for the controller and the success test
    ET.SubElement(body, "site", name="place_target", type="sphere",
                  size="0.004", pos="0 0 0", rgba="0.2 0.85 0.3 0")


def add_table(world):
    """Four slabs around a rectangular hole (see the module docstring)."""

    tx, ty = TABLE_HALF
    hx, hy = HOLE_CENTRE
    hhx, hhy = HOLE_HALF
    z = TABLE_TOP - 0.5 * TABLE_THICK
    half_z = 0.5 * TABLE_THICK

    x0, x1 = hx - hhx, hx + hhx            # hole extent in x
    y0, y1 = hy - hhy, hy + hhy

    # The left slab is the one the object starts and lands on, so it keeps
    # the plain name "table" that hanging_robot.Scene looks for.
    slabs = {
        # name:        (xmin, xmax, ymin, ymax)
        "table":       (-tx, x0, -ty, ty),
        "table_right": (x1, tx, -ty, ty),
        "table_front": (x0, x1, -ty, y0),
        "table_back":  (x0, x1, y1, ty),
    }
    for name, (a, b, c, d) in slabs.items():
        if b - a < 1e-6 or d - c < 1e-6:
            continue
        ET.SubElement(world, "geom", name=name, type="box",
                      size=fmt([0.5 * (b - a), 0.5 * (d - c), half_z]),
                      pos=fmt([0.5 * (a + b), 0.5 * (c + d), z]),
                      rgba="0.55 0.42 0.3 1", contype="1", conaffinity="1",
                      friction=B.FRICTION)

    for dx in (-1, 1):
        for dy in (-1, 1):
            ET.SubElement(world, "geom", type="box",
                          size=fmt([0.015, 0.015,
                                    0.5 * (TABLE_TOP - 0.03)]),
                          pos=fmt([dx * (tx - 0.03), dy * (ty - 0.03),
                                   0.5 * (TABLE_TOP - 0.03)]),
                          rgba="0.45 0.34 0.24 1", contype="0",
                          conaffinity="0")


def add_world(world, objects=None):
    top = MOUNT_HEIGHT

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

    add_table(world)
    add_bowl(world)

    # the objects: the cube on the table, the rest parked on the floor
    start = default_object_xy()
    for k, (name, spec) in enumerate((objects or B.OBJECTS).items()):
        on_table = name == "cube"
        pos = ([*start, TABLE_TOP + spec["rest"] + 1e-4] if on_table
               else [B.PARKING[0] + 0.15 * k, B.PARKING[1],
                     spec["rest"] + 1e-4])
        body = ET.SubElement(world, "body", name=f"obj_{name}", pos=fmt(pos))
        ET.SubElement(body, "freejoint", name=f"obj_{name}_free")
        B.OBJECT_BUILDERS[name](body, spec)

    ET.SubElement(world, "camera", name="task_camera",
                  pos=fmt([0.62, -0.62, 0.78]),
                  xyaxes="0.707 0.707 0 -0.39 0.39 0.834")


def default_object_xy():
    """Middle of the object sector: straight out along -x."""

    r = 0.5 * sum(OBJECT_RADIUS_RANGE)
    return (-r, 0.0)


def add_wrist_camera(palm):
    """Eye-in-hand camera on the wrist, where the arm meets the gripper.

    The palm's local +z runs down the tool axis (the arm is mounted upside
    down), so the camera sits just past the palm disc on that axis and looks
    straight down it: the three fingers frame the view and whatever is about
    to be grasped sits in the middle.

    MuJoCo cameras look along their own -z with +y up, so xyaxes="1 0 0 0 -1 0"
    gives camera -z = palm +z.
    """

    ET.SubElement(palm, "camera", name="wrist_camera",
                  pos=fmt([0.0, 0.0, 2 * G.PALM_HALF_THICKNESS + 0.001]),
                  xyaxes="1 0 0 0 -1 0", fovy=f"{WRIST_FOVY}")


WRIST_FOVY = 76.0


def build(objects=None, model_name="i_support_2module_bowl_drop"):
    root = ET.fromstring(build_scene.build().split("?>", 1)[1])

    root.set("model", model_name)
    root.find("compiler").set("meshdir", "sim/meshes")

    option = root.find("option")
    option.set("cone", "elliptic")
    option.set("impratio", "10")

    world = root.find("worldbody")

    base = world.find("body[@name='robot_base']")
    base.set("pos", fmt([0, 0, MOUNT_HEIGHT]))
    base.set("quat", "0 1 0 0")                   # 180 deg about X: hang down

    seg10 = base.find(".//body[@name='segment_10']")
    seg10.find("site[@name='end_effector']").set("rgba", "1 0 0 0")
    B.add_gripper(seg10)
    add_wrist_camera(seg10.find("body[@name='gripper_palm']"))

    add_world(world, objects)

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
              "    Two-module I-Support robot HANGING from a ceiling mount,\n"
              "    with a PneuNet-style soft gripper and a BOWL sunk into the\n"
              "    table: pick the object up and drop it in.\n"
              "    30 arm actuators (6 pressures) + 12 finger actuators "
              "(1 pressure).\n\n"
              "    GENERATED FILE - edit build_bowl_scene.py and re-run.\n"
              "-->\n")
    return header + ET.tostring(root, encoding="unicode") + "\n"


def main():
    OUTPUT.write_text(build())
    print(f"Wrote {OUTPUT}")
    print(f"  table top      {TABLE_TOP * 1000:.0f} mm, hole "
          f"{2000 * HOLE_HALF[0]:.0f}x{2000 * HOLE_HALF[1]:.0f} mm at "
          f"({HOLE_CENTRE[0] * 1000:.0f}, {HOLE_CENTRE[1] * 1000:.0f}) mm")
    print(f"  bowl rim       {BOWL_RIM_Z * 1000:.0f} mm "
          f"({BOWL_RIM_ABOVE_TABLE * 1000:.0f} mm above the table), "
          f"inner radius {BOWL_R_RIM * 1000:.0f} mm")
    print(f"  bowl floor     {BOWL_FLOOR_Z * 1000:.0f} mm "
          f"({BOWL_DEPTH * 1000:.0f} mm deep, "
          f"{BOWL_FLOOR_THICK * 1000:.0f} mm thick floor)")
    print(f"  object sector  r {OBJECT_RADIUS_RANGE[0] * 1000:.0f}-"
          f"{OBJECT_RADIUS_RANGE[1] * 1000:.0f} mm, "
          f"{OBJECT_ANGLE_RANGE_DEG[0]:.0f}-{OBJECT_ANGLE_RANGE_DEG[1]:.0f} deg")
    print(f"  cameras        task_camera (world) + wrist_camera "
          f"(on the palm, fovy {WRIST_FOVY:.0f})")


if __name__ == "__main__":
    main()
