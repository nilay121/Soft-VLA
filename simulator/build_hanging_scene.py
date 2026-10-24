#!/usr/bin/env python3
"""
Generate the hanging two-module I-Support scene with a soft gripper.

    python 2_module_ISupport/ISupport_real_tasks/build_hanging_scene.py

Writes scene_hanging_gripper.xml next to this file.

The arm is NOT redefined here. The upright two-module XML is produced by
../build_scene.py and then transformed, so segments, joints, masses and the
30 arm actuators stay identical to every other experiment:

  * robot_base moves to the ceiling mount at MOUNT_HEIGHT and is rotated
    180 degrees about X, so the arm hangs straight down;
  * a PneuNet-style gripper (gripper.py) is attached to the end of segment
    10, with 12 finger hinge actuators driven from ONE pressure line;
  * a ceiling plate and frame, a table, a place marker and FOUR free
    objects to pick up - cube, apple, banana, ball (see OBJECTS). The cube
    starts on the table; the others are parked on the floor off-camera, and
    hanging_robot.Scene.set_layout() swaps which one is on the table;
  * contacts are enabled only between the gripper, the objects, the table
    and the floor. The arm itself stays collision-free, as in the upright
    scene.
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

import build_scene  # noqa: E402  (the upright two-module generator)
import gripper as G  # noqa: E402

OUTPUT = HERE / "scene_hanging_gripper.xml"

MOUNT_HEIGHT = 0.80          # m, underside of the ceiling plate
TABLE_TOP = 0.372            # m; cube centre lands inside the reach band
TABLE_HALF = (0.22, 0.22)    # m
TABLE_CENTRE = (0.0, 0.0)

OBJECT_START = (0.060, 0.000)  # m, where the active object sits on the table
PLACE_TARGET = (-0.050, 0.050)

# Objects are sized to the robot. The gripper opens to ~84 mm, but the
# binding limit is vertical: the hanging arm's controllable TCP band is only
# ~395-437 mm (measured), and a pick needs the object's height plus a few mm
# of travel (open fingertips over its top when hovering, just above the table
# when grasping). So nothing may stand taller than ~36 mm. "rest" is the
# height of the body frame above the table in the natural pose; "top" is the
# object's highest point.
OBJECTS = {
    "cube":   {"mass": 0.020, "rest": 0.015, "top": 0.030,
               "rgba": "0.15 0.6 0.9 1"},
    "apple":  {"mass": 0.025, "rest": 0.015, "top": 0.036,
               "rgba": "0.78 0.1 0.12 1"},
    "banana": {"mass": 0.025, "rest": 0.010, "top": 0.020,
               "rgba": "0.96 0.83 0.2 1"},
    "ball":   {"mass": 0.012, "rest": 0.016, "top": 0.032,
               "rgba": "0.8 0.9 0.2 1"},
}
PARKING = (1.4, 1.2)         # m, floor spot for inactive objects, off-camera

FRICTION = "1.2 0.02 0.001"
SOFT_SOLREF = "0.02 1"      # MuJoCo's default contact time constant
SOFT_SOLIMP = "0.9 0.97 0.001"

FINGER_COLOUR = "0.95 0.95 0.92 0.95"
CHAMBER_COLOUR = "0.85 0.35 0.25 0.9"


def fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def add_gripper(parent):
    """Palm + three fingers, as children of segment_10."""

    palm = ET.SubElement(parent, "body", name="gripper_palm",
                         pos=fmt([0, 0, build_scene.SEGMENT_HEIGHT]))
    ET.SubElement(palm, "geom", name="palm", type="cylinder",
                  size=fmt([G.PALM_RADIUS, G.PALM_HALF_THICKNESS]),
                  pos=fmt([0, 0, G.PALM_HALF_THICKNESS]),
                  mass=f"{G.PALM_MASS}", rgba="0.25 0.25 0.28 1",
                  contype="1", conaffinity="1", friction=FRICTION)

    ET.SubElement(palm, "site", name="tcp", type="sphere", size="0.004",
                  pos=fmt([0, 0, G.TCP_OFFSET]), rgba="1 0.85 0.1 1")

    for f, angle in enumerate(G.FINGER_ANGLES_DEG):
        a = np.radians(angle)
        root = [G.FINGER_ROOT_RADIUS * np.cos(a),
                G.FINGER_ROOT_RADIUS * np.sin(a),
                2 * G.PALM_HALF_THICKNESS]
        # local x points radially outward, local z along the arm axis
        body = ET.SubElement(palm, "body", name=f"finger{f}_seg0",
                             pos=fmt(root), euler=f"0 0 {angle:.6g}")

        for s in range(G.FINGER_SEGMENTS):
            if s > 0:
                body = ET.SubElement(body, "body", name=f"finger{f}_seg{s}",
                                     pos=fmt([0, 0, G.SEGMENT_LENGTH]))

            ET.SubElement(body, "joint", name=G.joint_name(f, s),
                          type="hinge", axis="0 1 0", limited="true",
                          range=fmt(G.JOINT_RANGE_DEG), damping="0.002",
                          armature="0.0002")

            half_len = 0.5 * G.SEGMENT_LENGTH
            # strain-limiting layer = the gripping surface (collides)
            ET.SubElement(body, "geom", name=f"finger{f}_pad{s}", type="box",
                          size=fmt([G.FINGER_HALF_THICK, G.FINGER_HALF_WIDTH,
                                    half_len * 0.98]),
                          pos=fmt([0, 0, half_len]),
                          mass=f"{G.SEGMENT_MASS}", rgba=FINGER_COLOUR,
                          contype="1", conaffinity="1", condim="4",
                          friction=FRICTION, solref=SOFT_SOLREF,
                          solimp=SOFT_SOLIMP)
            # PneuNet chamber on the outside of the finger (visual only)
            ET.SubElement(body, "geom", type="box",
                          size=fmt([0.004, G.FINGER_HALF_WIDTH * 0.95,
                                    half_len * 0.62]),
                          pos=fmt([G.FINGER_HALF_THICK + 0.004, 0, half_len]),
                          rgba=CHAMBER_COLOUR, contype="0", conaffinity="0",
                          density="0")

            if s == G.FINGER_SEGMENTS - 1:
                ET.SubElement(body, "geom", name=f"finger{f}_tip",
                              type="capsule",
                              fromto=fmt([0, -G.FINGER_HALF_WIDTH * 0.8,
                                          G.SEGMENT_LENGTH,
                                          0, G.FINGER_HALF_WIDTH * 0.8,
                                          G.SEGMENT_LENGTH]),
                              size=f"{G.FINGER_HALF_THICK:.6g}",
                              rgba=FINGER_COLOUR, contype="1",
                              conaffinity="1", condim="4", friction=FRICTION,
                              solref=SOFT_SOLREF, solimp=SOFT_SOLIMP,
                              density="0")


def _object_geom(body, name, spec, mass, **attrs):
    return ET.SubElement(body, "geom", name=name, mass=f"{mass:.6g}",
                         rgba=attrs.pop("rgba", spec["rgba"]), contype="1",
                         conaffinity="1", condim=attrs.pop("condim", "4"),
                         friction=attrs.pop("friction", FRICTION), **attrs)


def build_cube(body, spec):
    _object_geom(body, "obj_cube_geom", spec, spec["mass"], type="box",
                 size=fmt([0.015] * 3))


def build_apple(body, spec):
    """A slightly flattened sphere with a stem and a leaf.

    `radius` and `half_height` default to the original 38 x 30 mm body, so a
    spec that does not set them produces byte-identical XML.
    """

    r = spec.get("radius", 0.019)
    h = spec.get("half_height", 0.015)
    k = r / 0.019                                  # scale the trimmings too
    _object_geom(body, "obj_apple_geom", spec, spec["mass"] * 0.97,
                 type="ellipsoid", size=fmt([r, r, h]))
    _object_geom(body, "obj_apple_stem", spec, spec["mass"] * 0.03,
                 type="cylinder", size=fmt([0.002 * k, 0.004 * k]),
                 pos=fmt([0.001 * k, 0, h + 0.002 * k]), euler="8 0 0",
                 rgba="0.35 0.22 0.1 1")
    ET.SubElement(body, "geom", type="ellipsoid",
                  size=fmt([0.006 * k, 0.0026 * k, 0.001 * k]),
                  pos=fmt([0.005 * k, 0, h + 0.004 * k]), euler="0 -25 0",
                  rgba="0.25 0.6 0.15 1", contype="0", conaffinity="0",
                  density="0")


def build_banana(body, spec):
    # a curved chain of capsules along an arc in the table plane; the body
    # frame sits at the middle of the arc, concave side towards +y
    radius, span, n = 0.075, np.radians(80.0), 7
    theta = np.linspace(-span / 2, span / 2, n + 1)
    pts = np.column_stack([radius * np.sin(theta),
                           radius * (1 - np.cos(theta)),
                           np.zeros_like(theta)])
    for i in range(n):
        middle = abs((i + 0.5) / n - 0.5) * 2          # 0 centre .. 1 ends
        _object_geom(body, f"obj_banana_geom{i}", spec, spec["mass"] / n,
                     type="capsule", fromto=fmt([*pts[i], *pts[i + 1]]),
                     size=f"{0.010 - 0.0022 * middle ** 2:.6g}")
    for end in (0, n):
        ET.SubElement(body, "geom", type="sphere", size="0.0045",
                      pos=fmt(pts[end]), rgba="0.3 0.22 0.08 1",
                      contype="0", conaffinity="0", density="0")


def build_can(body, spec):
    """A drinks can: a tall, narrow aluminium cylinder with a rim and a label.

    Deliberately the opposite proportion to the cube and the apple - 22 mm
    across and 38 mm tall, an aspect ratio of 1.7 against the cube's 1.0 - so
    the three objects do not all read as "roughly fist-sized blob". The
    height is set by the arm, not by taste: the open fingertips reach only
    412 mm and the table is at 372 mm, so an object taller than 40 mm cannot
    be cleared on the approach.

    The cylinder is the only contact geom; the rim and label are paint.
    """

    r = spec.get("radius", 0.011)
    h = spec.get("half_height", 0.019)
    _object_geom(body, "obj_can_geom", spec, spec["mass"], type="cylinder",
                 size=fmt([r, h]))
    for z in (-h + 0.0008, h - 0.0008):
        ET.SubElement(body, "geom", type="cylinder",
                      size=fmt([r * 0.88, 0.0012]), pos=fmt([0, 0, z]),
                      rgba="0.55 0.58 0.62 1", contype="0", conaffinity="0",
                      density="0")
    ET.SubElement(body, "geom", type="cylinder",
                  size=fmt([r * 1.02, h * 0.42]), pos=fmt([0, 0, 0.001]),
                  rgba="0.1 0.55 0.45 1", contype="0", conaffinity="0",
                  density="0")


def build_ball(body, spec):
    # condim 6 adds rolling friction so the ball does not roll away when
    # a finger brushes it
    _object_geom(body, "obj_ball_geom", spec, spec["mass"], type="sphere",
                 size="0.016", condim="6", friction="1.2 0.02 0.002")


def build_cup(body, spec):
    """A small cup: a solid cylinder for contact, painted with a dark mouth,
    a rim and a handle.

    Solid on purpose - the soft fingers close on the OUTSIDE wall, so the
    hollow interior would only add contacts, not change the grasp. The handle
    is paint (no contact) so it cannot snag a finger or a neighbour.
    """

    r = spec.get("radius", 0.013)
    h = spec.get("half_height", 0.016)
    _object_geom(body, "obj_cup_geom", spec, spec["mass"], type="cylinder",
                 size=fmt([r, h]))
    ET.SubElement(body, "geom", type="cylinder", size=fmt([r * 0.8, 0.0006]),
                  pos=fmt([0, 0, h]), rgba="0.25 0.18 0.08 1",
                  contype="0", conaffinity="0", density="0")        # mouth
    ET.SubElement(body, "geom", type="cylinder",
                  size=fmt([r * 1.04, 0.0012]), pos=fmt([0, 0, h - 0.0012]),
                  rgba="0.98 0.98 0.95 1", contype="0", conaffinity="0",
                  density="0")                                       # rim
    for z0, z1 in ((0.008, 0.0045), (-0.008, -0.0045)):
        ET.SubElement(body, "geom", type="capsule", size="0.0022",
                      fromto=fmt([r, 0, z0, r + 0.006, 0, z1]),
                      rgba=spec["rgba"], contype="0", conaffinity="0",
                      density="0")
    ET.SubElement(body, "geom", type="capsule", size="0.0022",
                  fromto=fmt([r + 0.006, 0, 0.0045, r + 0.006, 0, -0.0045]),
                  rgba=spec["rgba"], contype="0", conaffinity="0",
                  density="0")                                       # handle


def build_teddy(body, spec):
    """A teddy bear: a round body and a head, both contact geoms, with ears,
    muzzle, eyes, arms and legs as paint. Its top stays under the ~40 mm
    the open fingertips can clear."""

    br = spec.get("body_radius", 0.0135)
    bh = spec.get("body_half_height", 0.012)
    hr = spec.get("head_radius", 0.0085)
    hz = bh + hr * 0.75
    _object_geom(body, "obj_teddy_body", spec, spec["mass"] * 0.7,
                 type="ellipsoid", size=fmt([br, br * 0.9, bh]))
    _object_geom(body, "obj_teddy_head", spec, spec["mass"] * 0.3,
                 type="sphere", size=fmt([hr]), pos=fmt([0, 0, hz]))
    paint = dict(contype="0", conaffinity="0", density="0")
    for side in (-1, 1):
        ET.SubElement(body, "geom", type="sphere", size="0.0034",
                      pos=fmt([0, side * hr * 0.75, hz + hr * 0.7]),
                      rgba=spec["rgba"], **paint)                    # ears
        ET.SubElement(body, "geom", type="sphere", size="0.0011",
                      pos=fmt([hr * 0.85, side * 0.003, hz + 0.0025]),
                      rgba="0.05 0.04 0.03 1", **paint)              # eyes
        ET.SubElement(body, "geom", type="ellipsoid",
                      size="0.0035 0.0035 0.006",
                      pos=fmt([0.002, side * br * 0.95, 0.002]),
                      rgba=spec["rgba"], **paint)                    # arms
        ET.SubElement(body, "geom", type="ellipsoid",
                      size="0.005 0.004 0.0035",
                      pos=fmt([br * 0.7, side * br * 0.45, -bh * 0.7]),
                      rgba=spec["rgba"], **paint)                    # legs
    ET.SubElement(body, "geom", type="sphere", size="0.0038",
                  pos=fmt([hr * 0.8, 0, hz - 0.001]),
                  rgba="0.85 0.7 0.5 1", **paint)                    # muzzle
    ET.SubElement(body, "geom", type="ellipsoid", size="0.0015 0.0055 0.007",
                  pos=fmt([br * 0.8, 0, 0]), rgba="0.85 0.7 0.5 1",
                  **paint)                                           # tummy


OBJECT_BUILDERS = {"cube": build_cube, "apple": build_apple,
                   "banana": build_banana, "ball": build_ball,
                   "can": build_can, "cup": build_cup, "teddy": build_teddy}

# The can is NOT in OBJECTS, so every scene built from OBJECTS - including
# the pick-and-place scene a trained policy was measured on - is unchanged by
# its existence. Scenes that want it build from ALL_OBJECTS instead; see
# build_continual_scene.py. Order matters: appending keeps every existing
# object's parking slot (PARKING[0] + 0.15 * k) where it was.
EXTRA_OBJECTS = {
    "can": {"mass": 0.020, "rest": 0.019, "top": 0.038, "radius": 0.011,
            "half_height": 0.019, "rgba": "0.72 0.74 0.78 1"},
    # the five-object continual set adds these two (appended, so every
    # earlier object keeps its parking slot)
    "cup": {"mass": 0.018, "rest": 0.016, "top": 0.032, "radius": 0.013,
            "half_height": 0.016, "rgba": "0.95 0.78 0.12 1"},
    "teddy": {"mass": 0.020, "rest": 0.012, "top": 0.0369,
              "body_radius": 0.0135, "body_half_height": 0.012,
              "head_radius": 0.0085, "rgba": "0.55 0.34 0.17 1"},
}

# The continual experiment wants the three objects to look and behave
# DIFFERENTLY, not to be three 30 mm blobs, so it overrides the apple with a
# bigger one. `OBJECTS` itself is untouched, so scene_bowl_drop.xml - which a
# measured pick-and-place checkpoint depends on - still regenerates
# byte-identically.
#
#   cube    30 x 30 mm box          aspect 1.00   half-width 15.0 mm
#   apple   44 mm wide, 34 mm tall  aspect 0.77   half-width 22.0 mm
#   can     22 mm wide, 38 mm tall  aspect 1.73   half-width 11.0 mm
#
# Every "top" stays under the 40 mm the open fingertips can clear.
CONTINUAL_OBJECTS = {
    **OBJECTS,
    # 30 g, not the 45 g this started at. Density is the check: the cube is
    # 20 g of 27 cm3, i.e. 0.74 g/cm3, and this apple is 34.5 cm3, so 45 g
    # made it 1.8x denser than everything else on the table. It mattered -
    # the fingertips close on a 22 mm radius at about 0.20 bar, and at that
    # pressure they sit ~3 mm BELOW the apple's equator, so a round object
    # gets squeezed upward out of the hand and a heavier one goes sooner.
    # Measured over 12 episodes: 45 g grasped 10/12, 30 g grasped 12/12, and
    # over a full 170-episode run 45 g failed to pick 45 times.
    "apple": {"mass": 0.030, "rest": 0.017, "top": 0.039, "radius": 0.022,
              "half_height": 0.017, "rgba": "0.78 0.1 0.12 1"},
    **EXTRA_OBJECTS,
}

# half-width on the table, used to keep an object clear of the table's hole
# and of its neighbours when several are placed at once
FOOTPRINT = {"cube": 0.0212,        # a 30 mm box, corner on
             "apple": 0.022,
             "banana": 0.045,
             "ball": 0.016,
             "can": 0.011,
             "cup": 0.014,          # 26 mm body; the painted handle is not counted
             "teddy": 0.016}        # 27 mm body plus painted arms

ALL_OBJECTS = CONTINUAL_OBJECTS


def add_world(world):
    top = MOUNT_HEIGHT

    # ceiling plate + two-post frame (visual)
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

    # table
    ET.SubElement(world, "geom", name="table", type="box",
                  size=fmt([*TABLE_HALF, 0.015]),
                  pos=fmt([*TABLE_CENTRE, TABLE_TOP - 0.015]),
                  rgba="0.55 0.42 0.3 1", contype="1", conaffinity="1",
                  friction=FRICTION)
    for dx in (-1, 1):
        for dy in (-1, 1):
            ET.SubElement(world, "geom", type="box",
                          size=fmt([0.015, 0.015, 0.5 * (TABLE_TOP - 0.03)]),
                          pos=fmt([TABLE_CENTRE[0] + dx * (TABLE_HALF[0] - 0.03),
                                   TABLE_CENTRE[1] + dy * (TABLE_HALF[1] - 0.03),
                                   0.5 * (TABLE_TOP - 0.03)]),
                          rgba="0.45 0.34 0.24 1", contype="0",
                          conaffinity="0")

    # the objects: the cube on the table, the rest parked on the floor
    for k, (name, spec) in enumerate(OBJECTS.items()):
        on_table = name == "cube"
        pos = ([*OBJECT_START, TABLE_TOP + spec["rest"] + 1e-4] if on_table
               else [PARKING[0] + 0.15 * k, PARKING[1], spec["rest"] + 1e-4])
        body = ET.SubElement(world, "body", name=f"obj_{name}", pos=fmt(pos))
        ET.SubElement(body, "freejoint", name=f"obj_{name}_free")
        OBJECT_BUILDERS[name](body, spec)

    # where it should end up (visual)
    ET.SubElement(world, "site", name="place_target", type="cylinder",
                  size="0.022 0.0006",
                  pos=fmt([*PLACE_TARGET, TABLE_TOP + 0.0006]),
                  rgba="0.2 0.85 0.3 0.8")

    ET.SubElement(world, "camera", name="task_camera",
                  pos=fmt([0.62, -0.62, 0.78]),
                  xyaxes="0.707 0.707 0 -0.39 0.39 0.834")


def build():
    root = ET.fromstring(build_scene.build().split("?>", 1)[1])

    root.set("model", "i_support_2module_hanging_gripper")
    compiler = root.find("compiler")
    compiler.set("meshdir", "sim/meshes")

    option = root.find("option")
    option.set("cone", "elliptic")
    option.set("impratio", "10")

    world = root.find("worldbody")

    # The upright scene's green target site and floor stay; hang the arm.
    base = world.find("body[@name='robot_base']")
    base.set("pos", fmt([0, 0, MOUNT_HEIGHT]))
    base.set("quat", "0 1 0 0")                  # 180 deg about X: hang down

    seg10 = base.find(".//body[@name='segment_10']")
    old_site = seg10.find("site[@name='end_effector']")
    old_site.set("rgba", "1 0 0 0")               # keep, hidden
    add_gripper(seg10)
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
              "    Two-module I-Support robot HANGING from a ceiling mount,\n"
              "    with a PneuNet-style soft gripper (one pressure line).\n"
              "    30 arm actuators (6 pressures) + 12 finger actuators "
              "(1 pressure).\n\n"
              "    GENERATED FILE - edit build_hanging_scene.py / gripper.py "
              "and re-run.\n"
              "-->\n")
    return header + ET.tostring(root, encoding="unicode") + "\n"


def main():
    OUTPUT.write_text(build())
    print(f"Wrote {OUTPUT}")
    print(f"  mount height {MOUNT_HEIGHT} m, table top {TABLE_TOP} m")
    print(f"  gripper: {G.NUM_FINGERS} fingers x {G.FINGER_SEGMENTS} hinges, "
          f"one pressure line (0-{G.P_MAX} bar)")
    print(f"  objects: {', '.join(OBJECTS)}")


if __name__ == "__main__":
    main()
