#!/usr/bin/env python3
"""
Generate the two-module I-Support scene.

    python 2_module_ISupport/build_scene.py

Writes `scene_2module.xml` next to this file. The generated XML is committed,
so you only need to re-run this if you change the geometry.

--------------------------------------------------------------------------
WHAT IS DIFFERENT FROM THE ONE-MODULE ROBOT
--------------------------------------------------------------------------
The single-module arm in sim/scene.xml is one 5-segment module driven by
three pneumatic chambers. This scene stacks a *second identical module* on
top of the first:

    module A : segments  1..5,  chambers A1 A2 A3   (valves 1-3)
    module B : segments  6..10, chambers B1 B2 B3   (valves 4-6)

So the robot has 10 segments, 30 position actuators and 6 independent
pressure inputs. Each module bends and extends on its own; module B rides on
whatever pose module A has taken.

The two modules are geometrically identical - module B reuses the exact same
five segmented meshes as module A - with an interface disc between them
standing in for the coupling flange.

--------------------------------------------------------------------------
MESHES
--------------------------------------------------------------------------
`meshdir` points at ../sim/meshes so the 18 MB of STLs are not duplicated.
That makes this folder depend on its sibling; if you ever need it standalone,
copy sim/meshes/ in here and drop the ../ from the compiler meshdir.
"""

from __future__ import annotations

from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "scene_2module.xml"

SEGMENTS_PER_MODULE = 5
MODULES = 2

SEGMENT_HEIGHT = 0.024          # m, spacing between segment frames
SEGMENT_MASS = 0.034            # kg, carried by the link capsule
LINK_RADIUS = 0.030             # m, capsule radius (visual/mass only)

BEND_LIMIT_DEG = 45
EXTENSION_LIMIT = 0.015         # m per segment

BEND_KP = 50
EXTENSION_KP = 1000

# Local placement of a module's end plate, measured at the top of that
# module's last segment. MuJoCo recentres mesh vertices on the mesh frame and
# folds the remainder into geom_pos, so this number is a property of
# endA1_binary.STL and is the same wherever the disc is used.
END_DISC_POS = -0.14921000671

CAVITY_COLOURS = {
    1: "1 0 0 0.5",
    2: "0 1 0 0.5",
    3: "0 0 1 0.5",
}

SHELL_COLOUR = "0.8 0.8 0.8 1"
LINK_COLOUR = "0.15 0.45 1 0.28"


def indent(level):
    return "    " * level


def mesh_assets():
    """
    Declare every mesh once.

    The five per-segment meshes are shared by both modules: module B's
    segment 6 uses the same `border_1` asset as module A's segment 1.
    """

    out = []
    add = out.append

    add(indent(1) + "<!-- Base and end plates -->")
    add(indent(1) + '<mesh name="base_mesh" file="baseA1_binary.STL" '
                    'scale="0.001 0.001 0.001"/>')
    add(indent(1) + '<mesh name="end_mesh" file="endA1_binary.STL" '
                    'scale="0.001 0.001 0.001"/>')
    add("")

    add(indent(1) + "<!-- Fixed lower CAD section (segment 0) -->")
    for name, stem in (
        ("fixed_border_0", "borderA1_binary_seg0"),
        ("fixed_chambers_0", "chambersA1_binary_seg0"),
        ("fixed_cavity1_0", "cavityA1_binary_seg0"),
        ("fixed_cavity2_0", "cavityA2_binary_seg0"),
        ("fixed_cavity3_0", "cavityA3_binary_seg0"),
    ):
        add(indent(1) + f'<mesh name="{name}" '
                        f'file="segmented_corrected/{stem}.stl" '
                        f'scale="0.001 0.001 0.001"/>')
    add("")

    add(indent(1) + "<!-- One module's five segments; both modules "
                    "reference these -->")
    for s in range(1, SEGMENTS_PER_MODULE + 1):
        add(indent(1) + f'<mesh name="border_{s}" '
                        f'file="segmented_corrected/borderA1_binary_seg{s}.stl" '
                        f'scale="0.001 0.001 0.001"/>')
        add(indent(1) + f'<mesh name="chambers_{s}" '
                        f'file="segmented_corrected/chambersA1_binary_seg{s}.stl" '
                        f'scale="0.001 0.001 0.001"/>')
        for c in (1, 2, 3):
            add(indent(1) + f'<mesh name="cavity{c}_{s}" '
                            f'file="segmented_corrected/'
                            f'cavityA{c}_binary_seg{s}.stl" '
                            f'scale="0.001 0.001 0.001"/>')
        add("")

    return out


def segment_body(global_index, module, local_index, level):
    """
    Emit one segment body, opened but not closed.

    `global_index` runs 1..10 and names the joints and actuators.
    `local_index` runs 1..5 and selects which mesh to draw, so module B
    repeats module A's geometry exactly.
    """

    i = indent(level)
    j = indent(level + 1)

    out = []
    add = out.append

    add("")
    add(i + f"<!-- ============ MODULE {module} — SEGMENT "
            f"{global_index} ============ -->")
    add(i + f'<body name="segment_{global_index}" '
            f'pos="0 0 {SEGMENT_HEIGHT}">')

    add(j + f'<joint name="bend_x_{global_index}" type="hinge" '
            f'axis="1 0 0" limited="true" '
            f'range="-{BEND_LIMIT_DEG} {BEND_LIMIT_DEG}"/>')
    add(j + f'<joint name="bend_y_{global_index}" type="hinge" '
            f'axis="0 1 0" limited="true" '
            f'range="-{BEND_LIMIT_DEG} {BEND_LIMIT_DEG}"/>')
    add(j + f'<joint name="extension_{global_index}" type="slide" '
            f'axis="0 0 1" limited="true" range="0 {EXTENSION_LIMIT}"/>')
    add("")

    # The capsule carries the segment mass; every CAD mesh is density="0".
    add(j + f'<geom name="link_{global_index}" type="capsule" '
            f'fromto="0 0 0 0 0 {SEGMENT_HEIGHT}" size="{LINK_RADIUS}" '
            f'mass="{SEGMENT_MASS}" contype="0" conaffinity="0" '
            f'rgba="{LINK_COLOUR}"/>')

    add(j + f'<geom type="mesh" mesh="border_{local_index}" contype="0" '
            f'conaffinity="0" density="0" rgba="{SHELL_COLOUR}"/>')
    add(j + f'<geom type="mesh" mesh="chambers_{local_index}" contype="0" '
            f'conaffinity="0" density="0" rgba="{SHELL_COLOUR}"/>')

    for c in (1, 2, 3):
        add(j + f'<geom type="mesh" mesh="cavity{c}_{local_index}" '
                f'contype="0" conaffinity="0" density="0" '
                f'rgba="{CAVITY_COLOURS[c]}"/>')

    return out


def worldbody():
    out = []
    add = out.append

    add(indent(1) + '<site name="target_site" type="sphere" '
                    'pos="0.10 0 0.30" size="0.01" rgba="0 1 0 0"/>')
    add(indent(1) + '<geom name="floor" type="plane" size="2 2 0.1" '
                    'material="ground"/>')
    add(indent(1) + '<light name="main_light" pos="0 0 2.0" dir="0 0 -1"/>')
    add(indent(1) + '<camera name="rgb_camera" pos="0.45 -0.45 0.32" '
                    'xyaxes="0.707 0.707 0  -0.408 0.408 0.816"/>')
    add("")

    add(indent(1) + '<body name="robot_base" pos="0 0 0">')
    add(indent(2) + '<geom name="base" type="mesh" mesh="base_mesh" '
                    f'contype="0" conaffinity="0" density="0" '
                    f'rgba="{SHELL_COLOUR}"/>')

    for name, colour in (
        ("fixed_border_0", SHELL_COLOUR),
        ("fixed_chambers_0", SHELL_COLOUR),
        ("fixed_cavity1_0", CAVITY_COLOURS[1]),
        ("fixed_cavity2_0", CAVITY_COLOURS[2]),
        ("fixed_cavity3_0", CAVITY_COLOURS[3]),
    ):
        add(indent(2) + f'<geom name="{name}" type="mesh" mesh="{name}" '
                        f'contype="0" conaffinity="0" density="0" '
                        f'rgba="{colour}"/>')

    # -- nested segments ----------------------------------------------------

    level = 2
    total = MODULES * SEGMENTS_PER_MODULE

    for g in range(1, total + 1):
        module = "A" if g <= SEGMENTS_PER_MODULE else "B"
        local = ((g - 1) % SEGMENTS_PER_MODULE) + 1

        out.extend(segment_body(g, module, local, level))

        last_of_module = (g % SEGMENTS_PER_MODULE) == 0

        if last_of_module and g < total:
            # Coupling flange between the two modules.
            add("")
            add(indent(level + 1) + f'<!-- Interface disc: module {module} '
                                    f'to module B -->')
            add(indent(level + 1) + f'<geom name="interface_disc" type="mesh" '
                                    f'mesh="end_mesh" '
                                    f'pos="0 0 {END_DISC_POS}" contype="0" '
                                    f'conaffinity="0" density="0" '
                                    f'rgba="{SHELL_COLOUR}"/>')

        if g == total:
            add("")
            add(indent(level + 1) + '<!-- End plate and tip reference -->')
            add(indent(level + 1) + f'<geom name="end_disc" type="mesh" '
                                    f'mesh="end_mesh" '
                                    f'pos="0 0 {END_DISC_POS}" contype="0" '
                                    f'conaffinity="0" density="0" '
                                    f'rgba="{SHELL_COLOUR}"/>')
            add(indent(level + 1) + f'<site name="end_effector" '
                                    f'pos="0 0 {SEGMENT_HEIGHT}" '
                                    f'size="0.005" rgba="1 0 0 1"/>')

        level += 1

    # close every segment body, then robot_base
    for lvl in range(level - 1, 1, -1):
        add(indent(lvl) + "</body>")

    add(indent(1) + "</body>")

    return out


def actuators():
    out = []
    add = out.append

    total = MODULES * SEGMENTS_PER_MODULE

    for g in range(1, total + 1):
        module = "A" if g <= SEGMENTS_PER_MODULE else "B"

        if (g - 1) % SEGMENTS_PER_MODULE == 0:
            add("")
            add(indent(1) + f"<!-- Module {module} -->")

        add(indent(1) + f'<position joint="bend_x_{g}" kp="{BEND_KP}"/>')
        add(indent(1) + f'<position joint="bend_y_{g}" kp="{BEND_KP}"/>')
        add(indent(1) + f'<position joint="extension_{g}" '
                        f'kp="{EXTENSION_KP}"/>')

    return out


def build():
    total = MODULES * SEGMENTS_PER_MODULE

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "",
        "<!--",
        "    Two-module I-Support soft continuum robot.",
        "",
        f"    {MODULES} modules x {SEGMENTS_PER_MODULE} segments = "
        f"{total} segments, {3 * total} position actuators.",
        "    6 independent pressure inputs: chambers A1 A2 A3 (module A),",
        "    B1 B2 B3 (module B).",
        "",
        "    GENERATED FILE - edit build_scene.py and re-run it instead.",
        "-->",
        "",
        '<mujoco model="i_support_2module">',
        "",
        indent(1) + '<compiler angle="degree" inertiafromgeom="true" '
                    'meshdir="sim/meshes"/>',
        "",
        indent(1) + '<option gravity="0 0 -9.80513" timestep="0.0002" '
                    'integrator="RK4"/>',
        "",
        indent(1) + "<asset>",
        "",
        indent(2) + '<texture name="sky" type="skybox" builtin="gradient" '
                    'rgb1="0.2 0.3 0.5" rgb2="0 0 0" width="512" '
                    'height="512"/>',
        indent(2) + '<material name="ground" rgba="0.8 0.8 0.8 1"/>',
        "",
    ]

    lines.extend(indent(1) + line if line else "" for line in mesh_assets())
    lines.append(indent(1) + "</asset>")
    lines.append("")
    lines.append(indent(1) + "<worldbody>")
    lines.append("")
    lines.extend(worldbody())
    lines.append("")
    lines.append(indent(1) + "</worldbody>")
    lines.append("")
    lines.append(indent(1) + "<actuator>")
    lines.extend(actuators())
    lines.append("")
    lines.append(indent(1) + "</actuator>")
    lines.append("")
    lines.append("</mujoco>")

    return "\n".join(lines) + "\n"


def main():
    xml = build()
    OUTPUT.write_text(xml)

    total = MODULES * SEGMENTS_PER_MODULE

    print(f"Wrote {OUTPUT}")
    print(f"  modules          : {MODULES}")
    print(f"  segments         : {total}")
    print(f"  joints           : {3 * total}")
    print(f"  actuators        : {3 * total}")
    print(f"  pressure inputs  : {3 * MODULES}")
    print(f"  lines            : {len(xml.splitlines())}")


if __name__ == "__main__":
    main()
