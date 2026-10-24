#!/usr/bin/env python3
"""
Generate the scene for the CONTINUAL pick-and-drop experiment.

    python simulator/build_continual_scene.py     -> scene_continual.xml

It is the bowl-drop scene with one object added: a drinks can. Everything
else - arm, gripper, wrist camera, table with its hole, bowl - is
`build_bowl_scene` verbatim, so a policy trained here sees exactly the
environment the pick-and-place policy was measured in.

WHY A SEPARATE XML RATHER THAN ADDING THE CAN TO scene_bowl_drop.xml
-------------------------------------------------------------------
`pick_place_exp` has a trained checkpoint and a measured success rate tied
to that file. A parked can would never appear in either camera, so the
policy's inputs would be identical - but it is one more free body with a
floor contact, and MuJoCo solves the contacts of the whole model at once, so
the arm's own numbers are not guaranteed to reproduce bit for bit. Writing a
second XML costs nothing and keeps the earlier result exactly reproducible;
`build_bowl_scene.py` still regenerates byte-identically after this change.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_bowl_scene as BS  # noqa: E402
import build_hanging_scene as B  # noqa: E402

OUTPUT = HERE / "scene_continual.xml"

OBJECTS = B.ALL_OBJECTS          # cube, apple, banana, ball, can, cup, teddy


def build():
    xml = BS.build(objects=OBJECTS,
                   model_name="i_support_2module_continual")
    return xml.replace(
        "    with a PneuNet-style soft gripper and a BOWL sunk into the\n"
        "    table: pick the object up and drop it in.\n",
        "    with a PneuNet-style soft gripper and a BOWL sunk into the\n"
        "    table: pick the NAMED object up and drop it in.\n"
        "    Five objects (cube, apple, banana, ball, can); one is on the\n"
        "    table, the rest are parked off camera.\n").replace(
        "    GENERATED FILE - edit build_bowl_scene.py and re-run.",
        "    GENERATED FILE - edit build_continual_scene.py and re-run.")


def main():
    OUTPUT.write_text(build())
    print(f"Wrote {OUTPUT}")
    print(f"  objects  : {', '.join(OBJECTS)}")
    for name, spec in OBJECTS.items():
        print(f"    {name:7s} rest {spec['rest'] * 1000:5.1f} mm   "
              f"top {spec['top'] * 1000:5.1f} mm   "
              f"mass {spec['mass'] * 1000:4.1f} g")
    print(f"  bowl rim : {BS.BOWL_RIM_Z * 1000:.0f} mm, inner radius "
          f"{BS.BOWL_R_RIM * 1000:.0f} mm, {BS.BOWL_DEPTH * 1000:.0f} mm deep")


if __name__ == "__main__":
    main()
