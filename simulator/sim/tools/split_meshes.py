#!/usr/bin/env python3
"""
Split long STL meshes into Z segments using boolean clipping.

Requirements:
    pip install trimesh manifold3d numpy

Edit INPUT_DIR and OUTPUT_DIR if needed.
"""

import os
import numpy as np
import trimesh

INPUT_DIR = "sim/meshes"
OUTPUT_DIR = os.path.join(INPUT_DIR, "segmented")

FILES = [
    "borderA1_binary.STL",
    "chambersA1_binary.STL",
    "cavityA1_binary.STL",
    "cavityA2_binary.STL",
    "cavityA3_binary.STL",
]

# mm
PLANES = [-15.59, 25.0, 49.0, 73.0, 97.0, 121.0, 145.0, 179.21]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clip_box(mesh, z0, z1, pad=1000.0):
    box = trimesh.creation.box(extents=[pad, pad, z1-z0])
    box.apply_translation([0,0,(z0+z1)/2])
    try:
        out = trimesh.boolean.intersection([mesh, box], engine="manifold")
    except BaseException as e:
        print(f"Boolean failed: {e}")
        return None
    return out

def main():
    for f in FILES:
        path = os.path.join(INPUT_DIR, f)
        if not os.path.exists(path):
            print(f"Missing {path}")
            continue
        mesh = trimesh.load(path, force="mesh")
        print(f"Loaded {f}: {len(mesh.faces)} faces")

        stem = os.path.splitext(f)[0]
        for i in range(len(PLANES)-1):
            z0 = PLANES[i]
            z1 = PLANES[i+1]
            seg = clip_box(mesh, z0, z1)
            if seg is None or seg.is_empty:
                print(f"Segment {i} empty")
                continue
            out = os.path.join(OUTPUT_DIR, f"{stem}_seg{i}.stl")
            seg.export(out)
            print("Saved", out)

if __name__ == "__main__":
    main()