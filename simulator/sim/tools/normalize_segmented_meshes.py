#!/usr/bin/env python3
"""
Re-localize segmented STL meshes for the MuJoCo model.

Input:
    model/meshes/segmented/*_seg0.stl ... *_seg6.stl

Output:
    model/meshes/segmented_corrected/*_seg0.stl ... *_seg6.stl

The segmentation was done in the original CAD/world coordinate system.
This script translates every piece in a segment into that segment's local
coordinate system while preserving its X/Y coordinates.

Install once:
    pip install trimesh numpy
"""

from pathlib import Path
import re
import shutil

import numpy as np
import trimesh


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_DIR / "model" / "meshes" / "segmented"
OUTPUT_DIR = PROJECT_DIR / "model" / "meshes" / "segmented_corrected"

# The MuJoCo model currently uses 5 moving segments.
# The segmentation output may contain 6 or 7 slices; keep all generated
# pieces so nothing is silently deleted.
SEGMENT_LENGTH = 0.038


def segment_number(path: Path):
    m = re.search(r"_seg(\d+)\.stl$", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def load_mesh(path: Path):
    mesh = trimesh.load_mesh(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in INPUT_DIR.glob("*.stl") if segment_number(p) is not None],
        key=lambda p: (segment_number(p), p.name.lower()),
    )

    if not files:
        raise FileNotFoundError(
            f"No segmented STL files found in {INPUT_DIR}"
        )

    # Find the global Z start of every segmentation slice from the actual
    # mesh geometry. All meshes belonging to the same slice receive the
    # exact same translation.
    grouped = {}
    for path in files:
        grouped.setdefault(segment_number(path), []).append(path)

    for seg, seg_files in sorted(grouped.items()):
        meshes = []
        for path in seg_files:
            mesh = load_mesh(path)
            meshes.append((path, mesh))

        # Use the lowest Z coordinate among ALL pieces in this segment.
        # This keeps chambers, borders and cavities aligned with one another.
        z_min = min(float(mesh.bounds[0, 2]) for _, mesh in meshes)

        # Translate the whole segment into local coordinates.
        # X/Y are intentionally untouched.
        for path, mesh in meshes:
            mesh.vertices[:, 2] -= z_min

            out = OUTPUT_DIR / path.name
            mesh.export(out)

        print(
            f"Segment {seg}: {len(meshes)} meshes, "
            f"global z_min={z_min:.6f} m/mm-unscaled -> local z=0"
        )

    # Copy any non-segmented STL files so the corrected folder can be used
    # as a complete mesh source if needed.
    for path in INPUT_DIR.glob("*.stl"):
        if segment_number(path) is None:
            shutil.copy2(path, OUTPUT_DIR / path.name)

    print()
    print(f"Done.")
    print(f"Corrected meshes: {OUTPUT_DIR}")
    print()
    print("IMPORTANT:")
    print("The XML must reference segmented_corrected/ for the segmented")
    print("meshes. Do not add another Z translation to the mesh itself.")


if __name__ == "__main__":
    main()