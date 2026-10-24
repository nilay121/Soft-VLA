#!/usr/bin/env python3
"""
Have the expert trace the evaluation circles and save what it looked like.

    python record_goal_trajectory.py
    python record_goal_trajectory.py --only circle_r45_z415

This is the "teach" half of teach-and-repeat. The heuristic controller drives
the arm around each circle in `training_paths.EVAL_CIRCLES` and every frame
from both policy cameras is written out. Those frames are the only thing the
policy will be told about the path it has to follow - deploy.py feeds them in
as the goal channels and never passes the reference coordinates.

The expert runs CLEAN here: no disturbance injection. The recording is meant
to be the tidiest available demonstration of the circle, which is what you
would get by hand-guiding a real robot once.

Written to goals/ next to this file:
    <name>.npy          frames [T, 2, 112, 112, 3] uint8 - the goal images
    <name>.npz          reference path, achieved tcp, expert pressures,
                        on_shape mask
    <name>_sheet.png    a contact sheet of both views, for looking at
    goals/manifest.json every circle, its geometry and the expert's error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import conditioning as C  # noqa: E402
import heuristic_track as H  # noqa: E402
import training_paths as P  # noqa: E402
from track_cameras import policy_cameras  # noqa: E402
from track_scene import TrackScene, balanced_pressure  # noqa: E402

# dataset/, goals/, checkpoints/, results/ live in this folder, or in
# TRACK_OUT (e.g. seeds/, the 3-seed study; relative = to this folder)
OUT_DIR = HERE / os.environ.get("TRACK_OUT", ".")
GOAL_DIR = OUT_DIR / "goals"


def record(spec, args):
    import mujoco
    from viz.common import hide_link_capsules, style_scene

    hargs = H.parse_args(["--no-video", "--gain", str(args.gain)])
    scene = TrackScene(hargs.control_dt)
    scene.reset(balanced_pressure(hargs.p0))
    start = scene.tcp()

    reference, on_shape = P.eval_reference(spec, scene.control_dt, start,
                                           speed=args.speed)

    model = scene.model
    hide_link_capsules(model)
    style_scene(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.image)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.image)
    cameras, camera_cfg = policy_cameras(scene, model)
    renderer = mujoco.Renderer(model, height=args.image, width=args.image)

    every = max(1, int(round(C.POLICY_DT / scene.control_dt)))
    frames, marks = [], []

    def on_step(k, scene, pressure, t):
        if k % every:
            return None
        views = []
        for cam in cameras:
            renderer.update_scene(scene.data, camera=cam)
            views.append(renderer.render().copy())
        frames.append(np.stack(views))
        marks.append(bool(on_shape[k]))
        return None          # clean: the taught trace carries no disturbance

    log = H.follow(scene, reference, hargs, on_step=on_step)
    m = H.metrics(log, on_shape)

    GOAL_DIR.mkdir(parents=True, exist_ok=True)
    name = spec["name"]
    np.save(GOAL_DIR / f"{name}.npy", np.asarray(frames, dtype=np.uint8))
    np.savez(GOAL_DIR / f"{name}.npz",
             reference=log["reference"][::every].astype(np.float32),
             tcp=log["tcp"][::every].astype(np.float32),
             pressure=log["pressure"][::every].astype(np.float32),
             on_shape=np.asarray(marks, dtype=bool),
             reference_full=log["reference"].astype(np.float32),
             on_shape_full=np.asarray(on_shape, dtype=bool))
    contact_sheet(frames, marks, GOAL_DIR / f"{name}_sheet.png", args)

    return {"name": name, "shape": spec.get("shape", "circle"),
            "radius_m": spec["radius"],
            "centre_m": list(spec["centre"]),
            "frames": len(frames), "policy_dt_s": C.POLICY_DT,
            "speed_mps": args.speed,
            "expert_error_mm": {k: round(v, 3) for k, v in m.items()},
            "camera": camera_cfg}


def contact_sheet(frames, marks, path, args, columns=8):
    """Both views, evenly spaced over the traced part of the run."""

    import cv2

    on = np.nonzero(np.asarray(marks))[0]
    picks = np.linspace(on[0], on[-1], columns).round().astype(int)
    tile = 168
    rows = []
    for v in (0, 1):
        strip = [cv2.resize(np.asarray(frames[i][v]), (tile, tile),
                            interpolation=cv2.INTER_NEAREST) for i in picks]
        rows.append(np.hstack(strip))
    sheet = np.vstack(rows)
    sheet = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
    for c, i in enumerate(picks):
        cv2.putText(sheet, f"t={i * C.POLICY_DT:.1f}s", (c * tile + 4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.putText(sheet, "side_a", (4, tile - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (120, 230, 120), 1, cv2.LINE_AA)
    cv2.putText(sheet, "side_b", (4, 2 * tile - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (120, 230, 120), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet)


def main(args):
    print()
    print("=" * 76)
    print("RECORDING THE TAUGHT SHAPES — expert traces, cameras watch")
    print("=" * 76)

    specs = [s for s in P.EVAL_SHAPES
             if args.only is None or s["name"] == args.only]
    if not specs:
        raise SystemExit(f"no circle named {args.only!r}; choose from "
                         f"{[s['name'] for s in P.EVAL_SHAPES]}")

    print(f"  {len(specs)} circle(s) at {args.speed * 1000:.0f} mm/s, "
          f"{args.image}x{args.image}, both views, clean (no disturbance)")
    print()

    started = time.perf_counter()
    rows = []
    for spec in specs:
        row = record(spec, args)
        rows.append(row)
        print(f"  {row['name']:18s} r {row['radius_m'] * 1000:3.0f} mm  "
              f"z {row['centre_m'][2] * 1000:3.0f} mm  "
              f"{row['frames']:3d} frames  "
              f"expert error mean {row['expert_error_mm']['mean_mm']:.2f} mm, "
              f"max {row['expert_error_mm']['max_mm']:.2f} mm", flush=True)

    GOAL_DIR.mkdir(parents=True, exist_ok=True)
    (GOAL_DIR / "manifest.json").write_text(json.dumps(
        {"note": "goal images for teach-and-repeat; deploy.py feeds these "
                 "as the future frames and never sees the coordinates",
         "policy_dt_s": C.POLICY_DT, "image_size": args.image,
         "speed_mps": args.speed, "circles": rows,
         "seconds": time.perf_counter() - started}, indent=2))

    print()
    print("=" * 76)
    print(f"  written to {GOAL_DIR}")
    print("=" * 76)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Record the expert tracing the evaluation circles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--only", default=None,
                   help="Record just this circle by name.")
    p.add_argument("--speed", type=float, default=P.SPEED)
    p.add_argument("--gain", type=float, default=0.12)
    p.add_argument("--image", type=int, default=112)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
