#!/usr/bin/env python3
"""
Record expert trajectory-tracking demonstrations for the VLA.

    python generate_demos.py
    python generate_demos.py --episodes 200 --workers 10

The expert is `heuristic_track.follow()`, driven through its `on_step` hook -
the control loop is not copied, so the demonstrations are by construction the
controller measured in that file.

Each episode draws a fresh smooth random path inside the measured envelope
(`training_paths.py`), rejecting anything that looks like a circle so the
held-out evaluation stays honest. At 20 Hz it records

    image      112x112 RGB from BOTH external views
    pressure   the 7-line command applied at that step (6 arm + gripper)

and nothing else. The gripper line is constant - the gripper is closed
throughout - but it is kept in the vector so the checkpoint is
interchangeable with the pick-and-place one.

WHAT MAKES THIS DATASET DIFFERENT
---------------------------------
Nothing, at recording time. The same [T, 2, 112, 112, 3] frames and [T, 7]
pressures come out. The difference is entirely in how train.py SLICES them:
a sample there takes some frames from before t (the arm's history) and some
from after t (the goal), and it is that choice the whole experiment is
about. See conditioning.py.

Disturbances are injected once per policy step exactly as in pick-and-place:
a clean demonstration never shows how to recover, so the expert is knocked
off the path and its corrections become the recovery labels.

Output: dataset/ next to this file
    episode_XXX.npy   frames [T, 2, 112, 112, 3] uint8, memory-mapped later
    episode_XXX.npz   pressure [T,7], the reference and the achieved path
    metadata.json     cameras, rates, envelope, per-episode outcome
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

# dataset/, goals/, checkpoints/, results/ live in this folder, or in
# TRACK_OUT (e.g. seeds/, the 3-seed study; relative = to this folder)
OUT_DIR = HERE / os.environ.get("TRACK_OUT", ".")
DATA_DIR = OUT_DIR / "dataset"


def run_episode(job):
    """One demonstration. Returns an outcome dict."""

    index, seed, args = job

    import mujoco
    import conditioning as C
    import heuristic_track as H
    import training_paths as P
    from sim.mujoco_env import P_MAX
    from track_scene import TrackScene, balanced_pressure
    from track_cameras import policy_cameras
    from viz.common import hide_link_capsules, style_scene

    rng = np.random.default_rng(seed)

    hargs = H.parse_args(["--no-video"])
    hargs.p0 = float(hargs.p0 + rng.uniform(-args.start_jitter,
                                            args.start_jitter))
    hargs.gain = args.gain

    scene = TrackScene(hargs.control_dt)
    scene.reset(balanced_pressure(hargs.p0))
    start = scene.tcp()

    reference, on_path, info = P.sample_path(rng, start, scene.control_dt,
                                             speed=args.speed,
                                             duration=args.duration)

    model = scene.model
    hide_link_capsules(model)
    style_scene(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.image)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.image)
    cameras, camera_cfg = policy_cameras(scene, model)
    renderer = mujoco.Renderer(model, height=args.image, width=args.image)

    every = max(1, int(round(C.POLICY_DT / scene.control_dt)))
    frames, pressures, marks = [], [], []

    def on_step(k, scene, pressure, t):
        if k % every:
            return None

        # knock the arm off the path once per policy step; the expert
        # corrects and the policy learns the correction
        if args.disturb:
            pressure = np.clip(
                pressure + rng.normal(0.0, args.disturb, pressure.shape),
                hargs.arm_p_min, P_MAX)

        views = []
        for cam in cameras:
            renderer.update_scene(scene.data, camera=cam)
            views.append(renderer.render().copy())
        frames.append(np.stack(views))
        pressures.append(np.r_[pressure, scene.grip].astype(np.float32))
        marks.append(bool(on_path[k]))

        return pressure if args.disturb else None

    log = H.follow(scene, reference, hargs, on_step=on_step)
    m = H.metrics(log, on_path)

    keep = m["mean_mm"] <= args.accept_mm
    outcome = {"index": index, "seed": int(seed),
               "mean_mm": round(m["mean_mm"], 3),
               "p95_mm": round(m["p95_mm"], 3),
               "max_mm": round(m["max_mm"], 3),
               "kept": bool(keep), "frames": len(frames),
               "path": info, "camera": camera_cfg}

    if keep:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(DATA_DIR / f"episode_{index:03d}.npy",
                np.asarray(frames, dtype=np.uint8))
        np.savez(DATA_DIR / f"episode_{index:03d}.npz",
                 pressure=np.asarray(pressures, dtype=np.float32),
                 on_path=np.asarray(marks, dtype=bool),
                 reference=log["reference"][::every].astype(np.float32),
                 tcp=log["tcp"][::every].astype(np.float32))

    return outcome


def main(args):
    print()
    print("=" * 76)
    print("TRAJECTORY-TRACKING DEMONSTRATIONS FOR THE VLA")
    print("=" * 76)

    import conditioning as C
    import training_paths as P

    print(f"  episodes  : {args.episodes} random smooth paths "
          f"(circle-like ones rejected)")
    print(f"  envelope  : r <= {P.R_MAX * 1000:.0f} mm, z "
          f"{P.Z_RANGE[0] * 1000:.0f}-{P.Z_RANGE[1] * 1000:.0f} mm "
          f"(workspace_probe.py)")
    print(f"  recording : 2 external views of {args.image}x{args.image} RGB "
          f"+ 7 pressures at {1 / C.POLICY_DT:.0f} Hz")
    print(f"  each      : {args.duration:.0f} s of tracking at "
          f"{args.speed * 1000:.0f} mm/s, gripper closed")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("episode_*.npz", "episode_*.npy"):
        for old in DATA_DIR.glob(pattern):
            old.unlink()

    started = time.perf_counter()
    jobs = [(i, args.seed + i, args) for i in range(args.episodes)]
    with ProcessPoolExecutor(args.workers,
                             mp_context=mp.get_context("spawn")) as ex:
        infos = list(ex.map(run_episode, jobs))

    kept = [i for i in infos if i["kept"]]
    frames = sum(i["frames"] for i in kept)
    errs = np.array([i["mean_mm"] for i in kept]) if kept else np.array([0.0])

    metadata = {
        "task": "trajectory tracking, hanging two-module I-Support, "
                "gripper closed throughout",
        "expert": "heuristic_track.py (driven through its on_step hook, "
                  "not copied)",
        "episodes_run": len(infos), "episodes_kept": len(kept),
        "frames_kept": frames,
        "policy_dt_s": C.POLICY_DT,
        "image_size": args.image,
        "cameras": infos[0]["camera"],
        "num_views": 2,
        "disturb_bar": args.disturb,
        "start_jitter_bar": args.start_jitter,
        "speed_mps": args.speed, "duration_s": args.duration,
        "approach_s": P.APPROACH_S,
        "envelope": {"r_max_m": P.R_MAX, "z_range_m": list(P.Z_RANGE)},
        "shape_reject": P.SHAPE_REJECT,
        "accept_mm": args.accept_mm,
        "expert_mean_mm": {"mean": float(errs.mean()),
                           "p95": float(np.percentile(errs, 95)),
                           "max": float(errs.max())},
        "inputs": "camera frames (past and/or future) + previous 7 pressures",
        "action": "next 7 pressures [A1 A2 A3 B1 B2 B3 grip], bar",
        "gripper": "held closed; the 7th line is constant",
        "seed": args.seed,
        "generation_seconds": time.perf_counter() - started,
        "episodes": infos,
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print()
    print(f"  kept {len(kept)}/{len(infos)} demos, {frames:,} frames "
          f"({frames * 2 * args.image ** 2 * 3 / 1e9:.2f} GB on disk)")
    print(f"  expert error on the path: mean {errs.mean():.2f} mm, "
          f"p95 {np.percentile(errs, 95):.2f} mm, max {errs.max():.2f} mm")
    dropped = [i for i in infos if not i["kept"]]
    if dropped:
        print(f"  discarded {len(dropped)} above {args.accept_mm} mm: "
              + ", ".join(f"#{i['index']} ({i['mean_mm']:.1f})"
                          for i in dropped[:8])
              + (" ..." if len(dropped) > 8 else ""))
    print(f"  wall clock {(time.perf_counter() - started) / 60:.1f} min")
    print(f"  written to {DATA_DIR}")
    print("=" * 76)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Expert trajectory-tracking demos for the VLA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--episodes", type=int, default=200,
                   help="Same count as the pick-and-place dataset.")
    p.add_argument("--duration", type=float, default=12.0,
                   help="Seconds of tracking per demo, after the approach.")
    p.add_argument("--speed", type=float, default=0.030,
                   help="Path speed [m/s]; the evaluation circles use the "
                        "same one.")
    p.add_argument("--gain", type=float, default=0.12)
    p.add_argument("--disturb", type=float, default=0.004,
                   help="Std of the pressure noise injected once per policy "
                        "step [bar]; the expert's corrections become the "
                        "recovery labels.")
    p.add_argument("--start-jitter", type=float, default=0.05,
                   help="Range of the random offset on the starting arm "
                        "pressure [bar].")
    p.add_argument("--accept-mm", type=float, default=5.0,
                   help="Discard a demo whose mean tracking error exceeds "
                        "this; cloning a bad trace teaches a bad policy.")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--image", type=int, default=112)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
