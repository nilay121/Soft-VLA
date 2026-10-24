#!/usr/bin/env python3
"""
Teach-and-repeat: show the policy the taught circle, ask it to trace it.

    python record_goal_trajectory.py     # first: teach
    python deploy.py                     # then: repeat
    python deploy.py --conditioning past_only --videos 0

The policy replaces the heuristic completely. Every 50 ms it is handed

    its own past frames     from its own rollout, both cameras
    the goal frames         from the taught recording, both cameras
    the 7 pressures         it commanded last step

and returns the next 7 pressures. Which of those frame groups exist depends
on the checkpoint's conditioning (see conditioning.py); the offsets are read
from the checkpoint, so a model can only ever be run the way it was trained.

WHAT THE POLICY IS NOT GIVEN
----------------------------
The reference coordinates. Not the circle's centre, not its radius, not the
tool point, not the tracking error. The scene contains no target marker by
construction. The only statement of "where to go" is a picture of the arm
somewhere it has not been yet.

The goal frames are indexed by the policy's own step counter, so a policy
that falls behind finds the goals running ahead of it - the same way a
teach-and-repeat system on real hardware behaves.

Error is still measured against the reference path, because that is the
question being asked; it just never enters the policy.

Outputs (results/): track_<cond>.json/.png, and per-circle video and CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import conditioning as C  # noqa: E402
import training_paths as P  # noqa: E402
from sim.mujoco_env import P_MAX  # noqa: E402
from track_cameras import policy_cameras  # noqa: E402
from track_scene import TrackScene, balanced_pressure  # noqa: E402
from vla_model import (  # noqa: E402
    CHUNK_EXECUTE,
    PRESSURE_LABELS,
    load_checkpoint,
    stack_frames,
)
from viz.common import (  # noqa: E402
    COLOR_REFERENCE,
    COLOR_TARGET,
    COLOR_TIP,
    COLOR_TRACE,
    VideoRecorder,
    add_geom,
    add_polyline,
    free_camera,
    hide_link_capsules,
    style_scene,
    write_csv,
    write_json,
)

# Thicker and brighter than the shared defaults: these two lines ARE the
# result, and at this zoom the stock 1.6 mm ribbons were hard to separate.
REF_BLUE = np.array([0.20, 0.65, 1.00, 1.0], dtype=np.float32)
TRACE_RED = np.array([1.00, 0.20, 0.15, 1.0], dtype=np.float32)

# dataset/, goals/, checkpoints/, results/ live in this folder, or in
# TRACK_OUT (e.g. seeds/, the 3-seed study; relative = to this folder)
OUT_DIR = HERE / os.environ.get("TRACK_OUT", ".")
RESULTS_DIR = OUT_DIR / "results"
GOAL_DIR = OUT_DIR / "goals"
CKPT_DIR = OUT_DIR / "checkpoints"


def run_circle(net, ckpt, device, spec, args, video_path=None, csv_path=None):
    name = spec["name"]
    goal_frames = np.load(GOAL_DIR / f"{name}.npy", mmap_mode="r")
    taught = np.load(GOAL_DIR / f"{name}.npz")
    reference = taught["reference"]          # [T,3] at policy rate
    on_shape = taught["on_shape"]
    n_steps = len(goal_frames)

    offsets = list(ckpt["frame_offsets"])
    n_past = int(ckpt["n_past"])

    scene = TrackScene(args.control_dt)
    arm = balanced_pressure(args.p0)
    scene.reset(arm)

    model, data = scene.model, scene.data
    hide_link_capsules(model)
    style_scene(model)
    cameras, _ = policy_cameras(scene, model)
    image_size = int(ckpt.get("image_size", args.image))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, image_size,
                                     args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, image_size,
                                      args.height)
    renderer = mujoco.Renderer(model, height=image_size, width=image_size)

    view = free_camera(model, lookat=[0.0, 0.0, args.cam_lookat_z],
                       distance=args.cam_distance, azimuth=args.cam_azimuth,
                       elevation=args.cam_elevation)
    video = VideoRecorder(model, video_path or HERE / "unused.mp4", args.width,
                          args.height, args.fps, view, shell_alpha=1.0,
                          hide_links=True, enabled=video_path is not None)

    every = max(1, int(round(C.POLICY_DT / scene.control_dt)))
    render_every = max(1, int(round((1.0 / args.fps) / C.POLICY_DT)))

    command = np.r_[arm, scene.grip].astype(np.float32)
    history = deque(maxlen=max(1, (n_past - 1) * C.PAST_STRIDE + 1))
    trace, rows, errors = [], [], []
    policy_seconds = 0.0
    started = time.perf_counter()
    chunk, chunk_at = None, 0

    for k in range(n_steps):
        views = []
        for cam in cameras:
            renderer.update_scene(data, camera=cam)
            views.append(renderer.render().copy())
        frame = np.stack(views)
        if not history:
            history.extend([frame] * history.maxlen)
        history.append(frame)

        if chunk is None or chunk_at >= min(CHUNK_EXECUTE, len(chunk)):
            picks = []
            for o in offsets:
                if o <= 0:
                    # the policy's OWN past, one frame per policy step
                    picks.append(history[max(0, len(history) - 1 + o)])
                else:
                    # the TAUGHT future, clamped at the end of the recording
                    picks.append(np.asarray(goal_frames[min(k + o,
                                                            n_steps - 1)]))
            images = torch.from_numpy(stack_frames(picks)
                                      ).unsqueeze(0).to(device)
            tick = time.perf_counter()
            with torch.no_grad():
                out = net.pressure_chunk(
                    images, torch.from_numpy(command).unsqueeze(0).to(device))
            policy_seconds += time.perf_counter() - tick
            chunk, chunk_at = out.cpu().numpy()[0], 0

        target = chunk[chunk_at]
        chunk_at += 1
        step_limit = np.r_[[args.max_rate * C.POLICY_DT] * 6,
                           args.grip_rate * C.POLICY_DT]
        command = (command + np.clip(target - command, -step_limit,
                                     step_limit)).astype(np.float32)
        command[:6] = np.clip(command[:6], args.arm_p_min, P_MAX)
        command[6] = float(scene.grip)          # the gripper stays closed

        scene.apply(command[:6])
        for _ in range(every):
            scene.step()

        tcp = scene.tcp()
        trace.append(tcp.copy())
        err = float(np.linalg.norm(reference[k] - tcp))
        errors.append(err)

        row = {"time_s": k * C.POLICY_DT,
               "ref_x_m": reference[k][0], "ref_y_m": reference[k][1],
               "ref_z_m": reference[k][2],
               "tcp_x_m": tcp[0], "tcp_y_m": tcp[1], "tcp_z_m": tcp[2],
               "error_mm": err * 1000, "on_shape": int(on_shape[k])}
        row.update({f"p_{n}_bar": v for n, v in zip(PRESSURE_LABELS, command)})
        rows.append(row)

        if video.enabled and k % render_every == 0:
            def decorate(scn, k=k):
                add_polyline(scn, reference[on_shape], 0.0024, REF_BLUE)
                add_polyline(scn, trace, 0.0020, TRACE_RED)
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0055] * 3,
                         reference[k], COLOR_TARGET)
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.0045] * 3,
                         trace[-1], COLOR_TIP)

            video.capture(data, decorate, hud=[
                ("Hanging two-module I-Support, gripper closed   "
                 f"VLA teach-and-repeat: {name}", (255, 255, 255)),
                (f"t = {k * C.POLICY_DT:5.2f} s     "
                 f"error {err * 1000:6.2f} mm", (120, 220, 120)),
                (f"arm [{' '.join(f'{v:.2f}' for v in command[:6])}] bar",
                 (200, 210, 255)),
                (f"conditioning: {ckpt['conditioning_label']}",
                 (255, 210, 150)),
                ("the policy sees goal IMAGES only - no coordinates, no "
                 "marker in the scene", (200, 200, 200)),
                ("blue = reference   red = tool point path",
                 (200, 200, 200)),
            ])

    video.close()
    if csv_path is not None:
        write_csv(csv_path, rows)

    err_mm = np.array(errors) * 1000
    on = np.asarray(on_shape, dtype=bool)
    return {
        "name": name, "radius_m": spec["radius"],
        "centre_m": list(spec["centre"]),
        "steps": n_steps,
        "mean_mm": float(err_mm[on].mean()),
        "rmse_mm": float(np.sqrt((err_mm[on] ** 2).mean())),
        "p95_mm": float(np.percentile(err_mm[on], 95)),
        "max_mm": float(err_mm[on].max()),
        "final_mm": float(err_mm[-1]),
        "error_pct_of_radius": float(100 * err_mm[on].mean()
                                     / (spec["radius"] * 1000)),
        "policy_ms_per_step": round(1000 * policy_seconds / n_steps, 2),
        "wall_clock_s": round(time.perf_counter() - started, 1),
    }


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    circles = results["circles"]
    names = [c["name"] for c in circles]
    x = np.arange(len(circles))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    axes[0].bar(x - 0.2, [c["mean_mm"] for c in circles], 0.4,
                label="VLA mean", color="#2f6fb0", edgecolor="black", lw=0.5)
    axes[0].bar(x + 0.2, [c["expert_mean_mm"] for c in circles], 0.4,
                label="expert mean", color="#2f9e44", edgecolor="black", lw=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=7,
                                                   rotation=15)
    axes[0].set_ylabel("tracking error [mm]")
    axes[0].set_title("VLA against the expert it cloned")
    axes[0].grid(axis="y", alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].bar(x, [c["error_pct_of_radius"] for c in circles], 0.5,
                color="#1f9e89", edgecolor="black", lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=7,
                                                   rotation=15)
    axes[1].set_ylabel("mean error [% of circle radius]")
    axes[1].set_title("Error relative to the size of the circle")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Trajectory tracking by teach-and-repeat — "
                 f"{results['conditioning_label']}  "
                 f"(mean {results['summary']['mean_mm']:.1f} mm over "
                 f"{len(circles)} circles)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(args):
    print()
    print("=" * 78)
    print("TRAJECTORY-TRACKING VLA — TEACH AND REPEAT")
    print("=" * 78)

    ckpt_path = (Path(args.ckpt) if args.ckpt else
                 CKPT_DIR / f"track_vla_{args.conditioning}.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ckpt = load_checkpoint(ckpt_path, device)

    if not (GOAL_DIR / "manifest.json").exists():
        raise SystemExit(f"no taught circles in {GOAL_DIR} - "
                         "run record_goal_trajectory.py first")
    manifest = json.loads((GOAL_DIR / "manifest.json").read_text())
    taught = {c["name"]: c for c in manifest["circles"]}

    specs = [s for s in P.EVAL_SHAPES if s["name"] in taught]
    if args.only:
        specs = [s for s in specs if s["name"] == args.only]
    if not specs:
        raise SystemExit("no taught circle matches - re-run "
                         "record_goal_trajectory.py")

    print(f"  policy     : {ckpt_path.name} ({ckpt.get('arch')}, epoch "
          f"{ckpt.get('epoch')}, val MAE "
          f"{ckpt.get('val_mae_bar', float('nan')):.4f} bar)")
    print(f"  condition  : {ckpt['conditioning_label']}")
    print(f"  frames     : offsets {ckpt['frame_offsets']} policy steps "
          f"x 2 views")
    print(f"  circles    : {len(specs)} taught traces from goals/")
    print(f"  given      : goal IMAGES only - no coordinates, no marker")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, spec in enumerate(specs):
        stem = f"{args.tag or 'track'}_{args.conditioning}_{spec['name']}"
        video = RESULTS_DIR / f"{stem}.mp4" if i < args.videos else None
        csv = RESULTS_DIR / f"{stem}.csv" if i < args.videos else None
        row = run_circle(net, ckpt, device, spec, args, video, csv)
        row["expert_mean_mm"] = taught[spec["name"]]["expert_error_mm"]["mean_mm"]
        rows.append(row)
        print(f"  {row['name']:18s} VLA mean {row['mean_mm']:6.2f} mm  "
              f"p95 {row['p95_mm']:6.2f}  max {row['max_mm']:6.2f}  "
              f"({row['error_pct_of_radius']:4.1f}% of r)   "
              f"expert {row['expert_mean_mm']:.2f} mm", flush=True)

    mean_mm = float(np.mean([r["mean_mm"] for r in rows]))
    results = {
        "task": "trajectory tracking by teach-and-repeat, hanging "
                "two-module I-Support, gripper closed",
        "controller": "vision VLA: two external views (own past + taught "
                      "future) + previous 7 pressures -> next 7 pressures",
        "checkpoint": str(ckpt_path),
        "conditioning": ckpt["conditioning"],
        "conditioning_label": ckpt["conditioning_label"],
        "frame_offsets": list(ckpt["frame_offsets"]),
        "given_to_policy": "goal images only; no reference coordinates and "
                           "no target marker exists in the scene",
        "summary": {
            "mean_mm": mean_mm,
            "expert_mean_mm": float(np.mean([r["expert_mean_mm"]
                                             for r in rows])),
            "circles": len(rows),
        },
        "settings": {k: getattr(args, k) for k in
                     ("p0", "arm_p_min", "max_rate", "control_dt")},
        "circles": rows,
    }
    stem = f"{args.tag or 'track'}_{args.conditioning}"
    write_json(RESULTS_DIR / f"{stem}.json", results)
    plot(results, RESULTS_DIR / f"{stem}.png")

    print()
    print("=" * 78)
    print(f"  VLA mean tracking error {mean_mm:.2f} mm over "
          f"{len(rows)} circles   (expert "
          f"{results['summary']['expert_mean_mm']:.2f} mm)")
    print(f"  outputs in {RESULTS_DIR}")
    print("=" * 78)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the tracking VLA on the taught circles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--conditioning", default=C.DEFAULT_VARIANT,
                   choices=tuple(C.VARIANTS))
    p.add_argument("--ckpt", default=None,
                   help="Override the checkpoint path.")
    p.add_argument("--only", default=None, help="Just this circle.")
    p.add_argument("--tag", default=None)
    p.add_argument("--videos", type=int, default=2,
                   help="Record video + CSV for the first N circles.")

    p.add_argument("--p0", type=float, default=0.4)
    p.add_argument("--arm-p-min", type=float, default=0.25)
    p.add_argument("--max-rate", type=float, default=1.2)
    p.add_argument("--grip-rate", type=float, default=2.0)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--image", type=int, default=112)

    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--cam-distance", type=float, default=0.28,
                   help="Close enough that the traced shape fills a useful\n                        part of the frame; at 0.75 it was a few pixels.")
    p.add_argument("--cam-azimuth", type=float, default=250.0)
    p.add_argument("--cam-elevation", type=float, default=-45.0,
                   help="Looking down at the trace plane. Near side-on (-12) "
                        "flattens a horizontal path into a line.")
    p.add_argument("--cam-lookat-z", type=float, default=0.396)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
