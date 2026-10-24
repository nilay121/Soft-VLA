#!/usr/bin/env python3
"""
Heuristic trajectory tracking for the hanging two-module arm.

    python heuristic_track.py --shape circle
    python heuristic_track.py --shape circle --radius 0.045 --no-video

NO learned models. This is the same Jacobian + damped least-norm controller
used everywhere else in this repo (robot.JacobianCorrector), driving the
gripper's tool point along a moving reference instead of to a fixed target.

The gripper is held CLOSED for the whole run, so the end effector is a
compact fist and the 3x6 Jacobian always sees the same tool.

    reference[k]  ->  error = ref[k + lookahead] - tcp
                  ->  dp = gain * J^T (J J^T + lambda I)^-1 error
                  ->  rate-limited, clipped to [arm_p_min, P_MAX]

This module is the expert for the VLA: `follow()` is imported by
generate_demos.py and by workspace_probe.py rather than being copied, so
the demonstrations are by construction the controller measured here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # this experiment
SIMULATOR = HERE.parent.parent / "simulator"      # ../../simulator
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import trajectories as T  # noqa: E402
from robot import CHAMBER_LABELS, JacobianCorrector, module_slice  # noqa: E402
from sim.mujoco_env import P_MAX  # noqa: E402
from track_scene import TrackScene, balanced_pressure  # noqa: E402

RESULTS_DIR = HERE / "results"


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def reference_for(shape, centre, scale, speed, control_dt, approach_s, start):
    """`trajectories.make_reference`, but the approach starts where the arm
    really is rather than at the shape's centre."""

    path, on_shape = T.make_reference(shape, centre, scale, speed,
                                      control_dt, approach_s)
    n_app = int(round(approach_s / control_dt))
    if n_app:
        s = np.linspace(0.0, 1.0, n_app, endpoint=False)
        s = s * s * (3.0 - 2.0 * s)                            # smoothstep
        start = np.asarray(start, dtype=np.float64)
        path[:n_app] = start[None] + s[:, None] * (path[n_app] - start)[None]
    return path, on_shape


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------

def follow(scene, reference, args, on_step=None, corrector=None):
    """Track `reference` [N,3] one control step per row.

    `on_step(k, scene, pressure, t)` is called just before the command
    reaches the simulator, on exactly the state the expert acted on. It may
    return a replacement pressure (the demo recorder injects disturbances
    that way) or None.

    Returns a dict of per-step logs.
    """

    reference = np.asarray(reference, dtype=np.float64)
    dt = scene.control_dt
    corrector = corrector or JacobianCorrector(
        scene.model, scene.mapper, scene.tcp_id,
        refresh_dt=args.jacobian_refresh)

    pressure = balanced_pressure(args.p0)
    scene.reset(pressure)

    n = len(reference)
    tcp_log = np.empty((n, 3))
    ref_log = np.empty((n, 3))
    press_log = np.empty((n, 7), dtype=np.float32)
    err_log = np.empty(n)
    t = 0.0

    for k in range(n):
        ahead = reference[min(k + args.lookahead, n - 1)]
        error = ahead - scene.tcp()

        desired = args.gain * corrector.correction(scene.data, pressure,
                                                   error, t)
        command = np.clip(pressure + desired, args.arm_p_min, P_MAX)
        step = np.clip(command - pressure, -args.max_rate * dt,
                       args.max_rate * dt)
        pressure = np.clip(pressure + step, args.arm_p_min, P_MAX)

        if on_step is not None:
            replacement = on_step(k, scene, pressure, t)
            if replacement is not None:
                pressure = np.asarray(replacement, dtype=np.float64)

        scene.apply(pressure)
        scene.step()

        tcp = scene.tcp()
        tcp_log[k] = tcp
        ref_log[k] = reference[k]
        press_log[k] = np.r_[pressure, scene.grip]
        err_log[k] = np.linalg.norm(reference[k] - tcp)
        t += dt

    return {"tcp": tcp_log, "reference": ref_log, "pressure": press_log,
            "error_m": err_log, "seconds": t}


def metrics(log, on_shape=None):
    """Tracking error in mm, over the shape only if `on_shape` is given."""

    err = log["error_m"] * 1000.0
    if on_shape is not None:
        err = err[np.asarray(on_shape, dtype=bool)]
    return {"mean_mm": float(err.mean()), "rmse_mm": float(np.sqrt((err ** 2).mean())),
            "p95_mm": float(np.percentile(err, 95)), "max_mm": float(err.max()),
            "final_mm": float(err[-1])}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(args):
    print()
    print("=" * 78)
    print("HANGING TWO-MODULE I-SUPPORT — HEURISTIC TRAJECTORY TRACKING")
    print("=" * 78)

    scene = TrackScene(args.control_dt)
    scene.reset(balanced_pressure(args.p0))
    start = scene.tcp()

    centre = np.array(args.centre if args.centre else
                      [0.0, 0.0, float(start[2])])
    reference, on_shape = reference_for(args.shape, centre, args.radius,
                                        args.speed, scene.control_dt,
                                        args.approach, start)

    print(f"  shape      : {args.shape}, radius {args.radius * 1000:.0f} mm "
          f"at {args.speed * 1000:.0f} mm/s")
    print(f"  centre     : {np.round(centre * 1000, 1).tolist()} mm")
    print(f"  TCP start  : {np.round(start * 1000, 1).tolist()} mm")
    print(f"  gripper    : closed at {scene.grip:.2f} bar throughout")
    print(f"  steps      : {len(reference)} "
          f"({len(reference) * scene.control_dt:.1f} s)")
    print()

    started = time.perf_counter()
    log = follow(scene, reference, args)
    m = metrics(log, on_shape)

    print(f"  tracking error over the shape: mean {m['mean_mm']:.2f} mm, "
          f"rmse {m['rmse_mm']:.2f}, p95 {m['p95_mm']:.2f}, "
          f"max {m['max_mm']:.2f}")
    print(f"  wall clock {time.perf_counter() - started:.1f} s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        record_video(scene, log, on_shape, args)
    plot(log, on_shape, m, args, RESULTS_DIR / f"{args.tag}.png")

    from viz.common import write_json
    write_json(RESULTS_DIR / f"{args.tag}.json", {
        "task": "trajectory tracking, hanging two-module I-Support, "
                "gripper closed",
        "controller": "heuristic: 3x6 Jacobian damped least-norm on the TCP",
        "shape": args.shape, "radius_m": args.radius, "speed_mps": args.speed,
        "centre_m": centre.tolist(),
        "metrics_on_shape": m,
        "settings": {k: getattr(args, k) for k in
                     ("p0", "gain", "max_rate", "lookahead", "arm_p_min",
                      "approach", "control_dt", "jacobian_refresh")},
    })
    print(f"  outputs in {RESULTS_DIR}")
    print("=" * 78)
    return log, on_shape, m


def record_video(scene, log, on_shape, args):
    """Re-run the trajectory with the recorder attached.

    The reference is drawn here, at render time, on the human-facing camera
    only - it is never in the scene and never in what a policy would see.
    """

    import mujoco
    from viz.common import (COLOR_REFERENCE, COLOR_TARGET, COLOR_TIP,
                            COLOR_TRACE, VideoRecorder, add_geom,
                            add_polyline, free_camera)

    camera = free_camera(scene.model, lookat=[0.0, 0.0, args.cam_lookat_z],
                         distance=args.cam_distance,
                         azimuth=args.cam_azimuth,
                         elevation=args.cam_elevation)
    video = VideoRecorder(scene.model, RESULTS_DIR / f"{args.tag}.mp4",
                          args.width, args.height, args.fps, camera,
                          shell_alpha=1.0, hide_links=True, enabled=True)

    reference = log["reference"]
    render_every = max(1, int(round((1.0 / args.fps) / scene.control_dt)))
    trace = []

    scene.reset(balanced_pressure(args.p0))
    for k in range(len(reference)):
        # replay the recorded commands: identical motion, no re-solving
        scene.apply(log["pressure"][k][:6])
        scene.step()
        trace.append(scene.tcp().copy())

        if k % render_every == 0:
            err_mm = log["error_m"][k] * 1000

            def decorate(scn, k=k, ref=reference[k]):
                add_polyline(scn, reference[on_shape], 0.0024,
                             COLOR_REFERENCE)
                add_polyline(scn, trace, 0.0020, COLOR_TRACE)
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.005] * 3,
                         ref, COLOR_TARGET)
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.004] * 3,
                         trace[-1], COLOR_TIP)

            video.capture(scene.data, decorate, hud=[
                ("Hanging two-module I-Support, gripper closed   "
                 f"heuristic trajectory tracking: {args.shape.upper()}",
                 (255, 255, 255)),
                (f"t = {k * scene.control_dt:5.2f} s     "
                 f"error {err_mm:5.2f} mm", (120, 220, 120)),
                (f"arm [{' '.join(f'{v:.2f}' for v in log['pressure'][k][:6])}]"
                 " bar", (200, 210, 255)),
                ("blue = reference   red = tool point path",
                 (200, 200, 200)),
            ])
    video.close()


def plot(log, on_shape, m, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tcp = log["tcp"] * 1000
    ref = log["reference"] * 1000
    err = log["error_m"] * 1000
    t = np.arange(len(err)) * args.control_dt
    on = np.asarray(on_shape, dtype=bool)

    fig = plt.figure(figsize=(16, 4.4))

    a = fig.add_subplot(1, 3, 1)
    a.plot(ref[on, 0], ref[on, 1], lw=2.4, color="#2f8bfd", alpha=0.45,
           label="reference")
    a.plot(tcp[on, 0], tcp[on, 1], lw=1.3, color="#e8422f", label="TCP")
    a.plot(tcp[~on, 0], tcp[~on, 1], lw=0.9, color="#999999", label="approach")
    a.set_xlabel("X [mm]"); a.set_ylabel("Y [mm]"); a.set_aspect("equal")
    a.set_title(f"{args.shape} in the XY plane"); a.grid(alpha=0.3)
    a.legend(fontsize=7)

    b = fig.add_subplot(1, 3, 2)
    b.plot(t, ref[:, 2], lw=2.4, color="#2f8bfd", alpha=0.45, label="reference Z")
    b.plot(t, tcp[:, 2], lw=1.3, color="#e8422f", label="TCP Z")
    b.set_xlabel("time [s]"); b.set_ylabel("height Z [mm]")
    b.set_title("Height"); b.grid(alpha=0.3); b.legend(fontsize=7)

    c = fig.add_subplot(1, 3, 3)
    c.plot(t, err, lw=1.1, color="#e8422f")
    c.axvspan(t[on][0], t[on][-1], color="#2f9e44", alpha=0.08,
              label="on the shape")
    c.axhline(m["mean_mm"], ls="--", lw=1.0, color="black",
              label=f"mean {m['mean_mm']:.2f} mm")
    c.set_xlabel("time [s]"); c.set_ylabel("tracking error [mm]")
    c.set_title("Error"); c.grid(alpha=0.3); c.legend(fontsize=7)

    fig.suptitle(f"Heuristic tracking, {args.shape} r={args.radius * 1000:.0f} mm "
                 f"at {args.speed * 1000:.0f} mm/s — mean {m['mean_mm']:.2f} mm, "
                 f"p95 {m['p95_mm']:.2f} mm")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Heuristic trajectory tracking, hanging arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--shape", default="circle", choices=T.TEST_SHAPES)
    p.add_argument("--radius", type=float, default=0.040,
                   help="Scale of the shape [m]; its farthest point sits "
                        "this far from the centre.")
    p.add_argument("--speed", type=float, default=0.030,
                   help="Path speed [m/s].")
    p.add_argument("--centre", type=float, nargs=3, default=None,
                   help="Shape centre [m]; default is straight below the "
                        "mount at the arm's resting height.")
    p.add_argument("--approach", type=float, default=2.0,
                   help="Smooth move from the start pose onto the shape [s].")

    p.add_argument("--p0", type=float, default=0.4)
    p.add_argument("--arm-p-min", type=float, default=0.25,
                   help="Floor on every arm pressure, above the 0.2 bar "
                        "strain threshold of sim/kinematics.py.")
    p.add_argument("--gain", type=float, default=0.12)
    p.add_argument("--max-rate", type=float, default=1.2)
    p.add_argument("--lookahead", type=int, default=4)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--jacobian-refresh", type=float, default=0.12)

    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--cam-distance", type=float, default=0.28)
    p.add_argument("--cam-azimuth", type=float, default=250.0)
    p.add_argument("--cam-elevation", type=float, default=-45.0)
    p.add_argument("--cam-lookat-z", type=float, default=0.396)

    args = p.parse_args(argv)
    args.tag = args.tag or f"track_{args.shape}"
    return args


if __name__ == "__main__":
    run(parse_args())
