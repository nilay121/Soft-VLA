#!/usr/bin/env python3
"""
Heuristic PICK AND DROP: hanging two-module arm, soft gripper, bowl.

    python .../bowl_drop/heuristic_pick_drop.py
    python .../bowl_drop/heuristic_pick_drop.py --object banana --no-video

NO learned models. The arm is the same Jacobian + damped least-norm
controller as ../heuristic_pick_place.py (robot.py), driving the gripper's
tool point; the gripper is one pressure line, ramped to close and vented to
open.

WHY THIS IS SHORTER THAN PICK AND PLACE
---------------------------------------
Putting an object down on a marker needs the set-down guarded (stop as soon
as it sits flat), a two-stage release (cage, then vent) and a re-centring
move so an opening finger does not drag it. Three of the ten phases, and
most of the rules in ../README.md, exist only to land the object gently on
a flat table within 15 mm.

Dropping it into a bowl needs none of that. The bowl's 45 mm mouth catches
anything released near its axis and its flared wall funnels the object to
the floor, so the phases collapse to:

    approach   above the object
    descend    open fingers around it
    grasp      pressurise the gripper line, arm holds still
    lift       straight up
    transport  over the bowl, object in hand, high enough to clear the rim
    release    vent the line; the object falls in
    settle     hold while it lands
    retreat    rise clear

The one height that matters is the transport height. The object hangs a
measured distance below the tool point, so the controller reads that offset
during the carry and asks for

    TCP  =  rim + clearance + (object centre above its base) - offset

clipped into the arm's reachable band. Nothing is tuned per object.

Success = the object ends up inside the bowl: below the rim plane and
within the rim radius. Whether it has stopped moving is logged as `settled`
but is not required - a 68 mm bowl does not give anything back.

Outputs (results/): pick_drop_<object>.mp4 / .png / .csv / .json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent            # this experiment
SIMULATOR = HERE.parent.parent / "simulator"      # ../../simulator
# HERE goes in last so it wins: the experiment's own modules must never be
# shadowed by a same-named module in the simulator.
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from bowl_scene import BowlScene, balanced_pressure, sample_layout  # noqa: E402
import build_bowl_scene as BS  # noqa: E402
from hanging_robot import G  # noqa: E402
from robot import CHAMBER_LABELS, JacobianCorrector, module_slice  # noqa: E402
from sim.mujoco_env import P_MAX  # noqa: E402

from viz.common import (  # noqa: E402
    COLOR_REFERENCE,
    COLOR_TARGET,
    COLOR_TIP,
    COLOR_TRACE,
    VideoRecorder,
    add_geom,
    add_polyline,
    free_camera,
    write_csv,
    write_json,
)

RESULTS_DIR = HERE / "results"

PHASE_COLOURS = {"approach": "#2f6fb0", "descend": "#1f9e89",
                 "grasp": "#e8a33f", "lift": "#8e5bb5",
                 "transport": "#d9534f", "release": "#e8a33f",
                 "settle": "#f0c060", "retreat": "#2f6fb0"}

HOLD_PHASES = ("grasp", "release", "settle")     # arm frozen, fingers move
CARRY_PHASES = ("lift", "transport")             # object in hand: gentler gain
TRACKING_PHASES = ("transport",)                 # target follows the object


def min_jerk(s):
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def drop_height(scene, args):
    """TCP height that carries the object clear of the bowl rim.

    `offset_z` is negative: the object hangs below the tool point. Adding
    the object's own half-height puts its BASE above the rim.
    """

    offset_z = float(scene.object_pos()[2] - scene.tcp()[2])
    need = (scene.bowl_centre()[2] + args.rim_clearance
            + scene.obj["rest"] - offset_z)
    return float(np.clip(need, *args.z_band))


def over_bowl(scene, args, z=None):
    """TCP target that puts the OBJECT, not the tool point, on the bowl axis."""

    offset = scene.object_pos() - scene.tcp()
    centre = scene.bowl_centre()
    z = drop_height(scene, args) if z is None else z
    return np.array([centre[0] - offset[0], centre[1] - offset[1], z])


def plan(scene, args):
    """Phase list: (name, start TCP, end TCP, duration, grip start, end).

    None = decided when the phase begins, from where the TCP and the object
    really are then.
    """

    obj = scene.object_pos()
    tcp0 = scene.tcp()
    drop = G.open_tip_drop()

    # Open fingertips start at the object's centre height (never below just
    # above the table): starting under a round object's equator squeezes it
    # up into the hand. Hover clears the object's own top.
    grasp_z = max(scene.table_top + args.tip_clearance, scene.rest_z()) + drop
    hover_z = max(grasp_z + args.hover,
                  scene.table_top + scene.obj["top"] + drop
                  + args.top_clearance)
    grasp_z = float(np.clip(grasp_z, *args.z_band))
    hover_z = float(np.clip(hover_z, *args.z_band))

    above_obj = np.array([obj[0], obj[1], hover_z])
    at_obj = np.array([obj[0], obj[1], grasp_z])

    g = args.grip_pressure
    return [
        ("approach", tcp0, above_obj, args.move_time, 0.0, 0.0),
        ("descend", above_obj, at_obj, args.vertical_time, 0.0, 0.0),
        ("grasp", None, None, args.grip_time, 0.0, g),
        ("lift", None, above_obj, args.vertical_time, g, g),
        ("transport", None, None, args.move_time, g, g),
        ("release", None, None, args.release_time, g, 0.0),
        ("settle", None, None, args.settle_time, 0.0, 0.0),
        ("retreat", None, None, args.vertical_time, 0.0, 0.0),
    ]


def resolve(name, start, end, scene, args):
    """Fill in the targets decided at phase start (see plan())."""

    tcp = scene.tcp()
    start = tcp if start is None else start

    if name in HOLD_PHASES:
        end = tcp
    elif name == "transport":
        end = over_bowl(scene, args)
    elif name == "retreat":
        end = np.array([tcp[0], tcp[1],
                        float(np.clip(tcp[2] + args.hover, *args.z_band))])

    return start, end


def run(args, snapshots=None, on_step=None):
    """Run one episode of the task.

    `snapshots`  if a list, (phase, qpos) is appended at the end of every
                 phase - the camera study poses the arm from those.
    `on_step`    called as f(step, scene, pressure, grip, t) just before the
                 command reaches the simulator, i.e. on exactly the state
                 the expert acted on. Return an array to REPLACE the arm
                 pressure (the demo recorder injects disturbances that way),
                 or None to leave it alone.

    Both hooks exist so VLA_control/generate_demos.py can record and disturb
    this controller rather than keeping its own copy of the loop, which is
    what keeps the demonstrations identical to the behaviour measured here.
    """

    print()
    print("=" * 78)
    print("HANGING TWO-MODULE I-SUPPORT — HEURISTIC PICK AND DROP INTO A BOWL")
    print("=" * 78)

    scene = BowlScene(args.control_dt)
    model, data = scene.model, scene.data
    dt = scene.control_dt

    if args.object_xy is None or args.bowl_xy is None:
        sampled_obj, sampled_bowl = sample_layout(
            np.random.default_rng(args.seed))
        object_xy = args.object_xy if args.object_xy is not None else sampled_obj
        bowl_xy = args.bowl_xy if args.bowl_xy is not None else sampled_bowl
    else:
        object_xy, bowl_xy = args.object_xy, args.bowl_xy

    scene.set_layout(args.object, object_xy, bowl_xy)
    pressure = balanced_pressure(args.p0)
    scene.reset(pressure, 0.0)

    obj0 = scene.object_pos()
    bowl = scene.bowl_centre()
    phases = plan(scene, args)

    print(f"  arm        : 6 pressure lines, start {args.p0} bar (balanced)")
    print(f"  gripper    : {G.NUM_FINGERS} PneuNet fingers, 1 pressure line, "
          f"grip at {args.grip_pressure} bar")
    print(f"  TCP start  : {np.round(scene.tcp() * 1000, 1).tolist()} mm")
    print(f"  object     : {args.object} at "
          f"{np.round(obj0 * 1000, 1).tolist()} mm")
    print(f"  bowl       : rim centre {np.round(bowl * 1000, 1).tolist()} mm, "
          f"mouth radius {BS.BOWL_R_RIM * 1000:.0f} mm, "
          f"{BS.BOWL_DEPTH * 1000:.0f} mm deep")
    print()

    corrector = JacobianCorrector(model, scene.mapper, scene.tcp_id,
                                  refresh_dt=args.jacobian_refresh)

    camera = free_camera(model, lookat=[0.02, 0.0, scene.table_top + 0.06],
                         distance=args.cam_distance, azimuth=args.cam_azimuth,
                         elevation=args.cam_elevation)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    video = VideoRecorder(model, RESULTS_DIR / f"{args.tag}.mp4", args.width,
                          args.height, args.fps, camera, shell_alpha=1.0,
                          hide_links=True, enabled=not args.no_video)
    render_every = max(1, int(round((1.0 / args.fps) / dt)))

    rows, tcp_trace, obj_trace = [], [], []
    phase_log = []
    t = 0.0
    step_count = 0
    started = time.perf_counter()

    for name, p_start, nominal_end, duration, g_start, g_end in phases:
        p_start, p_end = resolve(name, p_start, nominal_end, scene, args)
        n = int(round(duration / dt))
        settle_n = int(round(args.hold / dt))
        errors = []
        held = name in HOLD_PHASES

        for k in range(n + settle_n):
            if name in TRACKING_PHASES:
                # the object keeps shifting in the grip while it is carried,
                # so re-aim every step: the OBJECT must end up on the axis
                p_end = over_bowl(scene, args, z=p_end[2])

            s = min(k, n) / max(n, 1)
            s_ahead = min(k + args.lookahead, n) / max(n, 1)
            ref = p_start + (p_end - p_start) * min_jerk(s)
            ref_ahead = p_start + (p_end - p_start) * min_jerk(s_ahead)
            grip = g_start + (g_end - g_start) * s

            tcp = scene.tcp()
            error = ref_ahead - tcp

            scene.mapper.gripper_pressure = grip
            if not held:
                gain = args.carry_gain if name in CARRY_PHASES else args.gain
                desired = gain * corrector.correction(data, pressure, error, t)
                command = np.clip(pressure + desired, args.arm_p_min, P_MAX)
                step = np.clip(command - pressure,
                               -args.max_rate * dt, args.max_rate * dt)
                pressure = np.clip(pressure + step, args.arm_p_min, P_MAX)

            if on_step is not None:
                replacement = on_step(step_count, scene, pressure, grip, t)
                if replacement is not None:
                    pressure = np.asarray(replacement, dtype=np.float64)

            scene.apply(pressure, grip)
            scene.step()

            tcp = scene.tcp()
            obj = scene.object_pos()
            err_mm = float(np.linalg.norm(ref - tcp) * 1000)
            errors.append(err_mm)
            tcp_trace.append(tcp)
            obj_trace.append(obj)

            row = {"time_s": t, "phase": name,
                   "ref_x_m": ref[0], "ref_y_m": ref[1], "ref_z_m": ref[2],
                   "tcp_x_m": tcp[0], "tcp_y_m": tcp[1], "tcp_z_m": tcp[2],
                   "obj_x_m": obj[0], "obj_y_m": obj[1], "obj_z_m": obj[2],
                   "tcp_error_mm": err_mm, "gripper_bar": grip,
                   "in_bowl": int(scene.object_in_bowl()),
                   "finger_contacts": scene.finger_contacts()}
            for i, label in enumerate(CHAMBER_LABELS):
                row[f"p_{label}_bar"] = pressure[i]
            rows.append(row)

            if video.enabled and step_count % render_every == 0:
                def decorate(scn, ref=ref, tcp=tcp, bowl=bowl):
                    add_polyline(scn, tcp_trace, 0.0018, COLOR_TRACE)
                    add_polyline(scn, obj_trace, 0.0018, COLOR_REFERENCE)
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.006] * 3,
                             ref, COLOR_TARGET)
                    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.004] * 3,
                             tcp, COLOR_TIP)

                pa, pb = pressure[module_slice(0)], pressure[module_slice(1)]
                video.capture(data, decorate, hud=[
                    ("Hanging two-module I-Support + soft gripper   "
                     f"heuristic PICK AND DROP: {args.object.upper()} "
                     "-> bowl", (255, 255, 255)),
                    (f"phase: {name.upper():10s}  t = {t:5.2f} s   "
                     f"TCP error {err_mm:5.1f} mm", (120, 220, 120)),
                    (f"module A [{pa[0]:.2f} {pa[1]:.2f} {pa[2]:.2f}]  "
                     f"module B [{pb[0]:.2f} {pb[1]:.2f} {pb[2]:.2f}] bar",
                     (200, 210, 255)),
                    (f"gripper line {grip:.2f} bar   fingers touching "
                     f"{args.object}: {row['finger_contacts']}",
                     (255, 210, 150)),
                    (f"{args.object} {(obj[2] - obj0[2]) * 1000:+6.1f} mm vs "
                     f"start   {scene.bowl_offset_mm():5.1f} mm from the bowl "
                     f"axis   {'IN THE BOWL' if row['in_bowl'] else ''}",
                     (255, 255, 255)),
                    (f"red = TCP path   blue = {args.object} path   "
                     "magenta = reference", (200, 200, 200)),
                ])

            t += dt
            step_count += 1

        phase_log.append({
            "phase": name, "duration_s": duration + args.hold,
            "mean_tcp_error_mm": float(np.mean(errors)),
            "final_tcp_error_mm": float(errors[-1]),
            "obj_m": scene.object_pos().tolist(),
            "finger_contacts": scene.finger_contacts(),
        })
        if snapshots is not None:
            snapshots.append((name, data.qpos.copy()))

        print(f"  {name:10s} TCP error mean {np.mean(errors):5.2f} mm, "
              f"end {errors[-1]:5.2f} mm   {args.object} z "
              f"{(scene.object_pos()[2] - obj0[2]) * 1000:+6.1f} mm   "
              f"fingers on it {scene.finger_contacts()}")

    video.close()

    obj_final = scene.object_pos()
    obj_z = np.array([r["obj_z_m"] for r in rows])
    lifted_mm = float((obj_z.max() - obj0[2]) * 1000)
    in_bowl = scene.object_in_bowl()
    settled = scene.object_settled()
    picked = lifted_mm > args.min_lift * 1000
    # `settled` is reported, not required - see deploy.py.
    success = bool(picked and in_bowl)

    summary = {
        "task": "pick and drop into a bowl, hanging two-module I-Support "
                "+ soft gripper",
        "controller": "heuristic: min-jerk TCP waypoints tracked by 3x6 "
                      "Jacobian damped least-norm; gripper line ramped",
        "learned_components": "none",
        "object": args.object,
        "success": success,
        "picked": bool(picked),
        "in_bowl": bool(in_bowl),
        "settled": bool(settled),
        "obj_lift_mm": lifted_mm,
        "bowl_offset_mm": scene.bowl_offset_mm(),
        "obj_start_m": obj0.tolist(),
        "obj_final_m": obj_final.tolist(),
        "bowl_centre_m": bowl.tolist(),
        "bowl": {"rim_z_m": BS.BOWL_RIM_Z, "floor_z_m": BS.BOWL_FLOOR_Z,
                 "mouth_radius_m": BS.BOWL_R_RIM, "depth_m": BS.BOWL_DEPTH},
        "phases": phase_log,
        "settings": {k: getattr(args, k) for k in (
            "p0", "grip_pressure", "gain", "carry_gain", "max_rate",
            "tip_clearance", "top_clearance", "hover", "rim_clearance",
            "move_time", "vertical_time", "grip_time", "release_time",
            "settle_time", "hold", "lookahead")},
        "runtime": {"simulated_s": t,
                    "wall_clock_s": time.perf_counter() - started},
    }

    if not getattr(args, "no_outputs", False):
        write_csv(RESULTS_DIR / f"{args.tag}.csv", rows)
        write_json(RESULTS_DIR / f"{args.tag}.json", summary)
        plot(rows, summary, RESULTS_DIR / f"{args.tag}.png")

    print()
    print("=" * 78)
    print(f"  {'SUCCESS' if success else 'FAILED'}   {args.object} lifted "
          f"{lifted_mm:.1f} mm, "
          f"{'IN the bowl' if in_bowl else 'NOT in the bowl'}, "
          f"{scene.bowl_offset_mm():.1f} mm from its axis")
    print(f"  outputs in {RESULTS_DIR}")
    print("=" * 78)
    return summary


def plot(rows, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["time_s"] for r in rows])
    phase = [r["phase"] for r in rows]
    tcp = np.array([[r["tcp_x_m"], r["tcp_y_m"], r["tcp_z_m"]]
                    for r in rows]) * 1000
    ref = np.array([[r["ref_x_m"], r["ref_y_m"], r["ref_z_m"]]
                    for r in rows]) * 1000
    obj = np.array([[r["obj_x_m"], r["obj_y_m"], r["obj_z_m"]]
                    for r in rows]) * 1000
    bowl = np.array(summary["bowl_centre_m"]) * 1000
    rim_z = summary["bowl"]["rim_z_m"] * 1000
    floor_z = summary["bowl"]["floor_z_m"] * 1000
    mouth = summary["bowl"]["mouth_radius_m"] * 1000

    bounds = [0] + [i for i in range(1, len(phase))
                    if phase[i] != phase[i - 1]] + [len(phase)]

    fig = plt.figure(figsize=(19, 5.2))

    a = fig.add_subplot(1, 4, 1, projection="3d")
    a.plot(*ref.T, lw=2.5, color="#2f8bfd", alpha=0.35, label="reference")
    a.plot(*tcp.T, lw=1.2, color="#e8422f", label="TCP")
    a.plot(*obj.T, lw=1.6, color="#1f9e89", label=summary["object"])
    circle = np.linspace(0, 2 * np.pi, 60)
    a.plot(bowl[0] + mouth * np.cos(circle), bowl[1] + mouth * np.sin(circle),
           np.full_like(circle, rim_z), lw=1.6, color="#365562",
           label="bowl rim")
    a.set_xlabel("X [mm]"); a.set_ylabel("Y [mm]"); a.set_zlabel("Z [mm]")
    a.set_title(f"TCP and {summary['object']} paths"); a.legend(fontsize=7)

    def shade(ax):
        for i in range(len(bounds) - 1):
            p = phase[bounds[i]]
            ax.axvspan(t[bounds[i]], t[bounds[i + 1] - 1],
                       color=PHASE_COLOURS[p], alpha=0.10)
            ax.text(0.5 * (t[bounds[i]] + t[bounds[i + 1] - 1]),
                    ax.get_ylim()[1], p, ha="center", va="top", fontsize=7,
                    rotation=90)

    b = fig.add_subplot(1, 4, 2)
    b.plot(t, ref[:, 2], lw=2.5, color="#2f8bfd", alpha=0.35,
           label="TCP reference")
    b.plot(t, tcp[:, 2], lw=1.2, color="#e8422f", label="TCP")
    b.plot(t, obj[:, 2], lw=1.6, color="#1f9e89", label=summary["object"])
    b.axhline(rim_z, ls="--", lw=1.0, color="#365562", label="bowl rim")
    b.axhline(floor_z, ls=":", lw=1.0, color="#365562", label="bowl floor")
    b.set_xlabel("time [s]"); b.set_ylabel("height Z [mm]")
    b.set_title("Heights"); b.grid(alpha=0.3); b.legend(fontsize=7)
    shade(b)

    c = fig.add_subplot(1, 4, 3)
    radial = np.linalg.norm(obj[:, :2] - bowl[:2], axis=1)
    c.plot(t, radial, lw=1.3, color="#1f9e89",
           label=f"{summary['object']} to bowl axis")
    c.plot(t, [r["tcp_error_mm"] for r in rows], lw=1.1, color="#e8422f",
           label="TCP tracking error")
    c.axhline(mouth, ls="--", lw=1.0, color="#365562", label="mouth radius")
    c.set_xlabel("time [s]"); c.set_ylabel("distance [mm]")
    c.set_title("Closing on the bowl"); c.grid(alpha=0.3); c.legend(fontsize=7)
    shade(c)

    d = fig.add_subplot(1, 4, 4)
    for label in CHAMBER_LABELS:
        d.plot(t, [r[f"p_{label}_bar"] for r in rows], lw=1.0,
               ls="-" if label[0] == "A" else "--", label=label)
    d.plot(t, [r["gripper_bar"] for r in rows], lw=2.4, color="black",
           label="gripper line")
    d.set_xlabel("time [s]"); d.set_ylabel("pressure [bar]")
    d.set_title("7 pressure lines"); d.grid(alpha=0.3)
    d.legend(fontsize=7, ncol=2)
    shade(d)

    verdict = "SUCCESS" if summary["success"] else "FAILED"
    fig.suptitle(f"Heuristic pick and drop, {summary['object']} -> bowl — "
                 f"{verdict}: lifted {summary['obj_lift_mm']:.0f} mm, "
                 f"ended {summary['bowl_offset_mm']:.1f} mm from the bowl axis")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Heuristic pick and drop into a bowl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--p0", type=float, default=0.4,
                   help="Starting arm pressure (balanced, straight down).")
    p.add_argument("--arm-p-min", type=float, default=0.25,
                   help="Floor on every arm pressure, above the 0.2 bar "
                        "strain threshold of sim/kinematics.py.")
    p.add_argument("--grip-pressure", type=float, default=G.P_MAX)
    p.add_argument("--object", choices=("cube", "apple", "banana", "ball"),
                   default="cube")

    p.add_argument("--tip-clearance", type=float, default=0.003,
                   help="Height of the open fingertips above the table at "
                        "the grasp [m].")
    p.add_argument("--top-clearance", type=float, default=0.004,
                   help="Clearance of the open fingertips over the object's "
                        "top while hovering [m].")
    p.add_argument("--rim-clearance", type=float, default=0.008,
                   help="Clearance of the carried object's BASE over the "
                        "bowl rim during transport [m].")
    p.add_argument("--z-band", type=float, nargs=2, default=(0.397, 0.435),
                   help="TCP heights the controller can actually reach [m].")
    p.add_argument("--hover", type=float, default=0.030,
                   help="Clearance of the approach move [m].")

    p.add_argument("--move-time", type=float, default=3.0)
    p.add_argument("--vertical-time", type=float, default=2.0)
    p.add_argument("--grip-time", type=float, default=1.0)
    p.add_argument("--release-time", type=float, default=0.8,
                   help="Venting the gripper line [s]; one stage, not the "
                        "cage-then-vent the table set-down needed.")
    p.add_argument("--settle-time", type=float, default=1.2,
                   help="Hold after the release while the object falls [s].")
    p.add_argument("--hold", type=float, default=0.5,
                   help="Hold at the end of every phase [s].")

    p.add_argument("--gain", type=float, default=0.08)
    p.add_argument("--carry-gain", type=float, default=0.05,
                   help="Gain while an object is held.")
    p.add_argument("--max-rate", type=float, default=1.2)
    p.add_argument("--lookahead", type=int, default=4)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--jacobian-refresh", type=float, default=0.12)

    p.add_argument("--object-xy", type=float, nargs=2, default=None,
                   help="Object start on the table [m]; default sampled.")
    p.add_argument("--bowl-xy", type=float, nargs=2, default=None,
                   help="Bowl centre [m]; default sampled inside the hole.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the sampled layout.")
    p.add_argument("--tag", type=str, default=None,
                   help="Output file stem in results/ "
                        "(default pick_drop_<object>).")
    p.add_argument("--min-lift", type=float, default=0.020,
                   help="The object must rise at least this far to count as "
                        "picked [m].")

    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-outputs", action="store_true",
                   help="Skip the csv/json/plot; for callers that only want "
                        "the returned summary (the demo generator).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--cam-distance", type=float, default=1.05)
    p.add_argument("--cam-azimuth", type=float, default=115.0)
    p.add_argument("--cam-elevation", type=float, default=-18.0)

    args = p.parse_args(argv)
    args.tag = args.tag or f"pick_drop_{args.object}"
    return args


if __name__ == "__main__":
    run(parse_args())
