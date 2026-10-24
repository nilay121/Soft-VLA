#!/usr/bin/env python3
"""
Save what the tracking cameras see, and where they are.

    python save_representative_images.py
    python save_representative_images.py --count 12 --shape heart

Two kinds of picture, into representative_images/ next to this file:

    <view>/           frames from each policy camera, at 360 px for looking
                      at and at the policy's own 112 px
    overview_*.png    the whole environment from outside, with BOTH policy
                      cameras drawn in - a marker at each camera position and
                      a line along its view axis to the point it is aimed at

The overviews exist because the two policy views are `mjvCamera` objects
built in `track_cameras.py`, not bodies in the XML, so there is nothing in
the scene to look at and no other way to see how they are placed. They are
drawn with the recorder's `decorate` hook, which never touches the images
the policy is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import conditioning as C  # noqa: E402
import heuristic_track as H  # noqa: E402
import training_paths as P  # noqa: E402
from track_cameras import VIEWS, policy_cameras  # noqa: E402
from track_scene import TrackScene, balanced_pressure  # noqa: E402
from viz.common import (  # noqa: E402
    add_geom,
    add_line,
    add_polyline,
    free_camera,
    hide_link_capsules,
    style_scene,
)

OUT_DIR = HERE / "representative_images"

CAMERA_COLOURS = [np.array([1.0, 0.45, 0.10, 1.0], dtype=np.float32),
                  np.array([0.20, 0.85, 1.00, 1.0], dtype=np.float32)]


def camera_eye(view):
    """World position of a MuJoCo free camera from its orbit parameters.

    azimuth/elevation are degrees; the camera sits `distance` from `lookat`,
    on the far side of the direction it looks along.
    """

    az, el = np.radians(view["azimuth"]), np.radians(view["elevation"])
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                        np.sin(el)])
    return np.asarray(view["lookat"], dtype=np.float64) + view["distance"] * forward


def save_png(path, rgb):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def caption(img, lines, origin=(10, 10), scale=0.55, pad=7, spacing=23):
    """A dark band behind the whole caption, then the text on top.

    Drawing a thick black outline under thin glyphs leaves a halo that reads
    as ghost characters, so the background goes down once as a rectangle.
    """

    import cv2
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
              for t, _ in lines]
    x0, y0 = origin
    box = (x0, y0, x0 + max(widths) + 2 * pad,
           y0 + spacing * len(lines) + pad)
    panel = img[box[1]:box[3], box[0]:box[2]].astype(np.float32) * 0.25
    img[box[1]:box[3], box[0]:box[2]] = panel.astype(np.uint8)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(img, text, (x0 + pad, y0 + pad + spacing * i + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    return img


def contact_sheet(images, labels, path, columns=6, tile=200):
    import cv2
    rows = int(np.ceil(len(images) / columns))
    pad = 6
    sheet = np.full((rows * (tile + 20 + pad) + pad,
                     columns * (tile + pad) + pad, 3), 22, np.uint8)
    for i, (img, lab) in enumerate(zip(images, labels)):
        r, c = divmod(i, columns)
        y, x = pad + r * (tile + 20 + pad), pad + c * (tile + pad)
        sheet[y:y + tile, x:x + tile] = cv2.cvtColor(
            cv2.resize(img, (tile, tile), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_RGB2BGR)
        cv2.putText(sheet, lab, (x + 2, y + tile + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1,
                    cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def overview_decorator(scene_obj, eyes):
    """Draw both policy cameras into a rendered scene."""

    def decorate(scn):
        for eye, view, colour in zip(eyes, VIEWS, CAMERA_COLOURS):
            lookat = np.asarray(view["lookat"], dtype=np.float64)
            # camera body
            add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.022, 0.022, 0.022],
                     eye, colour)
            # its optical axis, out to what it is aimed at
            add_line(scn, eye, lookat, 0.0035, colour)
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012] * 3, lookat,
                     colour)
            # a crude frustum: four rays to the corners of the view
            forward = lookat - eye
            dist = np.linalg.norm(forward)
            forward = forward / dist
            right = np.cross(forward, [0, 0, 1.0])
            right /= max(np.linalg.norm(right), 1e-9)
            up = np.cross(right, forward)
            half = dist * np.tan(np.radians(22.5))
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                corner = lookat + sx * half * right + sy * half * up
                add_line(scn, eye, corner, 0.0014, colour)
    return decorate


def collect_views(args):
    """Run the expert once and grab frames from both policy cameras."""

    spec = next(s for s in P.EVAL_SHAPES if s["shape"] == args.shape)
    hargs = H.parse_args(["--no-video"])   # follow() writes nothing itself
    scene = TrackScene(hargs.control_dt)
    scene.reset(balanced_pressure(hargs.p0))
    start = scene.tcp()
    reference, on_shape = P.eval_reference(spec, scene.control_dt, start)

    model = scene.model
    hide_link_capsules(model)
    style_scene(model)
    for size in (args.big, args.small, args.wide):
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, size)
        model.vis.global_.offheight = max(model.vis.global_.offheight, size)
    cameras, cfgs = policy_cameras(scene, model)
    big = mujoco.Renderer(model, height=args.big, width=args.big)
    small = mujoco.Renderer(model, height=args.small, width=args.small)

    every = max(1, int(round(C.POLICY_DT / scene.control_dt)))
    on_idx = np.nonzero(on_shape)[0]
    wanted = set(np.linspace(on_idx[0], on_idx[-1], args.count
                             ).round().astype(int) // every * every)

    grabbed = {"big": [[], []], "small": [[], []], "t": []}

    def on_step(k, scene, pressure, t):
        if k % every or k not in wanted:
            return None
        for v, cam in enumerate(cameras):
            big.update_scene(scene.data, camera=cam)
            grabbed["big"][v].append(big.render().copy())
            small.update_scene(scene.data, camera=cam)
            grabbed["small"][v].append(small.render().copy())
        grabbed["t"].append(round(t, 2))
        return None

    H.follow(scene, reference, hargs, on_step=on_step)
    return scene, cfgs, grabbed, spec


def main(args):
    print()
    print("=" * 76)
    print("REPRESENTATIVE IMAGES — the two policy views, and where they are")
    print("=" * 76)

    if OUT_DIR.exists() and args.clean:
        import shutil
        shutil.rmtree(OUT_DIR)

    scene, cfgs, grabbed, spec = collect_views(args)
    names = [v["name"] for v in VIEWS]

    for v, name in enumerate(names):
        for j, (bigimg, smallimg, t) in enumerate(zip(
                grabbed["big"][v], grabbed["small"][v], grabbed["t"])):
            stem = f"{args.shape}_{j:02d}_t{t:05.2f}"
            save_png(OUT_DIR / name / f"{stem}.png", bigimg)
            save_png(OUT_DIR / name / "policy_input" / f"{stem}.png", smallimg)
        contact_sheet(grabbed["big"][v],
                      [f"{t:.1f}s" for t in grabbed["t"]],
                      OUT_DIR / f"{name}_contact_sheet.png")
        print(f"  {name}: {len(grabbed['big'][v])} frames "
              f"({args.big} px + {args.small} px policy input)")

    # --- the environment overviews -----------------------------------------
    eyes = [camera_eye(v) for v in VIEWS]
    model, data = scene.model, scene.data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.wide)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.wide)
    wide = mujoco.Renderer(model, height=int(args.wide * 0.68),
                           width=args.wide)
    decorate = overview_decorator(scene, eyes)

    shots = [("overview_front", 260.0, -14.0, 1.35),
             ("overview_above", 260.0, -48.0, 1.30),
             ("overview_side", 350.0, -10.0, 1.30)]
    for name, az, el, dist in shots:
        cam = free_camera(model, lookat=[0.0, 0.0, 0.52], distance=dist,
                          azimuth=az, elevation=el)
        wide.update_scene(data, camera=cam)
        decorate(wide.scene)
        img = wide.render().copy()
        caption(img, [
            ("two-module I-Support, hanging, gripper closed at "
             f"{scene.grip:.2f} bar", (255, 255, 255)),
            (f"orange = {names[0]} camera   (azimuth "
             f"{VIEWS[0]['azimuth']:.0f} deg)", (255, 150, 60)),
            (f"cyan   = {names[1]} camera   (azimuth "
             f"{VIEWS[1]['azimuth']:.0f} deg, 90 deg apart)", (90, 220, 255)),
            ("lines = optical axis and view frustum to the aim point",
             (200, 200, 200)),
        ])
        save_png(OUT_DIR / f"{name}.png", img)
        print(f"  {name}.png  (azimuth {az:.0f}, elevation {el:.0f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "shape_traced": spec["name"],
        "frames_per_view": len(grabbed["t"]),
        "times_s": grabbed["t"],
        "big_px": args.big, "policy_px": args.small,
        "cameras": [{**c, "eye_m": e.round(4).tolist()}
                    for c, e in zip(cfgs, eyes)],
        "overviews": [s[0] for s in shots],
    }, indent=2))

    print()
    print(f"  written to {OUT_DIR}")
    print("=" * 76)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Save policy-camera frames and environment overviews.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--shape", default="circle",
                   choices=[s["shape"] for s in P.EVAL_SHAPES])
    p.add_argument("--count", type=int, default=12,
                   help="Frames per camera.")
    p.add_argument("--big", type=int, default=360)
    p.add_argument("--small", type=int, default=112,
                   help="The policy's own resolution.")
    p.add_argument("--wide", type=int, default=1100,
                   help="Width of the environment overviews.")
    p.add_argument("--clean", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
