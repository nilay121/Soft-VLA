#!/usr/bin/env python3
"""
Measure the envelope the expert can actually track, before designing the
training distribution.

    python workspace_probe.py
    python workspace_probe.py --quick

The pick-and-place task quoted a tool-point band of 395-437 mm, but that was
measured with a table underneath and a controller tuned for reaching static
targets. This scene has no table, and the question here is different: over
what centres and radii can the controller follow a MOVING reference without
falling behind?

So the probe tracks a circle at a grid of (centre height, radius) and reports
the mean error over the shape. Anything the expert cannot follow has no place
in the demonstrations - cloning a failed trace teaches the policy to fail.

Writes results/workspace_probe.{json,png}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import heuristic_track as H  # noqa: E402
from track_scene import TrackScene, balanced_pressure  # noqa: E402

RESULTS_DIR = HERE / "results"


def probe_one(scene, args, centre_z, radius):
    hargs = H.parse_args(["--no-video", "--shape", "circle",
                          "--radius", str(radius), "--speed", str(args.speed),
                          "--gain", str(args.gain)])
    scene.reset(balanced_pressure(hargs.p0))
    start = scene.tcp()
    centre = np.array([0.0, 0.0, centre_z])
    reference, on_shape = H.reference_for("circle", centre, radius,
                                          args.speed, scene.control_dt,
                                          hargs.approach, start)
    log = H.follow(scene, reference, hargs)
    m = H.metrics(log, on_shape)
    tcp = log["tcp"][np.asarray(on_shape, bool)]
    return {"centre_z_m": float(centre_z), "radius_m": float(radius),
            "mean_mm": m["mean_mm"], "p95_mm": m["p95_mm"],
            "max_mm": m["max_mm"],
            "achieved_z_mm": float(tcp[:, 2].mean() * 1000),
            "achieved_radius_mm": float(
                np.linalg.norm(tcp[:, :2] - centre[:2], axis=1).mean() * 1000)}


def main(args):
    print()
    print("=" * 78)
    print("WORKSPACE PROBE — what the expert can actually track")
    print("=" * 78)

    heights = ([0.38, 0.41, 0.44] if args.quick else
               [0.355, 0.375, 0.395, 0.415, 0.435, 0.455])
    radii = ([0.03, 0.06] if args.quick else
             [0.020, 0.035, 0.050, 0.065, 0.080])

    print(f"  circles at {args.speed * 1000:.0f} mm/s, gain {args.gain}")
    print(f"  {len(heights)} heights x {len(radii)} radii = "
          f"{len(heights) * len(radii)} runs")
    print()

    scene = TrackScene(0.01)
    rows = []
    started = time.perf_counter()
    for z in heights:
        line = f"  z {z * 1000:5.0f} mm : "
        for r in radii:
            row = probe_one(scene, args, z, r)
            rows.append(row)
            flag = "ok " if row["mean_mm"] <= args.tolerance_mm else "BAD"
            line += f"r{r * 1000:3.0f} {row['mean_mm']:6.2f}mm {flag}  "
        print(line, flush=True)

    good = [r for r in rows if r["mean_mm"] <= args.tolerance_mm]
    envelope = {
        "tolerance_mm": args.tolerance_mm,
        "usable_centre_z_m": [min(r["centre_z_m"] for r in good),
                              max(r["centre_z_m"] for r in good)] if good else None,
        "max_usable_radius_m": max(r["radius_m"] for r in good) if good else None,
        "speed_mps": args.speed, "gain": args.gain,
        "runs": rows,
        "seconds": time.perf_counter() - started,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "workspace_probe.json").write_text(
        json.dumps(envelope, indent=2))
    plot(rows, heights, radii, args, RESULTS_DIR / "workspace_probe.png")

    print()
    print("=" * 78)
    if good:
        print(f"  usable within {args.tolerance_mm:.0f} mm: centre height "
              f"{1000 * envelope['usable_centre_z_m'][0]:.0f}"
              f"-{1000 * envelope['usable_centre_z_m'][1]:.0f} mm, "
              f"radius up to {1000 * envelope['max_usable_radius_m']:.0f} mm")
    else:
        print("  nothing tracked within tolerance - lower the speed or gain")
    print(f"  {len(rows)} runs in {envelope['seconds'] / 60:.1f} min")
    print(f"  outputs in {RESULTS_DIR}")
    print("=" * 78)


def plot(rows, heights, radii, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.full((len(heights), len(radii)), np.nan)
    for r in rows:
        grid[heights.index(r["centre_z_m"]), radii.index(r["radius_m"])] = \
            r["mean_mm"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r",
                   vmin=0, vmax=max(args.tolerance_mm * 2, np.nanmax(grid)))
    ax.set_xticks(range(len(radii)))
    ax.set_xticklabels([f"{r * 1000:.0f}" for r in radii])
    ax.set_yticks(range(len(heights)))
    ax.set_yticklabels([f"{z * 1000:.0f}" for z in heights])
    ax.set_xlabel("circle radius [mm]"); ax.set_ylabel("centre height [mm]")
    for i in range(len(heights)):
        for j in range(len(radii)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if grid[i, j] > args.tolerance_mm
                        else "black")
    fig.colorbar(im, ax=ax, label="mean tracking error [mm]")
    ax.set_title(f"What the expert can track at {args.speed * 1000:.0f} mm/s "
                 f"(numbers = mean error, mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Measure the trackable envelope of the hanging arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--speed", type=float, default=0.030)
    p.add_argument("--gain", type=float, default=0.12)
    p.add_argument("--tolerance-mm", type=float, default=3.0,
                   help="Mean error above this counts as not trackable.")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
