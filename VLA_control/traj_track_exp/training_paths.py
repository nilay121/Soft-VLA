"""
The random paths the demonstrations are recorded on, and the held-out
circles they are evaluated against.

ENVELOPE
--------
Measured by `workspace_probe.py`, not guessed. Between 395 and 435 mm the
expert tracks a circle of any radius up to 80 mm to about 0.2 mm; at 375 mm
it starts to fall behind beyond 65 mm; at 355 mm it loses anything past
35 mm; and 455 mm is above the arm entirely. The sampler stays inside the
part that is comfortably trackable, because cloning a trace the expert
itself botched teaches the policy to botch it.

WHY THE TRAINING PATHS ARE NOT CIRCLES
--------------------------------------
The policy is evaluated on circles. If circles were also in the training
set, the evaluation would measure memorisation rather than tracking, so
every sampled path is compared against the test templates
(`trajectories.shape_distance`, which is invariant to position, scale,
rotation and reflection, and checks sliding windows as well as the whole
path) and re-drawn if any stretch of it looks like one. The same check was
used on the previous robot's tracking dataset.

What is left is smooth random wandering: a cubic spline through a handful of
waypoints, arc-length resampled so the path speed is constant, which is the
same constant speed the test circles are traced at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parent.parent / "simulator"
for path in (str(SIMULATOR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import trajectories as T  # noqa: E402

# --- measured in results/workspace_probe.json ------------------------------
# Re-measured after the tool point moved to the closed fingertips, 18 mm
# below the old grasp TCP: the whole envelope came down with it. The expert
# now tracks any radius up to 80 mm between 375 and 415 mm, is good to 50 mm
# at 355, and cannot reach 435 at all.
Z_RANGE = (0.375, 0.414)      # m, centre height of the tool point
R_MAX = 0.070                 # m, lateral radius from the mount axis

SPEED = 0.030                 # m/s, the same for training and evaluation
DURATION = 12.0               # s of tracking per demonstration
APPROACH_S = 1.5              # s, smooth move onto the path

# How circle-like a path may be before it is thrown away. Calibrated
# against trajectories.template_cross_distances(), which puts the CLOSEST
# pair of test templates (circle vs square) at 0.092 - so any threshold at
# or above that is unsatisfiable, and the line has to sit below it.
# Measured against all five templates: a REAL test shape scores 0.014
# (circle) to 0.045 (square), while random sampled paths bottom out at
# 0.109. 0.075 sits in that gap with margin on both sides - it rejects
# anything genuinely shape-like and keeps 100% of honest random paths.
SHAPE_REJECT = 0.075

# Only the circle and the ellipse are checked, not all six templates: the
# evaluation set is circles, an ellipse is a squashed circle, and comparing
# against the other four costs three times the CPU for nothing. The full
# six-template check is still available as trajectories.shape_distance().
# Every shape the policy is evaluated on, plus the ellipse because it is a
# squashed circle. The other template (s_shape) is not evaluated, so it is
# not worth the CPU.
CHECK_TEMPLATES = ("circle", "ellipse", "square", "heart", "lemniscate",
                   "s_shape")
CHECK_WINDOWS = (0.35, 0.6, 1.0)

# --- the evaluation set ----------------------------------------------------
# All held out of training by the circle/square/heart/lemniscate rejection
# above. `radius` is the shape's largest distance from its centre, so a
# square of radius 45 mm has its CORNERS at 45 mm.
EVAL_SHAPES = [
    {"name": "circle", "shape": "circle", "radius": 0.045,
     "centre": (0.0, 0.0, 0.395)},
    {"name": "rectangle", "shape": "square", "radius": 0.045,
     "centre": (0.0, 0.0, 0.395)},
    {"name": "heart", "shape": "heart", "radius": 0.045,
     "centre": (0.0, 0.0, 0.395)},
    {"name": "lemniscate", "shape": "lemniscate", "radius": 0.050,
     "centre": (0.0, 0.0, 0.395)},
    {"name": "ellipse", "shape": "ellipse", "radius": 0.050,
     "centre": (0.0, 0.0, 0.395)},
    # the only open curve in the set - traced end to end, not round once
    {"name": "s_shape", "shape": "s_shape", "radius": 0.050,
     "centre": (0.0, 0.0, 0.395)},
]
EVAL_CIRCLES = EVAL_SHAPES          # old name, kept so nothing breaks


def _clip_into_envelope(points):
    """Pull a path back inside the measured envelope."""

    pts = np.array(points, dtype=np.float64)
    radial = np.linalg.norm(pts[:, :2], axis=1)
    too_far = radial > R_MAX
    if too_far.any():
        pts[too_far, :2] *= (R_MAX / radial[too_far])[:, None]
    pts[:, 2] = np.clip(pts[:, 2], *Z_RANGE)
    return pts


def _random_waypoints(rng, start, n):
    """`n` smooth waypoints inside the envelope, beginning at `start`."""

    pts = [np.asarray(start, dtype=np.float64)]
    for _ in range(n):
        r = R_MAX * np.sqrt(rng.uniform(0.05, 1.0))     # area-uniform
        a = rng.uniform(0.0, 2.0 * np.pi)
        z = rng.uniform(*Z_RANGE)
        pts.append(np.array([r * np.cos(a), r * np.sin(a), z]))
    return np.array(pts)


def _spline_through(waypoints, n_out, closed):
    from scipy.interpolate import CubicSpline

    pts = np.vstack([waypoints, waypoints[:1]]) if closed else waypoints
    u = np.linspace(0.0, 1.0, len(pts))
    spline = CubicSpline(u, pts, axis=0,
                         bc_type="periodic" if closed else "natural")
    return spline(np.linspace(0.0, 1.0, n_out))


def circle_likeness(path, window_fractions=CHECK_WINDOWS):
    """Smallest distance from any stretch of `path` to a circle or ellipse.

    The same machinery `trajectories.shape_distance` uses - project the
    window onto its best-fit plane, normalise, then match the template under
    every rotation and reflection - but restricted to the two templates that
    matter here and to three window sizes. 0 = identical.
    """

    path = np.asarray(path, dtype=np.float64)
    n = len(path)
    best = np.inf
    for frac in window_fractions:
        w = max(8, int(round(frac * n)))
        stride = max(1, w // 3)
        for start in range(0, n - w + 1, stride):
            seg = path[start:start + w]
            if np.linalg.norm(np.ptp(seg, axis=0)) < 1e-4:          # static
                continue
            xy = T._normalise(T.resample_by_arclength(
                T._project_to_plane(seg), T._N_CMP, closed=False))
            for name in CHECK_TEMPLATES:
                best = min(best, T._chamfer_best_pose(xy, T._TEMPLATES[name]))
    return float(best)


def sample_path(rng, start, control_dt, speed=SPEED, duration=DURATION,
                approach_s=APPROACH_S, max_tries=25):
    """One training reference: [N,3] at one row per control step.

    Returns (path, on_path, info). `on_path` marks the tracking part, after
    the smooth approach from `start`.
    """

    n_track = max(2, int(round(duration / control_dt)))
    target_length = speed * duration
    rejected = []

    for attempt in range(max_tries):
        n_way = int(rng.integers(4, 8))
        closed = bool(rng.random() < 0.4)
        raw = _spline_through(_random_waypoints(rng, start, n_way),
                              2000, closed)
        raw = _clip_into_envelope(raw)

        # rescale the curve so that tracing it at `speed` takes `duration`
        seg = np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()
        if seg < 1e-6:
            continue
        centre = raw.mean(0)
        raw = _clip_into_envelope(centre + (raw - centre) * (target_length / seg))

        path = T.resample_by_arclength(raw, n_track, closed=False)

        worst = circle_likeness(path)
        if worst < SHAPE_REJECT:
            rejected.append(round(float(worst), 3))
            continue

        n_app = int(round(approach_s / control_dt))
        s = np.linspace(0.0, 1.0, n_app, endpoint=False)
        s = s * s * (3.0 - 2.0 * s)
        start = np.asarray(start, dtype=np.float64)
        approach = start[None] + s[:, None] * (path[0] - start)[None]

        full = np.vstack([approach, path])
        on_path = np.r_[np.zeros(n_app, bool), np.ones(n_track, bool)]
        info = {"waypoints": n_way, "closed": closed, "attempts": attempt + 1,
                "rejected_for_shape": len(rejected),
                "min_shape_distance": round(float(worst), 3),
                "length_mm": round(float(np.linalg.norm(
                    np.diff(path, axis=0), axis=1).sum() * 1000), 1)}
        return full, on_path, info

    raise RuntimeError(f"could not draw a non-test-like path in {max_tries} "
                       "tries - loosen SHAPE_REJECT")


def eval_reference(spec, control_dt, start, speed=SPEED,
                   approach_s=APPROACH_S):
    """One held-out evaluation shape, as (path, on_shape)."""

    import heuristic_track as H
    return H.reference_for(spec.get("shape", "circle"),
                           np.asarray(spec["centre"]),
                           spec["radius"], speed, control_dt, approach_s,
                           start)


__all__ = ["sample_path", "eval_reference", "EVAL_SHAPES", "EVAL_CIRCLES", "Z_RANGE",
           "R_MAX", "SPEED", "DURATION", "APPROACH_S", "SHAPE_REJECT"]
