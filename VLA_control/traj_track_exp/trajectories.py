"""
Test shapes and the shape-similarity check that keeps them out of training.

Both generate_dataset.py (to reject look-alike training paths) and deploy.py
(to build the test references) import the shapes from here, so the two can
never disagree about what "the test shapes" are.

Every shape is a planar curve in unit coordinates, normalised so its largest
distance from the origin is 1, then placed at a centre and scaled in metres
by `make_reference()`.
"""

from __future__ import annotations

import numpy as np

TEST_SHAPES = ("circle", "ellipse", "heart", "lemniscate", "square", "s_shape")

# Closed shapes are traced once around; the S is open and traced end to end.
CLOSED = {"circle": True, "ellipse": True, "heart": True, "lemniscate": True,
          "square": True, "s_shape": False}


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def _raw_shape(name, n=2000):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    if name == "circle":
        xy = np.c_[np.cos(t), np.sin(t)]
    elif name == "ellipse":
        xy = np.c_[np.cos(t), 0.55 * np.sin(t)]
    elif name == "heart":
        xy = np.c_[16 * np.sin(t) ** 3,
                   13 * np.cos(t) - 5 * np.cos(2 * t)
                   - 2 * np.cos(3 * t) - np.cos(4 * t)]
    elif name == "lemniscate":                         # of Bernoulli
        d = 1.0 + np.sin(t) ** 2
        xy = np.c_[np.cos(t) / d, np.sin(t) * np.cos(t) / d]
    elif name == "square":
        s = np.linspace(0.0, 4.0, n, endpoint=False)
        side, f = np.floor(s).astype(int), s - np.floor(s)
        corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]], float)
        xy = corners[side] + (corners[side + 1] - corners[side]) * f[:, None]
    elif name == "s_shape":
        # two half-open arcs: upper counter-clockwise, lower clockwise
        a = np.linspace(0.0, 1.5 * np.pi, n // 2)
        upper = np.c_[0.5 * np.cos(a), 0.5 + 0.5 * np.sin(a)]
        b = np.linspace(0.5 * np.pi, -np.pi, n // 2)
        lower = np.c_[0.5 * np.cos(b), -0.5 + 0.5 * np.sin(b)]
        xy = np.vstack([upper, lower[1:]])
    else:
        raise ValueError(f"unknown shape {name!r}")

    xy = xy - 0.5 * (xy.max(0) + xy.min(0))           # centre the bounding box
    return xy / np.linalg.norm(xy, axis=1).max()


def resample_by_arclength(points, n, closed):
    """n points evenly spaced along the curve (constant speed)."""

    pts = np.vstack([points, points[:1]]) if closed else points
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    target = np.linspace(0.0, s[-1], n, endpoint=not closed)
    return np.column_stack([np.interp(target, s, pts[:, k])
                            for k in range(pts.shape[1])])


def shape_length(name):
    xy = _raw_shape(name)
    pts = np.vstack([xy, xy[:1]]) if CLOSED[name] else xy
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def make_reference(name, centre, scale, speed, control_dt, approach_s):
    """
    Reference for one test shape: a smooth approach from `centre` to the
    shape's start point, then the shape traced once at constant `speed`
    (m/s) in the horizontal plane through `centre`.

    Returns (path [N,3], on_shape [N] bool): metrics use on_shape only.
    """

    duration = shape_length(name) * scale / speed
    n_shape = max(2, int(round(duration / control_dt)))

    xy = resample_by_arclength(_raw_shape(name), n_shape, CLOSED[name])
    shape = np.column_stack([centre[0] + scale * xy[:, 0],
                             centre[1] + scale * xy[:, 1],
                             np.full(n_shape, centre[2])])

    n_app = int(round(approach_s / control_dt))
    s = np.linspace(0.0, 1.0, n_app, endpoint=False)
    s = s * s * (3.0 - 2.0 * s)                                   # smoothstep
    approach = np.asarray(centre)[None] + s[:, None] * (shape[0] - centre)[None]

    path = np.vstack([approach, shape])
    on_shape = np.r_[np.zeros(n_app, bool), np.ones(n_shape, bool)]
    return path, on_shape


# ---------------------------------------------------------------------------
# Shape similarity (position-, size- and rotation-invariant)
# ---------------------------------------------------------------------------

_N_CMP = 96
_ANGLES = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)


def _normalise(xy):
    xy = xy - xy.mean(0)
    return xy / max(np.sqrt((xy ** 2).sum(1).mean()), 1e-12)


def _project_to_plane(points):
    """Best-fit plane (PCA) of a 3-D curve, returned as 2-D coordinates."""

    centred = points - points.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


_TEMPLATES = {name: _normalise(resample_by_arclength(_raw_shape(name), _N_CMP,
                                                     CLOSED[name]))
              for name in TEST_SHAPES}


def _chamfer_best_pose(a, b):
    """Symmetric mean nearest-point distance, minimised over rotation and
    reflection of `a` (both already centred and RMS-normalised)."""

    best = np.inf
    for mirror in (1.0, -1.0):
        am = a * np.array([1.0, mirror])
        c, s = np.cos(_ANGLES), np.sin(_ANGLES)
        rot = np.stack([np.stack([c, -s], 1), np.stack([s, c], 1)], 1)  # [R,2,2]
        ar = np.einsum("rij,nj->rni", rot, am)                          # [R,N,2]
        d = np.linalg.norm(ar[:, :, None] - b[None, None], axis=3)      # [R,N,M]
        score = 0.5 * (d.min(2).mean(1) + d.min(1).mean(1))
        best = min(best, float(score.min()))
    return best


def shape_distance(path, window_fractions=(0.2, 0.35, 0.5, 0.75, 1.0)):
    """
    How close any stretch of a 3-D path looks to each test shape.

    The path (and every sliding window of it) is projected onto its best-fit
    plane, centred and scaled to unit RMS radius, then compared with each
    test template under the best rotation and reflection. Projection makes
    the check conservative: a path only has to LOOK like a shape from its
    best viewing direction to count.

    Returns {shape: smallest distance over all windows}; 0 = identical.
    """

    path = np.asarray(path, dtype=np.float64)
    n = len(path)
    out = {name: np.inf for name in TEST_SHAPES}

    for frac in window_fractions:
        w = max(8, int(round(frac * n)))
        stride = max(1, w // 4)
        for start in range(0, n - w + 1, stride):
            seg = path[start:start + w]
            if np.linalg.norm(np.ptp(seg, axis=0)) < 1e-4:            # static
                continue
            xy = _normalise(resample_by_arclength(_project_to_plane(seg),
                                                  _N_CMP, closed=False))
            for name, tpl in _TEMPLATES.items():
                out[name] = min(out[name], _chamfer_best_pose(xy, tpl))

    return out


def template_cross_distances():
    """Distances between the test shapes themselves, for calibration."""

    return {(a, b): _chamfer_best_pose(_TEMPLATES[a], _TEMPLATES[b])
            for i, a in enumerate(TEST_SHAPES) for b in TEST_SHAPES[i + 1:]}
