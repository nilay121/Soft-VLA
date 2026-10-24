"""
Shared machinery for the dry-run experiments.

Nothing here learns anything. It is the simulator, the analytic
constant-curvature kinematics, a finite-difference Jacobian, and enough
rendering glue to put a readable video on disk.

Used by:
    circle_tracking.py    follow a circle
    point_reaching.py     servo to a sequence of points
    elongation_reach.py   reach a point that needs the arm to stretch
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

# Offscreen rendering must be selected before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.parameters import NUM_SEGMENTS, THETA  # noqa: E402
from sim.mujoco_env import (  # noqa: E402
    P_MAX,
    P_MIN,
    PressureMapper,
    load_model,
    require_tip_site,
    tip_position,
)

OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Colours (RGBA)
# ---------------------------------------------------------------------------

COLOR_REFERENCE = np.array([0.20, 0.55, 1.00, 1.0], dtype=np.float32)
COLOR_TRACE = np.array([1.00, 0.25, 0.20, 1.0], dtype=np.float32)
# Magenta, not green: the three chamber cavities render red / green / blue,
# so a green target marker would be confusable with the hardware.
COLOR_TARGET = np.array([1.00, 0.15, 0.75, 1.0], dtype=np.float32)
COLOR_TIP = np.array([1.00, 0.85, 0.10, 1.0], dtype=np.float32)
COLOR_DONE = np.array([0.45, 0.45, 0.50, 1.0], dtype=np.float32)
COLOR_ERROR = np.array([1.00, 0.55, 0.10, 1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def settle(model, data, mapper, pressure, seconds):
    """Hold a pressure for a while so the arm stops moving."""

    data.ctrl[:] = mapper.convert(model, pressure)

    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def steady_tip(model, data, mapper, tip_id, pressure, seconds=0.4):
    """Reset, apply a pressure, settle, and report where the tip ends up."""

    mujoco.mj_resetData(model, data)
    data.ctrl[:] = mapper.convert(model, pressure)
    mujoco.mj_forward(model, data)

    settle(model, data, mapper, pressure, seconds)

    return tip_position(model, data, tip_id)


def cosine_pressure(p0, amplitude, phi):
    """Bend toward azimuth `phi` while holding the mean pressure at p0."""

    return np.clip(p0 + amplitude * np.cos(phi - THETA), P_MIN, P_MAX)


def substeps_for(model, control_dt):
    """How many physics steps make up one control period, and the real dt."""

    n = max(1, int(round(control_dt / model.opt.timestep)))
    return n, n * model.opt.timestep


# ---------------------------------------------------------------------------
# Arm geometry readout
# ---------------------------------------------------------------------------

class ArmGeometry:
    """
    Reads how long the arm currently is.

    `extension` is the sum of the five prismatic joints, i.e. how much the
    arm has stretched beyond its rest length. `backbone` walks the segment
    body origins and adds up the distances, which is the actual centre-line
    arc length including whatever bending is going on.
    """

    def __init__(self, model):
        self.extension_adr = []

        for i in range(1, NUM_SEGMENTS + 1):
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"extension_{i}"
            )

            if joint >= 0:
                self.extension_adr.append(int(model.jnt_qposadr[joint]))

        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                           "robot_base")]

        for i in range(1, NUM_SEGMENTS + 1):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     f"segment_{i}")
            if body >= 0:
                self.body_ids.append(body)

        self.body_ids = [b for b in self.body_ids if b >= 0]

    def extension(self, data):
        """Total prismatic extension in metres."""

        return float(sum(data.qpos[a] for a in self.extension_adr))

    def per_segment_extension(self, data):
        return np.array([data.qpos[a] for a in self.extension_adr])

    def backbone_length(self, data, tip_position_m=None):
        """Centre-line arc length from the base to the tip, in metres."""

        points = [data.xpos[b].copy() for b in self.body_ids]

        if tip_position_m is not None:
            points.append(np.asarray(tip_position_m, dtype=np.float64))

        points = np.asarray(points)

        return float(
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        )


# ---------------------------------------------------------------------------
# Finite-difference Jacobian + damped least squares
# ---------------------------------------------------------------------------

class JacobianCorrector:
    """
    Local pressure -> tip-position Jacobian, by rolling a copy of the sim
    forward with each chamber perturbed. Refreshed on an interval and reused
    in between, because six forward rollouts are not cheap.
    """

    def __init__(
        self,
        model,
        mapper,
        tip_id,
        horizon=0.05,
        probe=0.02,
        damping=1e-4,
        max_correction=0.35,
        refresh_dt=0.12,
    ):
        self.model = model
        self.mapper = mapper
        self.tip_id = tip_id

        self.horizon = horizon
        self.probe = probe
        self.damping = damping
        self.max_correction = max_correction
        self.refresh_dt = refresh_dt

        self.scratch = mujoco.MjData(model)

        self.jacobian = None
        self.last_refresh = -1e9
        self.refreshes = 0
        self.rollouts = 0

    def _rollout(self, source, pressure):
        mujoco.mj_copyData(self.scratch, self.model, source)

        self.scratch.ctrl[:] = self.mapper.convert(self.model, pressure)

        end = self.scratch.time + self.horizon

        while self.scratch.time < end - 1e-12:
            mujoco.mj_step(self.model, self.scratch)

        self.rollouts += 1

        return self.scratch.site_xpos[self.tip_id].copy()

    def _finite_difference(self, source, pressure):
        jacobian = np.zeros((3, 3))

        for axis in range(3):
            high = pressure.copy()
            low = pressure.copy()

            high[axis] = min(P_MAX, high[axis] + self.probe)
            low[axis] = max(P_MIN, low[axis] - self.probe)

            span = high[axis] - low[axis]

            if span <= 1e-12:
                continue

            jacobian[:, axis] = (
                self._rollout(source, high) - self._rollout(source, low)
            ) / span

        return jacobian

    def refresh(self, source, pressure, now, force=False):
        if (
            force
            or self.jacobian is None
            or now - self.last_refresh >= self.refresh_dt
        ):
            candidate = self._finite_difference(source, pressure)

            if np.all(np.isfinite(candidate)):
                self.jacobian = candidate
                self.last_refresh = now
                self.refreshes += 1

    def correction(self, source, pressure, error, now):
        """A bounded pressure delta that reduces `error`."""

        self.refresh(source, pressure, now)

        if self.jacobian is None:
            return np.zeros(3)

        J = self.jacobian

        try:
            delta = np.linalg.solve(
                J.T @ J + self.damping * np.eye(3),
                J.T @ error,
            )
        except np.linalg.LinAlgError:
            return np.zeros(3)

        if not np.all(np.isfinite(delta)):
            return np.zeros(3)

        norm = float(np.linalg.norm(delta))

        if norm > self.max_correction:
            delta *= self.max_correction / norm

        return delta


class PointReachController:
    """
    Servo the tip to a Cartesian point.

    Each control period: measure the error, ask the Jacobian for a pressure
    delta, apply a fraction of it, and rate-limit the result so the arm moves
    smoothly instead of snapping. Pure feedback - there is no plan and no
    model of the target beyond its coordinates.
    """

    def __init__(
        self,
        corrector,
        gain=0.65,
        max_rate=1.2,
        tolerance=0.002,
        hold_time=0.35,
    ):
        self.corrector = corrector
        self.gain = gain
        self.max_rate = max_rate
        self.tolerance = tolerance
        self.hold_time = hold_time

        self.within_tolerance_since = None

    def reset(self):
        self.within_tolerance_since = None

    def step(self, data, pressure, tip, target, now, dt):
        error = np.asarray(target, dtype=np.float64) - tip

        delta = self.corrector.correction(data, pressure, error, now)

        command = pressure + self.gain * delta

        # Rate limit so the pneumatics are not asked for a step change.
        step = np.clip(
            command - pressure,
            -self.max_rate * dt,
            self.max_rate * dt,
        )

        new_pressure = np.clip(pressure + step, P_MIN, P_MAX)

        distance = float(np.linalg.norm(error))

        if distance <= self.tolerance:
            if self.within_tolerance_since is None:
                self.within_tolerance_since = now
        else:
            self.within_tolerance_since = None

        return new_pressure, distance

    def settled(self, now):
        return (
            self.within_tolerance_since is not None
            and now - self.within_tolerance_since >= self.hold_time
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Anodised-aluminium look for the structural meshes. Reads as solid hardware
# and keeps the red / blue / green / yellow overlay markers legible against it.
SHELL_COLOR = (0.58, 0.64, 0.74)

# The three chamber cavities ship at alpha 0.5, which makes the whole arm look
# like frosted glass. Near-opaque keeps them readable as coloured chambers.
CAVITY_RGB = {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}

FLOOR_GREY = 0.40

HEADLIGHT_AMBIENT = 0.30
HEADLIGHT_DIFFUSE = 0.65
HEADLIGHT_SPECULAR = 0.20


def style_scene(
    model,
    shell_alpha=1.0,
    shell_rgb=SHELL_COLOR,
    cavity_alpha=0.95,
    floor_grey=FLOOR_GREY,
    relight=True,
):
    """
    Make the arm read clearly on video.

    The scene as authored renders as a pale, near-transparent ghost: the
    structural meshes are light grey on a light grey floor, the chamber
    cavities are at alpha 0.5, and the single overhead light leaves the
    headlight at diffuse 0.4. Stacked translucent surfaces then wash
    everything out.

    This recolours the shell to an anodised-aluminium tone, makes the
    cavities near-opaque, darkens the ground for contrast and brings the
    headlight up.

    All of it is rendering only: geom_rgba, mat_rgba and vis.headlight are
    never read by the solver, so the physics is bit-for-bit unchanged.
    """

    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue

        rgb = tuple(round(float(x), 1) for x in model.geom_rgba[geom_id, :3])

        if rgb in CAVITY_RGB:
            model.geom_rgba[geom_id, 3] = cavity_alpha
        else:
            if shell_rgb is not None:
                model.geom_rgba[geom_id, :3] = shell_rgb

            model.geom_rgba[geom_id, 3] = shell_alpha

    if floor_grey is not None:
        for mat_id in range(model.nmat):
            model.mat_rgba[mat_id, :3] = floor_grey

    if relight:
        model.vis.headlight.ambient[:] = [HEADLIGHT_AMBIENT] * 3
        model.vis.headlight.diffuse[:] = [HEADLIGHT_DIFFUSE] * 3
        model.vis.headlight.specular[:] = [HEADLIGHT_SPECULAR] * 3


def hide_link_capsules(model):
    """
    Hide the blue `link_*` capsules that wrap each segment.

    Those capsules exist only to carry mass - the CAD meshes are all
    density="0" - and they are collision-free (contype/conaffinity = 0).
    Visually they read as a fat blue sleeve with a dome over the top of the
    arm, which hides the real hardware. Setting alpha to 0 removes them from
    the render and changes nothing in the physics: body_mass is fixed at
    compile time and geom_rgba is never read by the solver.

    What is left is the actual I-Support geometry: the stacked rings, the
    three chamber cavities in red/green/blue, and the end disc.
    """

    hidden = 0

    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

        if name and name.startswith("link_"):
            model.geom_rgba[geom_id, 3] = 0.0
            hidden += 1

    return hidden


def add_geom(scene, geom_type, size, pos, rgba):
    """Append one decorative geom to a rendered scene."""

    if scene.ngeom >= scene.maxgeom:
        return False

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )

    # mjv_initGeom leaves category at 0, which the renderer's category mask
    # skips. Decoration geoms must say so explicitly or they never appear.
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    scene.ngeom += 1
    return True


def add_line(scene, start, end, width, rgba):
    """Append one capsule segment between two points."""

    if scene.ngeom >= scene.maxgeom:
        return False

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )

    mujoco.mjv_connector(
        geom,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )

    geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    scene.ngeom += 1
    return True


def add_polyline(scene, points, width, rgba, max_segments=600):
    """Append a polyline, decimated so long traces cannot exhaust maxgeom."""

    points = np.asarray(points)

    if len(points) < 2:
        return

    stride = max(1, len(points) // max_segments)
    points = points[::stride]

    for i in range(len(points) - 1):
        add_line(scene, points[i], points[i + 1], width, rgba)


def overlay_text(frame, lines, origin=(14, 26), scale=0.5, spacing=22):
    """Burn a small HUD into a BGR frame.

    The background is darkened once, as a panel, rather than giving every
    glyph a thick outline. Over a bright floor a 3 px outline under 1 px text
    smears the ends of each line into what looks like duplicated words.
    """

    import cv2

    x, y = origin
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
              for t, _ in lines]
    pad = 8
    x0, y0 = max(x - pad, 0), max(y - int(spacing * 0.8) - pad // 2, 0)
    x1 = min(x + max(widths) + pad, frame.shape[1])
    y1 = min(y + spacing * (len(lines) - 1) + pad + 4, frame.shape[0])
    frame[y0:y1, x0:x1] = (frame[y0:y1, x0:x1].astype(np.float32)
                           * 0.30).astype(np.uint8)

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x, y + i * spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _find_ffmpeg():
    """An ffmpeg binary with an H.264 encoder, or None.

    Prefers the static build shipped by the `imageio-ffmpeg` pip package
    (it always includes libx264), then a system ffmpeg on PATH.
    """

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def encode_h264(src, dst, ffmpeg=None):
    """Re-encode `src` to a widely playable H.264 mp4 at `dst`.

    Returns True on success. yuv420p + faststart is what browsers, VS Code,
    QuickTime and GitHub need; OpenCV's own 'mp4v' (MPEG-4 Part 2) output
    plays almost only in VLC.
    """

    import subprocess

    ffmpeg = ffmpeg or _find_ffmpeg()
    if ffmpeg is None:
        return False

    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-preset", "medium", "-movflags", "+faststart", str(dst)],
        capture_output=True, text=True)
    return result.returncode == 0


class VideoRecorder:
    """Offscreen renderer plus an mp4 writer.

    Frames are written with OpenCV, whose pip builds only offer the 'mp4v'
    codec (MPEG-4 Part 2) - a format most players refuse. close() therefore
    re-encodes the file to H.264 with ffmpeg (pip install imageio-ffmpeg);
    without ffmpeg the mp4v file is kept and a warning is printed.
    """

    def __init__(self, model, path, width, height, fps, camera,
                 shell_alpha=1.0, hide_links=True, enabled=True):
        self.enabled = enabled
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = camera

        if not enabled:
            self.renderer = None
            self.writer = None
            return

        import cv2

        if hide_links:
            hide_link_capsules(model)

        style_scene(model, shell_alpha=shell_alpha)

        # The scene's offscreen framebuffer defaults to 640x480. Enlarge it
        # here rather than in scene.xml, which other experiments share.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)

        self.renderer = mujoco.Renderer(model, height=height, width=width,
                                        max_geom=20000)

        # raw mp4v first; re-encoded to H.264 at self.path in close()
        self.raw_path = self.path.with_name(self.path.stem + ".mp4v.mp4")
        self.writer = cv2.VideoWriter(
            str(self.raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open the video writer for {path}")

    def capture(self, data, decorate=None, hud=None):
        if not self.enabled:
            return

        import cv2

        self.renderer.update_scene(data, camera=self.camera)

        if decorate is not None:
            decorate(self.renderer.scene)

        frame = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

        if hud:
            overlay_text(frame, hud)

        self.writer.write(frame)

    def close(self):
        if not self.enabled:
            return

        self.writer.release()
        self.renderer.close()

        if encode_h264(self.raw_path, self.path):
            self.raw_path.unlink()
        else:
            self.raw_path.replace(self.path)
            print(f"WARNING: {self.path.name} is left in the 'mp4v' codec, "
                  "which many players cannot open. pip install imageio-ffmpeg "
                  "to get H.264 output.")


def free_camera(model, lookat, distance, azimuth, elevation):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)

    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.lookat[:] = np.asarray(lookat, dtype=np.float64)

    return camera


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_scene(control_dt, shell_alpha=None):
    """Load the model and the usual handles in one call."""

    model = load_model()
    data = mujoco.MjData(model)

    tip_id = require_tip_site(model)
    mapper = PressureMapper()
    geometry = ArmGeometry(model)

    n_sub, real_dt = substeps_for(model, control_dt)

    return model, data, tip_id, mapper, geometry, n_sub, real_dt
