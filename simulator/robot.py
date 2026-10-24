"""
Two-module I-Support robot: model, actuation mapping and local Jacobian.

    module A : segments  1..5,  chambers A1 A2 A3   -> pressure[0:3]
    module B : segments  6..10, chambers B1 B2 B3   -> pressure[3:6]

Each module is an independent 5-segment continuum section with its own three
pneumatic valves. Module B is mounted on module A's end plate, so it rides on
whatever pose module A has taken.

The pressure -> joint-target map is the same analytic constant-curvature model
used by the one-module robot (sim/kinematics.py), applied once per module.
That is the whole point of a modular arm: the per-module physics is unchanged,
only the kinematic chain gets longer.

--------------------------------------------------------------------------
REDUNDANCY
--------------------------------------------------------------------------
The tip has 3 Cartesian degrees of freedom but the robot now has 6 pressure
inputs, so the map is redundant: many pressure combinations reach the same
point. The Jacobian is therefore 3x6 and the controller uses the damped
least-NORM solution

    dp = J^T (J J^T + lambda I_3)^-1 e

rather than the least-squares form used for the square 3x3 one-module case.
Least-norm picks the smallest pressure change that achieves the motion, which
keeps the two modules from fighting each other.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent     # the simulator root
# This layout is flat: configs/, sim/, viz/ and every simulator module live
# side by side under simulator/, so one entry on sys.path resolves them all.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sim.kinematics import Kinematics  # noqa: E402
from sim.mujoco_env import P_MAX, P_MIN, stabilize_model  # noqa: E402

SCENE_PATH = HERE / "scene_2module.xml"

MODULES = 2
SEGMENTS_PER_MODULE = 5
CHAMBERS_PER_MODULE = 3

NUM_SEGMENTS = MODULES * SEGMENTS_PER_MODULE
NUM_CHAMBERS = MODULES * CHAMBERS_PER_MODULE
NUM_ACTUATORS = 3 * NUM_SEGMENTS

MODULE_NAMES = ("A", "B")

CHAMBER_LABELS = tuple(
    f"{MODULE_NAMES[m]}{c + 1}"
    for m in range(MODULES)
    for c in range(CHAMBERS_PER_MODULE)
)

# Uniform pressure on every chamber: the direction that lengthens the whole
# arm. Useful for describing the motion, but NOT the right thing to constrain
# (see MODULE_EXTENSION_AXES).
ELONGATION_AXIS = np.ones(NUM_CHAMBERS) / np.sqrt(NUM_CHAMBERS)


def module_slice(module):
    """Pressure indices belonging to module 0 (A) or 1 (B)."""

    return slice(module * CHAMBERS_PER_MODULE,
                 (module + 1) * CHAMBERS_PER_MODULE)


def _module_extension_axes():
    """
    One unit vector per module: the pressure direction that changes THAT
    module's length.

    From sim/kinematics.py a module's extension is

        extension = (c * L / 3) * sum_i gamma_i * p_i

    so it depends on the strain-gain-weighted sum of that module's three
    pressures. Two things follow:

      * the correct axis is gamma / |gamma|, not (1,1,1)/sqrt(3) - those are
        2.6 degrees apart because the three chambers do not have equal gains;
      * there is one constraint PER MODULE. Conserving only the global mean
        lets the modules trade length with each other: module A shortens
        while module B grows, total pressure unchanged, arm length very much
        changed. That is a real failure mode, not a hypothetical - it is what
        the first version of this experiment did.

    The two axes have disjoint support, so they are automatically orthonormal
    and can be projected out independently.
    """

    from configs.parameters import STRAIN_GAIN

    gamma = np.asarray(STRAIN_GAIN, dtype=np.float64)
    gamma = gamma / np.linalg.norm(gamma)

    axes = np.zeros((MODULES, NUM_CHAMBERS))

    for m in range(MODULES):
        axes[m, module_slice(m)] = gamma

    return axes


MODULE_EXTENSION_AXES = _module_extension_axes()


def remove_extension_components(delta):
    """
    Strip every length-changing component from a pressure update.

    What is left can only bend the arm: each module's gamma-weighted pressure
    sum is preserved, so each module keeps its own length.
    """

    delta = np.asarray(delta, dtype=np.float64).copy()

    for axis in MODULE_EXTENSION_AXES:
        delta -= np.dot(delta, axis) * axis

    return delta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(scene_path=SCENE_PATH, stabilize=True):
    scene_path = Path(scene_path)

    if not scene_path.exists():
        raise FileNotFoundError(
            f"Two-module scene not found:\n{scene_path}\n"
            "Generate it with: python 2_module_ISupport/build_scene.py"
        )

    model = mujoco.MjModel.from_xml_path(str(scene_path))

    if model.nu != NUM_ACTUATORS:
        raise RuntimeError(
            f"Expected {NUM_ACTUATORS} actuators, found {model.nu}"
        )

    if stabilize:
        stabilize_model(model)

    return model


def require_tip_site(model):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                "end_effector")

    if site_id < 0:
        raise RuntimeError("Scene has no 'end_effector' site.")

    return site_id


def tip_position(model, data, site_id):
    return data.site_xpos[site_id].copy()


# ---------------------------------------------------------------------------
# Pressure -> 30 actuator setpoints
# ---------------------------------------------------------------------------

class TwoModulePressureMapper:
    """
    Map 6 chamber pressures to the 30 position-actuator setpoints.

    One `Kinematics` instance per module, each fed its own three pressures.
    Actuator order matches the XML: segments 1..10, and within each segment
    (bend_x, bend_y, extension).
    """

    def __init__(self):
        self.modules = [Kinematics() for _ in range(MODULES)]

    def convert(self, model, pressure):
        pressure = np.clip(
            np.asarray(pressure, dtype=np.float64),
            P_MIN,
            P_MAX,
        )

        if pressure.shape != (NUM_CHAMBERS,):
            raise ValueError(
                f"pressure must have shape ({NUM_CHAMBERS},), "
                f"got {pressure.shape}"
            )

        actuators = np.zeros(model.nu, dtype=np.float64)

        for m, kinematics in enumerate(self.modules):
            kinematics.pressure_to_strain(pressure[module_slice(m)])
            kinematics.compute_mean_strain()
            kinematics.compute_extension()
            kinematics.compute_curvature()

            slide, bend_x, bend_y = kinematics.compute_joint_targets()

            for s in range(SEGMENTS_PER_MODULE):
                segment = m * SEGMENTS_PER_MODULE + s
                base = 3 * segment

                if base + 2 >= model.nu:
                    break

                actuators[base] = bend_x[s]
                actuators[base + 1] = bend_y[s]
                actuators[base + 2] = slide[s]

        return actuators


# ---------------------------------------------------------------------------
# Geometry readout
# ---------------------------------------------------------------------------

class ArmGeometry:
    """Reads how long the arm is, overall and per module."""

    def __init__(self, model):
        self.extension_adr = []

        for g in range(1, NUM_SEGMENTS + 1):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                      f"extension_{g}")
            if joint >= 0:
                self.extension_adr.append(int(model.jnt_qposadr[joint]))

        self.body_ids = []

        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_base")
        if base >= 0:
            self.body_ids.append(base)

        for g in range(1, NUM_SEGMENTS + 1):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     f"segment_{g}")
            if body >= 0:
                self.body_ids.append(body)

    def per_segment_extension(self, data):
        return np.array([data.qpos[a] for a in self.extension_adr])

    def extension(self, data):
        """Total prismatic extension of the whole arm, in metres."""

        return float(self.per_segment_extension(data).sum())

    def module_extension(self, data, module):
        """Prismatic extension of one module, in metres."""

        per = self.per_segment_extension(data)
        lo = module * SEGMENTS_PER_MODULE
        return float(per[lo:lo + SEGMENTS_PER_MODULE].sum())

    def backbone_length(self, data, tip_position_m=None):
        """Centre-line arc length from base to tip, in metres."""

        points = [data.xpos[b].copy() for b in self.body_ids]

        if tip_position_m is not None:
            points.append(np.asarray(tip_position_m, dtype=np.float64))

        points = np.asarray(points)

        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    def module_joint_position(self, data, module):
        """World position of the frame where a module starts."""

        index = module * SEGMENTS_PER_MODULE
        return data.xpos[self.body_ids[index]].copy()


# ---------------------------------------------------------------------------
# Heuristic feedforward
# ---------------------------------------------------------------------------

def cosine_pressure(p0, amplitudes, phis):
    """
    Per-module cosine bend.

    `p0`, `amplitudes` and `phis` may each be a scalar (applied to both
    modules) or a length-2 sequence (one value per module). Returns the
    6-vector [A1 A2 A3 B1 B2 B3].
    """

    from configs.parameters import THETA

    def pair(value):
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        return np.resize(arr, MODULES)

    p0 = pair(p0)
    amplitudes = pair(amplitudes)
    phis = pair(phis)

    out = np.empty(NUM_CHAMBERS, dtype=np.float64)

    for m in range(MODULES):
        out[module_slice(m)] = (
            p0[m] + amplitudes[m] * np.cos(phis[m] - THETA)
        )

    return np.clip(out, P_MIN, P_MAX)


# ---------------------------------------------------------------------------
# Local Jacobian, damped least-norm
# ---------------------------------------------------------------------------

class JacobianCorrector:
    """
    Local 3x6 pressure -> tip Jacobian, finite-differenced by rolling a copy
    of the simulation forward with each chamber perturbed in turn.

    Twelve rollouts per refresh (6 chambers, central difference), so the
    Jacobian is cached and reused between refreshes exactly as in the
    one-module experiments.
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
        jacobian = np.zeros((3, NUM_CHAMBERS))

        for axis in range(NUM_CHAMBERS):
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
        """
        A bounded 6-vector pressure delta that reduces `error`.

        Damped least-norm: with 6 inputs for 3 outputs there is a null space,
        and least-norm chooses the smallest pressure change, which stops the
        two modules from pushing against one another.
        """

        self.refresh(source, pressure, now)

        if self.jacobian is None:
            return np.zeros(NUM_CHAMBERS)

        J = self.jacobian

        try:
            # dp = J^T (J J^T + lambda I)^-1 e
            lhs = J @ J.T + self.damping * np.eye(3)
            delta = J.T @ np.linalg.solve(lhs, error)
        except np.linalg.LinAlgError:
            return np.zeros(NUM_CHAMBERS)

        if not np.all(np.isfinite(delta)):
            return np.zeros(NUM_CHAMBERS)

        norm = float(np.linalg.norm(delta))

        if norm > self.max_correction:
            delta *= self.max_correction / norm

        return delta


class PointReachController:
    """Servo the tip to a Cartesian point. Same logic as the one-module
    version, widened to 6 chambers."""

    def __init__(self, corrector, gain=0.65, max_rate=1.2,
                 tolerance=0.002, hold_time=0.35):
        self.corrector = corrector
        self.gain = gain
        self.max_rate = max_rate
        self.tolerance = tolerance
        self.hold_time = hold_time

        self.within_tolerance_since = None

    def reset(self):
        self.within_tolerance_since = None

    def step(self, data, pressure, tip, target, now, dt, project_out=None):
        error = np.asarray(target, dtype=np.float64) - tip

        delta = self.corrector.correction(data, pressure, error, now)

        if project_out is not None:
            delta = delta - np.dot(delta, project_out) * project_out

        command = pressure + self.gain * delta

        step = np.clip(command - pressure,
                       -self.max_rate * dt, self.max_rate * dt)

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
# Convenience
# ---------------------------------------------------------------------------

def build_scene(control_dt):
    """Load the model and the usual handles in one call."""

    model = load_model()
    data = mujoco.MjData(model)

    tip_id = require_tip_site(model)
    mapper = TwoModulePressureMapper()
    geometry = ArmGeometry(model)

    substeps = max(1, int(round(control_dt / model.opt.timestep)))

    return model, data, tip_id, mapper, geometry, substeps, (
        substeps * model.opt.timestep
    )


def settle(model, data, mapper, pressure, seconds):
    data.ctrl[:] = mapper.convert(model, pressure)

    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def steady_tip(model, data, mapper, tip_id, pressure, seconds=0.5):
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = mapper.convert(model, pressure)
    mujoco.mj_forward(model, data)

    settle(model, data, mapper, pressure, seconds)

    return tip_position(model, data, tip_id)
