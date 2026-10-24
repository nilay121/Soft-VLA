"""
Shared MuJoCo helpers for the soft-robot scene.

This module holds everything that used to be duplicated verbatim across
main_figure8.py / main_heart.py / main_random.py:

    - model loading and numerical stabilisation
    - site lookup / tip position
    - target-marker drawing
    - offscreen RGB rendering for the VLA vision branch
    - pressure -> 15 actuator setpoints (PressureMapper)
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from sim.kinematics import Kinematics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SIM_DIR = Path(__file__).resolve().parent

SCENE_PATH = SIM_DIR / "scene.xml"

# ---------------------------------------------------------------------------
# Pressure limits (bar)
# ---------------------------------------------------------------------------

P_MIN = 0.0
P_MAX = 3.0

# ---------------------------------------------------------------------------
# Numerical stabilisation profile
#
# The scene ships with light damping so it can be inspected in the MuJoCo
# viewer. For closed-loop control the joints need extra damping and the
# position actuators need a matched derivative term, otherwise the 15-DOF
# chain rings at the actuator bandwidth.
# ---------------------------------------------------------------------------

JOINT_DAMPING_SCALE = 2.5
MIN_JOINT_DAMPING = 0.012

# Armature is a per-DOF quantity in MuJoCo (model.dof_armature). The original
# scripts tried to set `model.actuator_damping` / `model.actuator_armature`,
# which do not exist on MjModel in any MuJoCo version (2.x or 3.x) and raised
# AttributeError. Armature is applied per DOF below; actuator damping is
# applied the correct way, as the derivative term kv of each position
# actuator's affine bias (see ACTUATOR_DAMPING_RATIO).
DOF_ARMATURE = 0.003

ACTUATOR_DAMPING_RATIO = 1.25

EXPECTED_ACTUATORS = 15

TIP_SITE_NAMES = ("end_effector", "tip", "ee", "end")

TARGET_SITE_NAMES = ("target_site", "target", "goal_site", "goal")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(scene_path=SCENE_PATH, stabilize=True):
    """Load scene.xml and apply the closed-loop stabilisation profile."""

    scene_path = Path(scene_path)

    if not scene_path.exists():
        raise FileNotFoundError(f"MuJoCo scene not found:\n{scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))

    if model.nu != EXPECTED_ACTUATORS:
        raise RuntimeError(
            f"Expected {EXPECTED_ACTUATORS} actuators, found {model.nu}"
        )

    if stabilize:
        stabilize_model(model)

    return model


def stabilize_model(model):
    """Raise joint damping and give each position actuator a matched kv."""

    if model.nv > 0:
        model.dof_damping[:] = np.maximum(
            model.dof_damping * JOINT_DAMPING_SCALE,
            MIN_JOINT_DAMPING,
        )

        model.dof_armature[:] = np.maximum(
            model.dof_armature,
            DOF_ARMATURE,
        )

    # Give each position actuator a critically-ish damped derivative term.
    # For an affine bias the parameters are (bias, kp, kv) and MuJoCo applies
    # bias = -kp*(length - ctrl) - kv*velocity, so kv must be negative here.
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])

        if joint_id < 0 or joint_id >= model.njnt:
            continue

        if int(model.actuator_biastype[actuator_id]) != int(
            mujoco.mjtBias.mjBIAS_AFFINE
        ):
            continue

        kp = float(model.actuator_gainprm[actuator_id, 0])

        if kp <= 0.0:
            continue

        dof_adr = int(model.jnt_dofadr[joint_id])

        inertia = max(float(model.dof_M0[dof_adr]), 1e-8)

        kv = 2.0 * ACTUATOR_DAMPING_RATIO * math.sqrt(kp * inertia)

        model.actuator_biasprm[actuator_id, 2] = -kv

    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.tolerance = 1e-6
    model.opt.iterations = 30
    model.opt.ls_iterations = 10

    return model


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

def find_site(model, names=TIP_SITE_NAMES):
    for name in names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

        if site_id >= 0:
            return site_id

    return -1


def require_tip_site(model):
    site_id = find_site(model, TIP_SITE_NAMES)

    if site_id < 0:
        raise RuntimeError(
            "Could not find an end-effector site. Tried: "
            + ", ".join(TIP_SITE_NAMES)
        )

    return site_id


def tip_position(model, data, site_id):
    return data.site_xpos[site_id].copy()


def hide_xml_target(model):
    """Hide the static target site so only the live marker is visible."""

    target_id = find_site(model, TARGET_SITE_NAMES)

    if target_id >= 0:
        model.site_rgba[target_id, 3] = 0.0


def finite_state(model, data):
    return (
        np.all(np.isfinite(data.qpos))
        and np.all(np.isfinite(data.qvel))
        and np.all(np.isfinite(data.qacc))
    )


# ---------------------------------------------------------------------------
# Viewer / rendering
# ---------------------------------------------------------------------------

def draw_target(viewer, target, radius=0.0045):
    """Draw the live green target marker in the passive viewer."""

    scene = viewer.user_scn

    if scene.maxgeom < 1:
        return

    scene.ngeom = 0

    mujoco.mjv_initGeom(
        scene.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )

    scene.ngeom = 1


def render_rgb(renderer, data, camera=0):
    """Render one RGB frame as float32 CHW in [0,1] for the VLA."""

    renderer.update_scene(data, camera=camera)

    image = np.asarray(renderer.render(), dtype=np.float32)

    if image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError(f"Unexpected renderer output: {image.shape}")

    return np.transpose(image[:, :, :3] / 255.0, (2, 0, 1)).copy()


# ---------------------------------------------------------------------------
# Pressure -> actuators
# ---------------------------------------------------------------------------

class PressureMapper:
    """
    Map 3 chamber pressures to the 15 position-actuator setpoints via the
    constant-curvature kinematics in sim/kinematics.py.

    Actuator order per segment i is (bend_x, bend_y, extension).
    """

    def __init__(self):
        self.kinematics = Kinematics()

    def convert(self, model, pressure):
        pressure = np.clip(
            np.asarray(pressure, dtype=np.float64),
            P_MIN,
            P_MAX,
        )

        k = self.kinematics

        k.pressure_to_strain(pressure)
        k.compute_mean_strain()
        k.compute_extension()
        k.compute_curvature()

        slide, bend_x, bend_y = k.compute_joint_targets()

        actuators = np.zeros(model.nu, dtype=np.float64)

        segments = min(
            model.nu // 3,
            len(slide),
            len(bend_x),
            len(bend_y),
        )

        for i in range(segments):
            j = 3 * i
            actuators[j] = bend_x[i]
            actuators[j + 1] = bend_y[i]
            actuators[j + 2] = slide[i]

        return actuators
