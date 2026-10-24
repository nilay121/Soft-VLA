"""
PneuNet-style soft gripper: geometry and the single-pressure actuation model.

Three fingers at 120 degrees around the end disc. Each finger is a chain of
FINGER_SEGMENTS short links joined by hinges - the same rigid
pseudo-continuum idea the arm uses. One pressure line feeds all three
fingers, as in a real PneuNet gripper where the finger chambers share a
manifold:

    p = 0            vented: fingers straight, splayed slightly outward (open)
    p -> P_MAX       every finger curls inward with constant curvature (closed)

The pressure -> joint-angle map is linear, like the constant-curvature arm
model: each hinge turns by the same angle, so the finger bends as an arc.
When a finger meets the object its position actuator stalls and keeps
pushing - that stalled torque is the grip force.
"""

from __future__ import annotations

import numpy as np

NUM_FINGERS = 3
FINGER_SEGMENTS = 4
FINGER_ANGLES_DEG = (90.0, 210.0, 330.0)   # around the arm axis

PALM_RADIUS = 0.030        # m, matches the arm's link radius
PALM_HALF_THICKNESS = 0.006
FINGER_ROOT_RADIUS = 0.020  # m, finger roots sit on this circle
SEGMENT_LENGTH = 0.018      # m  -> 72 mm fingers
FINGER_HALF_WIDTH = 0.008   # m  (16 mm wide)
FINGER_HALF_THICK = 0.0045  # m  (9 mm thick strain-limiting layer)
SEGMENT_MASS = 0.004        # kg
PALM_MASS = 0.015           # kg

P_MAX = 1.0                 # bar, gripper supply pressure
OPEN_SPLAY_DEG = 18.0       # root hinge splay when vented; wide enough to
                            # clear a cube that has shifted ~8 mm in the grip
CLOSE_PER_JOINT_DEG = 26.0  # extra inward bend per hinge at P_MAX

JOINT_RANGE_DEG = (-45.0, 20.0)
FINGER_KP = 0.25            # N m / rad, per hinge
FINGER_FORCE_LIMIT = 0.08   # N m, per hinge actuator

# Where the fingertips meet when closed on nothing: the grasp point (TCP),
# measured from the arm's end disc along the arm axis.
TCP_OFFSET = 2 * PALM_HALF_THICKNESS + 0.050


def joint_name(finger, segment):
    return f"finger{finger}_joint{segment}"


def joint_names():
    return [joint_name(f, s) for f in range(NUM_FINGERS)
            for s in range(FINGER_SEGMENTS)]


def finger_targets(pressure):
    """Hinge angle targets [rad] for all fingers from the one line pressure.

    Positive rotation about the hinge axis swings a finger OUTWARD, so
    closing is negative. Order matches joint_names().
    """

    p = float(np.clip(pressure, 0.0, P_MAX)) / P_MAX
    per_joint = np.full(FINGER_SEGMENTS, -np.radians(CLOSE_PER_JOINT_DEG) * p)
    per_joint[0] += np.radians(OPEN_SPLAY_DEG) * (1.0 - p)
    return np.tile(per_joint, NUM_FINGERS)


def open_tip_drop():
    """How far the open (vented) fingertips reach below the TCP, along the
    arm axis [m]. The grasp height is set from this so the tips clear the
    table whatever the object."""

    reach = FINGER_SEGMENTS * SEGMENT_LENGTH * np.cos(np.radians(OPEN_SPLAY_DEG))
    return 2 * PALM_HALF_THICKNESS + reach + FINGER_HALF_THICK - TCP_OFFSET
