"""
Robot, material, simulation and controller parameters.

Values are loaded from configs/robot.yaml. The path is resolved relative to
this file, so the project can be imported from any working directory.
"""

from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "robot.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# Robot

TOTAL_LENGTH = float(config["robot"]["total_length"])

NUM_SEGMENTS = int(config["robot"]["num_segments"])

SEGMENT_LENGTH = TOTAL_LENGTH / NUM_SEGMENTS

TOTAL_MASS = float(config["robot"]["total_mass"])

SEGMENT_MASS = TOTAL_MASS / NUM_SEGMENTS

# Material

YOUNGS_MODULUS = float(config["material"]["youngs_modulus"])

POISSON_RATIO = float(config["material"]["poisson_ratio"])

SHEAR_MODULUS = (
    YOUNGS_MODULUS /
    (2 * (1 + POISSON_RATIO))
)

# Simulation

DT = float(config["simulation"]["dt"])

GRAVITY = float(config["simulation"]["gravity"])

# Controller

BEND_KP = float(config["controller"]["bend_kp"])

EXTENSION_KP = float(config["controller"]["extension_kp"])

# Pressure

T_RISE = config["pressure"]["rise_time"]

T_DROP = config["pressure"]["drop_time"]

P_THRESHOLD = config["pressure"]["threshold"]

# Strain

STRAIN_GAIN = np.array(
    config["strain"]["gain"]
)

# Geometry

DELTA = config["geometry"]["delta"]

THETA = np.deg2rad(
    config["geometry"]["theta"]
)

INNER_RADIUS = config["geometry"]["inner_radius"]

OUTER_RADIUS = config["geometry"]["outer_radius"]

# Areas

ACTUATOR_AREA = (
    np.pi *
    (OUTER_RADIUS**2 - INNER_RADIUS**2)
)

AIR_AREA = (
    np.pi *
    INNER_RADIUS**2
)

# Moments of Inertia

I_LOCAL = (
    np.pi / 4
) * (
    OUTER_RADIUS**4 -
    INNER_RADIUS**4
)

I1 = np.sum(
    I_LOCAL +
    ACTUATOR_AREA *
    (DELTA * np.sin(THETA))**2
)

I2 = np.sum(
    I_LOCAL +
    ACTUATOR_AREA *
    (DELTA * np.cos(THETA))**2
)

I3 = I1 + I2
