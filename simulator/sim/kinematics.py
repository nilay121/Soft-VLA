"""
Pressure -> Strain -> Extension -> Curvature -> Joint Targets

The pressure-strain model follows the AM I-Support model:

    epsilon_i = (N * A_air / (E * A_total)) * p_i

where:

    A_air   = pi * r_inner^2
    A_total = pi * (r_outer^2 - r_inner^2)

The individual actuator strain gains are then applied:

    epsilon_i <- gamma_i * epsilon_i

Pressure input is assumed to be in bar.
"""

import numpy as np

from configs.parameters import (
    TOTAL_LENGTH,
    NUM_SEGMENTS,
    SEGMENT_LENGTH,
    YOUNGS_MODULUS,
    STRAIN_GAIN,
    DELTA,
    THETA,
    ACTUATOR_AREA,
    AIR_AREA,
    I1,
    I2,
    P_THRESHOLD,
)


BAR_TO_PA = 1.0e5


class Kinematics:

    def __init__(self):

        self.strain = np.zeros(3, dtype=float)

        self.extension = 0.0

        self.mean_strain = 0.0

        self.kappa_x = 0.0
        self.kappa_y = 0.0

        self.slide_targets = np.zeros(
            NUM_SEGMENTS,
            dtype=float
        )

        self.bend_x_targets = np.zeros(
            NUM_SEGMENTS,
            dtype=float
        )

        self.bend_y_targets = np.zeros(
            NUM_SEGMENTS,
            dtype=float
        )

    def pressure_to_strain(self, pressure):

        pressure = np.asarray(
            pressure,
            dtype=float
        )

        if pressure.shape != (3,):
            raise ValueError(
                "Pressure must have shape (3,)"
            )

        effective_pressure = np.where(
            pressure >= P_THRESHOLD,
            pressure,
            0.0
        )

        pressure_pa = (
            effective_pressure
            * BAR_TO_PA
        )

        pressure_to_strain = (
            AIR_AREA
            /
            (
                YOUNGS_MODULUS
                * ACTUATOR_AREA
            )
        )

        base_strain = (
            pressure_to_strain
            * pressure_pa
        )

        self.strain = (
            STRAIN_GAIN
            * base_strain
        )

        return self.strain.copy()

    def compute_mean_strain(self):

        self.mean_strain = np.mean(
            self.strain
        )

        return self.mean_strain

    def compute_extension(self):

        self.extension = (
            self.mean_strain
            * TOTAL_LENGTH
        )

        return self.extension

    def compute_curvature(self):

        kappa_x = 0.0
        kappa_y = 0.0

        for i in range(3):

            kappa_x += (
                self.strain[i]
                * ACTUATOR_AREA
                * DELTA
                * np.sin(THETA[i])
            )

            kappa_y += (
                self.strain[i]
                * ACTUATOR_AREA
                * DELTA
                * np.cos(THETA[i])
            )

        self.kappa_x = (
            kappa_x / I1
        )

        self.kappa_y = (
            -kappa_y / I2
        )

        return (
            self.kappa_x,
            self.kappa_y
        )

    def compute_joint_targets(self):

        segment_extension = (
            self.extension
            / NUM_SEGMENTS
        )

        segment_theta_x = (
            self.kappa_x
            * SEGMENT_LENGTH
        )

        segment_theta_y = (
            self.kappa_y
            * SEGMENT_LENGTH
        )

        self.slide_targets[:] = (
            segment_extension
        )

        self.bend_x_targets[:] = (
            segment_theta_x
        )

        self.bend_y_targets[:] = (
            segment_theta_y
        )

        return (
            self.slide_targets.copy(),
            self.bend_x_targets.copy(),
            self.bend_y_targets.copy()
        )