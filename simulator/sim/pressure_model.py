import numpy as np
from configs.parameters import *


class PressureModel:

    def __init__(self):
        self.pressure = np.zeros(3, dtype=float)
        self.desired_pressure = np.zeros(3, dtype=float)

    def update(self, pressure_command):

        pressure_command = np.asarray(
            pressure_command,
            dtype=float
        )

        pressure_command = np.clip(
            pressure_command,
            0.0,
            3.0
        )

        self.desired_pressure = pressure_command.copy()

        for i in range(3):

            error = (
                self.desired_pressure[i]
                - self.pressure[i]
            )

            if abs(error) < P_THRESHOLD:
                self.pressure[i] = self.desired_pressure[i]
                continue

            if error > 0:
                rate = 1.0 / max(T_RISE, DT)
            else:
                rate = 1.0 / max(T_DROP, DT)

            step = rate * DT

            if abs(error) <= step:
                self.pressure[i] = self.desired_pressure[i]
            else:
                self.pressure[i] += np.sign(error) * step

        self.pressure = np.clip(
            self.pressure,
            0.0,
            3.0
        )

        return self.pressure.copy()

    def get_pressure(self):
        return self.pressure.copy()