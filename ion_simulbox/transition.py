from typing import List, Dict, Tuple
import numpy as np
from sympy import S
from sympy.physics.wigner import wigner_3j
import qutip
from .ion import Ion
from .laser import Laser
from .energy_level import (
    EnergyLevel,
    FineStructureZeemanLevel,
    HyperfineStructureZeemanLevel,
    FineStructure,
    HyperfineStructure,
)
from enum import Enum
from .units import Constants, Units
from .utils import number_to_sympy, sympy_to_number


class TransitionOrder(Enum):
    dipole = 0
    quadrupole = 1


class Transition:
    def __init__(
        self,
        level_1: EnergyLevel,
        level_2: EnergyLevel,
        laser: Laser,
    ):
        self.laser = laser
        if level_1.energy > level_2.energy:
            self.lower_level = level_2
            self.upper_level = level_1
        else:
            self.lower_level = level_1
            self.upper_level = level_2
        self.order = self.get_order()
        self.linewidth = self.upper_level.line_width
        self.frequency = (
            self.upper_level.energy - self.lower_level.energy
        ) / Constants.h
        self.angular_frequency = 2 * np.pi * self.frequency
        self.detunning = 2 * np.pi * (laser.frequency - self.frequency)
        self.branching_ratio = self.get_branching_ratio()
        self.rabi_frequency = self.get_rabi_frequency()

    def get_branching_ratio(self):
        return self.upper_level.parent.branching_ratios[self.lower_level.parent]

    def get_order(self):
        if abs(self.lower_level.L - self.upper_level.L) == 1:
            return TransitionOrder.dipole
        elif abs(self.lower_level.L - self.upper_level.L) == 2:
            return TransitionOrder.quadrupole
        else:
            raise ValueError("Transition order not supported")

    def __str__(self):
        return f"Transition(level_1={self.lower_level}, level_2={self.upper_level})"

    def __repr__(self):
        return self.__str__()

    def get_rabi_frequency(self):
        if self.order == TransitionOrder.dipole:
            coefficient = (
                self.laser.get_electric_field_amplitude()
                / Constants.h_bar
                * np.sqrt(
                    3
                    * Constants.epsilon_0
                    * Constants.h_bar
                    * self.laser.wavelength**3
                    * self.branching_ratio
                    * self.linewidth
                    / (8 * np.pi**2)
                )
            ) * np.sqrt(2 * self.upper_level.J + 1)
            sign = (-1) ** (
                self.lower_level.J
                + self.upper_level.J
                + max(self.lower_level.J, self.upper_level.J)
                - self.upper_level.m
            )
            polarization_effect = 0j
            for q, eps_q in zip(
                range(-1, 2), self.laser.polarization.epsilon_in_spherical_tensor
            ):
                polarization_effect += eps_q * sympy_to_number(
                    wigner_3j(
                        number_to_sympy(self.upper_level.J),
                        number_to_sympy(1),
                        number_to_sympy(self.lower_level.J),
                        -number_to_sympy(self.upper_level.m),
                        number_to_sympy(q),
                        number_to_sympy(self.lower_level.m),
                    )
                )
            return sign * coefficient * polarization_effect
        elif self.order == TransitionOrder.quadrupole:
            raise NotImplementedError("Quadrupole transitions not implemented")
        else:
            raise ValueError("Transition order not supported")

    