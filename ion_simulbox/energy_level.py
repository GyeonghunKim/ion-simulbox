from enum import Enum
from sympy import S
from sympy.core.numbers import Rational
from typing import Callable, Dict, List
import numpy as np
from .units import Units, Constants


class EnergyLevel:
    def __init__(
        self,
        name: str,
        energy: float,
        n: int,
        I: Rational,
        L: int,
        J: Rational,
        line_width: float,
        branching_ratios: List[Dict[str, float]],
    ):
        self.name = name
        self.energy = energy
        self.n = n
        self.I = I
        self.L = L
        self.J = J
        self.line_width = line_width
        self.branching_ratios = branching_ratios


class FineStructureZeemanLevel(EnergyLevel):
    def __init__(
        self,
        name: str,
        energy: float,
        n: int,
        I: Rational,
        L: int,
        J: Rational,
        m: float,
        line_width: float,
        branching_ratios: List[Dict[str, float]],
    ):
        super().__init__(name, energy, n, I, L, J, line_width, branching_ratios)
        self.m = m
        self.lande_g_factor = 1 + (J * (J + 1) - L * (L + 1) + 0.5 * (0.5 + 1)) / (
            2 * J * (J + 1)
        )
        self.zeeman_splitting_func: Callable[[float], float] = (
            lambda B_field: self.lande_g_factor * self.m * Constants.mu_B * B_field
        )

    def apply_magnetic_field(self, magentic_field: float):
        self.energy = self.energy + self.zeeman_splitting_func(magentic_field)

    def __str__(self):
        return f"FineStructureZeemanLevel(energy={self.energy/Constants.h/Units.THz} THz, n={self.n}, I={self.I}, L={self.L}, J={self.J}, m_J={self.m})"

    def __repr__(self):
        return self.__str__()


class HyperfineStructureZeemanLevel(EnergyLevel):
    def __init__(
        self,
        name: str,
        energy: float,
        n: int,
        I: Rational,
        L: int,
        J: Rational,
        F: Rational,
        m: float,
        line_width: float,
        branching_ratios: List[Dict[str, float]],
    ):
        super().__init__(name, energy, n, I, L, J, line_width, branching_ratios)
        self.F = F
        self.m = m
        self.lande_g_factor = 1 + (F * (F + 1) - J * (J + 1) + I * (I + 1)) / (
            2 * F * (F + 1)
        )
        self.zeeman_splitting_func: Callable[[float], float] = (
            lambda B_field: self.lande_g_factor * self.m * Constants.mu_B * B_field
        )

    def apply_magnetic_field(self, magentic_field: float):
        self.energy = self.energy + self.zeeman_splitting_func(magentic_field)

    def __str__(self):
        return f"HyperfineStructureZeemanLevel(energy={self.energy/Constants.h/Units.THz} THz, n={self.n}, I={self.I}, L={self.L}, J={self.J}, F={self.F}, m_F={self.m})"

    def __repr__(self):
        return self.__str__()


class FineStructure(EnergyLevel):
    def __init__(
        self,
        name: str,
        energy: float,
        n: int,
        I: Rational,
        L: int,
        J: Rational,
        line_width: float,
        branching_ratios: List[Dict[str, float]],
    ):
        super().__init__(name, energy, n, I, L, J, line_width, branching_ratios)
        self.n_zeeman_levels = 2 * J + 1
        self.lande_g_factor = 1 + (J * (J + 1) - L * (L + 1) + 0.5 * (0.5 + 1)) / (
            2 * J * (J + 1)
        )
        self.zeeman_levels = [
            FineStructureZeemanLevel(
                self.name,
                self.energy,
                self.n,
                self.I,
                self.L,
                self.J,
                m_J,
                self.line_width,
                self.branching_ratios,
            )
            for m_J in np.arange(-J, J + 1)
        ]
        self.spontaneous_emission: Dict[str, float] = {}

    def apply_magnetic_field(self, magentic_field: float):
        for zeeman_level in self.zeeman_levels:
            zeeman_level.energy = self.energy
            zeeman_level.apply_magnetic_field(magentic_field)

    def __str__(self):
        return f"FineStructure(energy={self.energy/Constants.h/Units.THz} THz, n={self.n}, I={self.I}, L={self.L}, J={self.J})"

    def __repr__(self):
        return self.__str__()


class HyperfineStructure(EnergyLevel):
    def __init__(
        self,
        name: str,
        energy: float,
        n: int,
        I: Rational,
        L: int,
        J: Rational,
        F: Rational,
        line_width: float,
        branching_ratios: List[Dict[str, float]],
    ):
        super().__init__(name, energy, n, I, L, J, line_width, branching_ratios)
        self.F = F
        self.n_zeeman_levels = 2 * F + 1
        self.lande_g_factor = 1 + (F * (F + 1) - J * (J + 1) + I * (I + 1)) / (
            2 * F * (F + 1)
        )
        self.zeeman_levels = [
            HyperfineStructureZeemanLevel(
                self.name,
                self.energy,
                self.n,
                self.I,
                self.L,
                self.J,
                self.F,
                m_F,
                self.line_width,
                self.branching_ratios,
            )
            for m_F in np.arange(-F, F + 1)
        ]
        self.spontaneous_emission: Dict[str, float] = {}

    def apply_magnetic_field(self, magentic_field: float):
        for zeeman_level in self.zeeman_levels:
            zeeman_level.energy = self.energy
            zeeman_level.apply_magnetic_field(magentic_field)

    def __str__(self):
        return f"HyperfineStructure(energy={self.energy/Constants.h/Units.THz} THz, n={self.n}, I={self.I}, L={self.L}, J={self.J}, F={self.F})"

    def __repr__(self):
        return self.__str__()
