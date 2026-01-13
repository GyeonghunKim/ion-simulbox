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
from itertools import combinations
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
        magnetic_field: float,
    ):
        self.laser = laser
        self.magnetic_field = magnetic_field
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
        return self.upper_level.branching_ratios[self.lower_level.name]

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


class Experiment:
    def __init__(self, ion: Ion, magnetic_field: float):
        self.ion = ion
        self.magnetic_field = magnetic_field
        self.levels: List[EnergyLevel] = []
        self.transitions: List[Transition] = []
        self.ion.apply_magnetic_field(magnetic_field)
        self.lasers: List[Laser] = []
        self.decay_channels = {}
        self.kets = {}

        self.total_level_name_map = {}
        for level in self.ion.energy_levels:
            self.total_level_name_map[level.name] = level
        self.c_ops = []

    def add_levels(self, levels: List[EnergyLevel]):
        self.levels.extend(levels)

    def _collect_levels(self) -> List[EnergyLevel]:
        """Return a list of unique energy levels involved in the experiment."""
        levels = list(self.levels)
        for t in self.transitions:
            if t.lower_level not in levels:
                levels.append(t.lower_level)
            if t.upper_level not in levels:
                levels.append(t.upper_level)
        return levels

    def get_hamiltonian(self, using_rwa: bool = True):
        pass
        # TODO

    def solve(
        self,
        t_list: List[float],
        using_rwa: bool = True,
        initial_state=None,
    ):
        pass
        # TODO

    def compile(self):
        if len(self.levels) < 2:
            raise ValueError(
                "At least two levels are required to compile the experiment"
            )

        # Setup transitions
        if isinstance(self.levels[0], FineStructure):
            for level_1, level_2 in combinations(self.levels, 2):
                if level_1.energy > level_2.energy:
                    lower_level = level_2
                    upper_level = level_1
                else:
                    lower_level = level_1
                    upper_level = level_2

                # Temporal
                if abs(lower_level.L - upper_level.L) == 2:
                    print("Quadrupole transition not supported yet")
                    continue
                for laser in self.lasers:
                    for zeeman_level_1 in lower_level.zeeman_levels:
                        for zeeman_level_2 in upper_level.zeeman_levels:
                            self.transitions.append(
                                Transition(
                                    zeeman_level_1,
                                    zeeman_level_2,
                                    laser,
                                    self.magnetic_field,
                                )
                            )

        elif isinstance(self.levels[0], HyperfineStructure):
            raise ValueError("HyperfineStructure levels are not supported yet")
        else:
            raise ValueError("Unsupported level type")

        level_name_map = {}
        for level in self.levels:
            level_name_map[level.name] = level
            for zeeman_level in level.zeeman_levels:
                self.kets[zeeman_level] = None
        n_levels = len(self.kets) + 1
        for i, level in enumerate(self.kets.keys()):
            self.kets[level] = qutip.basis(n_levels, i)
        self.kets["dummy"] = qutip.basis(n_levels, n_levels - 1)

        self.level_name_map = level_name_map

        # Setup decay channels
        for level_from in self.levels:
            for level_to_name, branching_ratio in self.ion.branching_ratios[
                level_from.name
            ].items():
                if level_to_name in level_name_map:
                    level_to = level_name_map[level_to_name]
                    for zeeman_level_from in level_from.zeeman_levels:
                        for zeeman_level_to in level_to.zeeman_levels:
                            cg_coefficient = (
                                abs(
                                    sympy_to_number(
                                        wigner_3j(
                                            number_to_sympy(zeeman_level_from.J),
                                            number_to_sympy(1),
                                            number_to_sympy(zeeman_level_to.J),
                                            number_to_sympy(zeeman_level_from.m),
                                            number_to_sympy(
                                                zeeman_level_to.m - zeeman_level_from.m
                                            ),
                                            number_to_sympy(-1 * zeeman_level_to.m),
                                        )
                                    )
                                )
                                ** 2
                            )
                            self.decay_channels[
                                (zeeman_level_from, zeeman_level_to)
                            ] = (
                                branching_ratio
                                * zeeman_level_from.line_width
                                * cg_coefficient
                            )
                            L = np.sqrt(
                                self.decay_channels[
                                    (zeeman_level_from, zeeman_level_to)
                                ]
                            ) * (
                                self.kets[zeeman_level_to]
                                * self.kets[zeeman_level_from].dag()
                            )
                            self.c_ops.append(L)
                else:
                    level_to = self.total_level_name_map[level_to_name]
                    for zeeman_level_from in level_from.zeeman_levels:
                        for zeeman_level_to in level_to.zeeman_levels:
                            cg_coefficient = (
                                abs(
                                    sympy_to_number(
                                        wigner_3j(
                                            number_to_sympy(zeeman_level_from.J),
                                            number_to_sympy(1),
                                            number_to_sympy(zeeman_level_to.J),
                                            number_to_sympy(zeeman_level_from.m),
                                            number_to_sympy(
                                                zeeman_level_to.m - zeeman_level_from.m
                                            ),
                                            number_to_sympy(-1 * zeeman_level_to.m),
                                        )
                                    )
                                )
                                ** 2
                            )
                            self.decay_channels[(zeeman_level_from, "dummy")] = (
                                branching_ratio
                                * zeeman_level_from.line_width
                                * cg_coefficient
                            )
                            L = np.sqrt(
                                self.decay_channels[(zeeman_level_from, "dummy")]
                            ) * (
                                self.kets["dummy"] * self.kets[zeeman_level_from].dag()
                            )
                            self.c_ops.append(L)

    def add_laser(
        self, laser: Laser  # , transition_pair: List[Tuple[EnergyLevel, EnergyLevel]]
    ):
        self.lasers.append(laser)

    def plot_transitions(self):
        """Plot available transitions with their Rabi frequencies."""
        import matplotlib.pyplot as plt

        if not self.transitions:
            return

        # gather unique energy levels and corresponding magnetic quantum numbers
        levels = {}
        m_values = set()
        for t in self.transitions:
            for lev in (t.lower_level, t.upper_level):
                levels[lev] = lev.energy
                m_values.add(getattr(lev, "m", 0))

        # map magnetic quantum number to x coordinate
        m_list = sorted(m_values)
        m_to_x = {m: i for i, m in enumerate(m_list)}
        level_pos = {lev: m_to_x.get(getattr(lev, "m", 0)) for lev in levels}
        energies_thz = {
            lev: energy / (Constants.h * Units.THz) for lev, energy in levels.items()
        }

        # prepare colors for lasers
        unique_lasers = list(dict.fromkeys([t.laser for t in self.transitions]))
        colors = plt.cm.tab10(range(len(unique_lasers)))
        laser_color = {laser: colors[i] for i, laser in enumerate(unique_lasers)}

        max_rabi = max(
            abs(t.rabi_frequency) for t in self.transitions if t.rabi_frequency != 0
        )
        if max_rabi == 0:
            max_rabi = 1

        # plot energy levels as small horizontal lines instead of points
        for lev, pos in level_pos.items():
            energy = energies_thz[lev]
            plt.hlines(energy, pos - 0.3, pos + 0.3, color="black")
            plt.text(pos, energy, lev.name, ha="center", va="bottom", fontsize=8)

        # plot transitions with line width proportional to rabi frequency
        for t in self.transitions:
            if t.rabi_frequency == 0:
                continue
            mixing_angle = np.arcsin(
                abs(t.rabi_frequency)
                / np.sqrt(abs(t.rabi_frequency) ** 2 + t.detunning**2)
            )
            x1 = level_pos[t.lower_level]
            x2 = level_pos[t.upper_level]
            y1 = energies_thz[t.lower_level]
            y2 = energies_thz[t.upper_level]
            width = 1 + 4 * abs(t.rabi_frequency) / max_rabi
            plt.plot(
                [x1, x2],
                [y1, y2],
                color=laser_color[t.laser],
                linewidth=width,
                alpha=mixing_angle / (np.pi / 2),
            )

        # make legend for lasers
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color=laser_color[l], lw=2, label=l.name)
            for l in unique_lasers
        ]
        plt.legend(handles=legend_elements)

        # label x axis with m values
        plt.xticks(range(len(m_list)), [str(m) for m in m_list])
        plt.xlabel("m")
        plt.ylabel("Energy (THz)")
        plt.tight_layout()
        plt.show()
