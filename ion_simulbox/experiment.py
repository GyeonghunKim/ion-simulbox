from typing import List, Dict, Tuple
import numpy as np
from sympy import S
from sympy.physics.wigner import wigner_3j
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
        self.transition_order = self.get_transition_order()
        self.transition_linewidth = (
            self.upper_level.line_width
            + self.lower_level.line_width
            + self.laser.line_width
        )
        self.transition_branching_ratio = self.get_transition_branching_ratio()
        self.rabi_frequency = self.get_rabi_frequency()

    def get_transition_branching_ratio(self):
        return self.upper_level.branching_ratios[self.lower_level.name]

    def get_transition_energy(self):
        return self.level1.energy - self.level2.energy

    def get_transition_order(self):
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
        if self.transition_order == TransitionOrder.dipole:
            coefficient = (
                self.laser.get_electric_field_amplitude()
                / Constants.h_bar
                * np.sqrt(
                    3
                    * Constants.epsilon_0
                    * Constants.h_bar
                    * self.laser.wavelength**3
                    * self.transition_branching_ratio
                    * self.transition_linewidth
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
        elif self.transition_order == TransitionOrder.quadrupole:
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
        """Construct the system Hamiltonian.

        Parameters
        ----------
        using_rwa : bool, optional
            If ``True`` the interaction Hamiltonian is constructed under the
            rotating wave approximation.
        """
        import qutip as qt

        levels = self._collect_levels()
        n = len(levels)
        energies = [lev.energy / Constants.h_bar for lev in levels]
        H0 = qt.Qobj(np.diag(energies))

        H = [H0]

        for tr in self.transitions:
            i = levels.index(tr.lower_level)
            j = levels.index(tr.upper_level)
            op = qt.basis(n, j) * qt.basis(n, i).dag()
            if using_rwa:
                w0 = abs(tr.upper_level.energy - tr.lower_level.energy) / Constants.h
                delta = tr.laser.get_frequency() - w0

                def f_plus(t, args=None, Omega=tr.rabi_frequency, d=delta):
                    return 0.5 * Omega * np.exp(-1j * d * t)

                def f_minus(t, args=None, Omega=tr.rabi_frequency, d=delta):
                    return 0.5 * Omega * np.exp(1j * d * t)

                H.append([op, f_plus])
                H.append([op.dag(), f_minus])
            else:
                wL = tr.laser.get_frequency()

                def f(t, args=None, Omega=tr.rabi_frequency, w=wL):
                    return Omega * np.cos(w * t)

                H.append([op + op.dag(), f])

        return H, levels

    def solve(
        self,
        t_list: List[float],
        using_rwa: bool = True,
        initial_state=None,
    ):
        """Solve the Schr\u00f6dinger equation for the experiment."""
        import qutip as qt

        H, levels = self.get_hamiltonian(using_rwa=using_rwa)
        n = len(levels)
        if initial_state is None:
            initial_state = qt.basis(n, 0)

        e_ops = [qt.basis(n, i) * qt.basis(n, i).dag() for i in range(n)]
        result = qt.sesolve(H, initial_state, t_list, e_ops=e_ops)
        self._last_levels = levels
        self._last_result = result
        return result

    def plot_populations(self, result=None):
        """Plot state populations as a function of time."""
        import matplotlib.pyplot as plt

        if result is None:
            result = getattr(self, "_last_result", None)
            if result is None:
                return
        levels = getattr(self, "_last_levels", self._collect_levels())
        times = result.times
        for i, lev in enumerate(levels):
            plt.plot(times, result.expect[i], label=lev.name)
        plt.xlabel("Time (s)")
        plt.ylabel("Population")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def add_laser(
        self, laser: Laser, transition_pair: List[Tuple[EnergyLevel, EnergyLevel]]
    ):
        self.lasers.append(laser)
        for level_1, level_2 in transition_pair:
            if isinstance(level_1, FineStructure) or isinstance(
                level_1, HyperfineStructure
            ):
                if isinstance(level_2, FineStructure) or isinstance(
                    level_2, HyperfineStructure
                ):
                    # level_1 is a FineStructure or HyperfineStructure and level_2 is a FineStructure or HyperfineStructure
                    for level_1_zeeman_level in level_1.zeeman_levels:
                        for level_2_zeeman_level in level_2.zeeman_levels:
                            self.transitions.append(
                                Transition(
                                    level_1_zeeman_level,
                                    level_2_zeeman_level,
                                    laser,
                                    self.magnetic_field,
                                )
                            )
                else:
                    for zeeman_level in level_1.zeeman_levels:
                        self.transitions.append(
                            Transition(
                                zeeman_level, level_2, laser, self.magnetic_field
                            )
                        )
            else:
                if isinstance(level_2, FineStructure) or isinstance(
                    level_2, HyperfineStructure
                ):
                    # level_1 is a ZeemanLevel and level_2 is a FineStructure or HyperfineStructure
                    for zeeman_level in level_2.zeeman_levels:
                        self.transitions.append(
                            Transition(
                                level_1, zeeman_level, laser, self.magnetic_field
                            )
                        )
                else:  # level_1 is a ZeemanLevel and level_2 is ZeemanLevel
                    self.transitions.append(
                        Transition(level_1, level_2, laser, self.magnetic_field)
                    )

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
            x1 = level_pos[t.lower_level]
            x2 = level_pos[t.upper_level]
            y1 = energies_thz[t.lower_level]
            y2 = energies_thz[t.upper_level]
            width = 1 + 4 * abs(t.rabi_frequency) / max_rabi
            plt.plot([x1, x2], [y1, y2], color=laser_color[t.laser], linewidth=width)

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
