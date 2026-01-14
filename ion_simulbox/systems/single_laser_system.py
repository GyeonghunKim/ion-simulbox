import qutip
import numpy as np
from itertools import product
from ..energy_level import EnergyLevel
from ..transition import Transition
from ..units import Constants

class SingleLaserSystem:
    def __init__(self, ion, lower_states, upper_states, laser, envelope_function,reference_state):
        self.ion = ion
        self.lower_states = lower_states
        self.upper_states = upper_states
        self.n_system = len(lower_states) + len(upper_states) + 1
        self.reference_state = reference_state
        self.laser = laser
        self.envelope_function = envelope_function
        self._setup_basis()
        self._setup_collapse_ops()
        self._setup_atomic_hamiltonian()
        self._setup_laser_hamiltonian()
        self._setup_measurement_ops()
        
    def _setup_basis(self):
        self.basis = {}
        i_basis = 0
        for state in self.lower_states:
            self.basis[state] = qutip.basis(self.n_system, i_basis)
            i_basis = i_basis + 1
        for state in self.upper_states:
            self.basis[state] = qutip.basis(self.n_system, i_basis)
            i_basis = i_basis + 1
        self.basis["dummy"] = qutip.basis(self.n_system, i_basis)

    def _setup_collapse_ops(self):
        self.collapse_ops = []
        for (state_from, state_to), coef in self.ion.decay_channels.items():
            if state_from in self.basis.keys():
                if state_to in self.basis.keys():
                    self.collapse_ops.append(
                        np.sqrt(coef) * self.basis[state_to] * self.basis[state_from].dag()
                    )
                else:
                    self.collapse_ops.append(
                        np.sqrt(coef) * self.basis["dummy"] * self.basis[state_from].dag()
                    )
           
    def _setup_atomic_hamiltonian(self):
        self.H0 = qutip.qzero(self.n_system)
        for state in self.lower_states:
            self.H0 = self.H0 + 2 * np.pi * (state.energy - self.reference_state.energy) / Constants.h * self.basis[state] * self.basis[state].dag()
        for state in self.upper_states:
            self.H0 = self.H0 + 2 * np.pi * ((state.energy - self.reference_state.energy) / Constants.h - self.laser.frequency) * self.basis[state] * self.basis[state].dag()
    
    def _setup_laser_hamiltonian(self):
        self.H_laser = qutip.qzero(self.n_system)
        for (state_1, state_2) in product(self.lower_states, self.upper_states):
            transition = Transition(state_1, state_2, self.laser)
            self.H_laser = self.H_laser + transition.rabi_frequency / 2 * self.basis[transition.lower_level] * self.basis[transition.upper_level].dag()
            self.H_laser = self.H_laser + transition.rabi_frequency.conj() / 2 * self.basis[transition.upper_level] * self.basis[transition.lower_level].dag()

    def _setup_measurement_ops(self):
        self.measurement_ops = []
        self.measurement_ops_names = []
        for state in self.lower_states:
            self.measurement_ops.append(self.basis[state] * self.basis[state].dag())
            self.measurement_ops_names.append(state.name)
        for state in self.upper_states:
            self.measurement_ops.append(self.basis[state] * self.basis[state].dag())
            self.measurement_ops_names.append(state.name)
        self.measurement_ops.append(self.basis["dummy"] * self.basis["dummy"].dag())
        self.measurement_ops_names.append("dummy")


    def solve(self, initial_state, t_list):
        result = qutip.mesolve(
            [self.H0, [self.H_laser, self.envelope_function]], initial_state, t_list, c_ops=self.collapse_ops, e_ops=self.measurement_ops,
            options={"store_states": True},
        )
        return dict(zip(self.measurement_ops_names, result.expect)), result.final_state