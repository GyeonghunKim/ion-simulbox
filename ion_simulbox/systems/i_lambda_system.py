import qutip
import numpy as np
from itertools import product
from ..energy_level import EnergyLevel
from ..transition import Transition
from ..units import Constants


class ILambdaSystem:
    def __init__(
        self,
        ion,
        i_lower_states,
        i_upper_states,
        lambda_lower_states_1,
        lambda_lower_states_2,
        lambda_upper_states,
        i_laser,
        lambda_laser_1,
        lambda_laser_2,
        i_envelope_function,
        lambda_envelope_function_1,
        lambda_envelope_function_2,
        i_reference_state,
        lambda_reference_state,
    ):
        self.ion = ion
        self.i_lower_states = i_lower_states
        self.i_upper_states = i_upper_states
        self.lambda_lower_states_1 = lambda_lower_states_1
        self.lambda_lower_states_2 = lambda_lower_states_2
        self.lambda_upper_states = lambda_upper_states
        self.n_system = (
            len(i_lower_states)
            + len(i_upper_states)
            + len(lambda_lower_states_1)
            + len(lambda_lower_states_2)
            + len(lambda_upper_states)
            + 1
        )
        self.i_reference_state = i_reference_state
        self.lambda_reference_state = lambda_reference_state
        self.i_laser = i_laser
        self.lambda_laser_1 = lambda_laser_1
        self.lambda_laser_2 = lambda_laser_2
        self.i_envelope_function = i_envelope_function
        self.lambda_envelope_function_1 = lambda_envelope_function_1
        self.lambda_envelope_function_2 = lambda_envelope_function_2
        self._setup_basis()
        self._setup_collapse_ops()
        self._setup_atomic_hamiltonian()
        self._setup_laser_hamiltonian()
        self._setup_measurement_ops()

    def _setup_basis(self):
        self.basis = {}
        i_basis = 0
        for state in self.i_lower_states:
            self.basis[state] = qutip.basis(self.n_system, i_basis)
            i_basis = i_basis + 1
        for state in self.i_upper_states:
            self.basis[state] = qutip.basis(self.n_system, i_basis)
            i_basis = i_basis + 1
        for state in self.lambda_lower_states_1:
            self.basis[state] = qutip.basis(self.n_system, i_basis)
            i_basis = i_basis + 1
        for state in self.lambda_lower_states_2:
            self.basis[state] = qutip.basis(self.n_system, i_basis)
            i_basis = i_basis + 1
        for state in self.lambda_upper_states:
            self.basis[state] = qutip.basis(self.n_system, i_basis)
            i_basis = i_basis + 1
        self.basis["dummy"] = qutip.basis(self.n_system, i_basis)

    def _setup_collapse_ops(self):
        self.collapse_ops = []
        for (state_from, state_to), coef in self.ion.decay_channels.items():
            if state_from in self.basis.keys():
                if state_to in self.basis.keys():
                    # if "6P0.5" in state_from.name and "6S0.5" in state_to.name:
                    #     print(
                    #         f"Adding collapse operator for {state_from} -> {state_to} with coefficient {coef / 3.92}"
                    #     )
                    #     self.collapse_ops.append(
                    #         np.sqrt(coef * 3.92)
                    #         * self.basis[state_to]
                    #         * self.basis[state_from].dag()
                    #     )
                    # else:
                    self.collapse_ops.append(
                        np.sqrt(coef)
                        * self.basis[state_to]
                        * self.basis[state_from].dag()
                    )
                else:
                    self.collapse_ops.append(
                        np.sqrt(coef)
                        * self.basis["dummy"]
                        * self.basis[state_from].dag()
                    )

    def _setup_atomic_hamiltonian(self):
        self.H0 = qutip.qzero(self.n_system)
        ## i system part
        for state in self.i_lower_states:
            self.H0 = (
                self.H0
                + 2
                * np.pi
                * (state.energy - self.i_reference_state.energy)
                / Constants.h
                * self.basis[state]
                * self.basis[state].dag()
            )
        for state in self.i_upper_states:
            self.H0 = (
                self.H0
                + 2
                * np.pi
                * (
                    (state.energy - self.i_reference_state.energy) / Constants.h
                    - self.i_laser.frequency
                )
                * self.basis[state]
                * self.basis[state].dag()
            )

        for state in self.lambda_upper_states:
            self.H0 = (
                self.H0
                + 2
                * np.pi
                * (self.lambda_reference_state.energy - state.energy)
                / Constants.h
                * self.basis[state]
                * self.basis[state].dag()
            )

        for state in self.lambda_lower_states_1:
            self.H0 = (
                self.H0
                + 2
                * np.pi
                * (
                    self.lambda_laser_1.frequency
                    - (self.lambda_reference_state.energy - state.energy) / Constants.h
                )
                * self.basis[state]
                * self.basis[state].dag()
            )
        for state in self.lambda_lower_states_2:
            self.H0 = (
                self.H0
                + 2
                * np.pi
                * (
                    self.lambda_laser_2.frequency
                    - (self.lambda_reference_state.energy - state.energy) / Constants.h
                )
                * self.basis[state]
                * self.basis[state].dag()
            )

    def _setup_laser_hamiltonian(self):
        self.H_i_laser = qutip.qzero(self.n_system)
        self.H_lambda_laser_1 = qutip.qzero(self.n_system)
        self.H_lambda_laser_2 = qutip.qzero(self.n_system)

        for state_1, state_2 in product(self.i_lower_states, self.i_upper_states):
            transition = Transition(state_1, state_2, self.i_laser)
            self.H_i_laser = (
                self.H_i_laser
                + transition.rabi_frequency
                / 2
                * self.basis[transition.lower_level]
                * self.basis[transition.upper_level].dag()
            )
            self.H_i_laser = (
                self.H_i_laser
                + transition.rabi_frequency.conj()
                / 2
                * self.basis[transition.upper_level]
                * self.basis[transition.lower_level].dag()
            )

        for state_1, state_2 in product(
            self.lambda_lower_states_1, self.lambda_upper_states
        ):
            transition = Transition(state_1, state_2, self.lambda_laser_1)
            self.H_lambda_laser_1 = (
                self.H_lambda_laser_1
                + transition.rabi_frequency
                / 2
                * self.basis[transition.lower_level]
                * self.basis[transition.upper_level].dag()
            )
            self.H_lambda_laser_1 = (
                self.H_lambda_laser_1
                + transition.rabi_frequency.conj()
                / 2
                * self.basis[transition.upper_level]
                * self.basis[transition.lower_level].dag()
            )

        for state_1, state_2 in product(
            self.lambda_lower_states_2, self.lambda_upper_states
        ):
            transition = Transition(state_1, state_2, self.lambda_laser_2)
            self.H_lambda_laser_2 = (
                self.H_lambda_laser_2
                + transition.rabi_frequency
                / 2
                * self.basis[transition.lower_level]
                * self.basis[transition.upper_level].dag()
            )
            self.H_lambda_laser_2 = (
                self.H_lambda_laser_2
                + transition.rabi_frequency.conj()
                / 2
                * self.basis[transition.upper_level]
                * self.basis[transition.lower_level].dag()
            )

    def _setup_measurement_ops(self):
        self.measurement_ops = []
        self.measurement_ops_names = []
        for state in self.i_lower_states:
            self.measurement_ops.append(self.basis[state] * self.basis[state].dag())
            self.measurement_ops_names.append(state.name)
        for state in self.i_upper_states:
            self.measurement_ops.append(self.basis[state] * self.basis[state].dag())
            self.measurement_ops_names.append(state.name)
        for state in self.lambda_upper_states:
            self.measurement_ops.append(self.basis[state] * self.basis[state].dag())
            self.measurement_ops_names.append(state.name)
        for state in self.lambda_lower_states_1:
            self.measurement_ops.append(self.basis[state] * self.basis[state].dag())
            self.measurement_ops_names.append(state.name)
        for state in self.lambda_lower_states_2:
            self.measurement_ops.append(self.basis[state] * self.basis[state].dag())
            self.measurement_ops_names.append(state.name)
        self.measurement_ops.append(self.basis["dummy"] * self.basis["dummy"].dag())
        self.measurement_ops_names.append("dummy")

    def solve(self, initial_state, t_list):
        result = qutip.mesolve(
            [
                self.H0,
                [self.H_i_laser, self.i_envelope_function],
                [self.H_lambda_laser_1, self.lambda_envelope_function_1],
                [self.H_lambda_laser_2, self.lambda_envelope_function_2],
            ],
            initial_state,
            t_list,
            c_ops=self.collapse_ops,
            e_ops=self.measurement_ops,
            options={"store_states": True},
        )
        return dict(zip(self.measurement_ops_names, result.expect)), result.final_state
