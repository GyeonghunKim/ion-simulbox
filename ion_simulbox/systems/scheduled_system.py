from typing import Dict, Callable, Optional, Any, List
from ..laser import Laser
import numpy as np
from ..laser import EnvelopeFactory
import itertools
import qutip


class ScheduledSystem:
    def __init__(
        self,
        ion,
        BaseSystem,
        system_parameters: Dict[str, Any],
        time_duration: float,
        base_lasers: List[Laser],
        base_envelope_functions: Dict[str, Callable[[float], float]],
    ):
        self.ion = ion
        self.BaseSystem = BaseSystem
        self.time_duration = time_duration
        self.system_parameters = system_parameters
        self.base_lasers = {k: laser for k, laser in base_lasers.items()}
        self.base_envelope_functions = {
            k: envelope_function
            for k, envelope_function in base_envelope_functions.items()
        }
        for k, laser in base_lasers.items():
            if laser.intensity > 0:
                base_lasers[k].intensity = 0

        self.schedule = {(0): (time_duration, "IDLE", None)}

    def add_schedule(
        self,
        time_start: float,
        time_end: float,
        laser_parameters: Dict[str, Laser],
        envelope_functions: Dict[str, Callable[[float], float]],
    ):
        for t_0, (t_1, lasers, _) in self.schedule.items():
            if t_0 <= time_start < t_1:
                if lasers != "IDLE":
                    raise Exception(f"Laser {lasers} is already scheduled at {t_0}")
                if time_end > t_1:
                    raise Exception(f"Laser {lasers} is already scheduled at {t_1}")
                remove_t_0 = t_0
                remove_t_1 = t_1
                break

        self.schedule.pop(remove_t_0)
        if not np.isclose(time_start, remove_t_0):
            self.schedule[remove_t_0] = (
                time_start,
                "IDLE",
                self.base_envelope_functions,
            )

        self.schedule[time_start] = (time_end, laser_parameters, envelope_functions)
        if not np.isclose(time_end, remove_t_1):
            self.schedule[time_end] = (remove_t_1, "IDLE", self.base_envelope_functions)

    def finalize_schedule(self):
        for t_0, (t_1, lasers, _) in self.schedule.items():
            if lasers == "IDLE":
                self.schedule[t_0] = (
                    t_1,
                    self.base_lasers,
                    self.base_envelope_functions,
                )

        self.compiled_schedule = []
        for t_0, (t_1, lasers, envelope_functions) in self.schedule.items():
            for k, envelope_function in envelope_functions.items():
                if envelope_function is None:
                    envelope_functions[k] = EnvelopeFactory.square_factory(
                        t_0, t_1 - t_0
                    )
                else:
                    envelope_functions[
                        k
                    ] = lambda t, envelope_function=envelope_function, t_0=t_0, t_1=t_1: envelope_function(
                        t
                    ) * EnvelopeFactory.square_factory(
                        t_0, t_1 - t_0
                    )(
                        t
                    )
            system = self.BaseSystem(
                self.ion, **self.system_parameters, **lasers, **envelope_functions
            )
            self.compiled_schedule.append(system)

        self._setup_basis()
        self._setup_collapse_ops()
        self._setup_atomic_hamiltonian()
        self._setup_laser_hamiltonian()
        self._setup_measurement_ops()

    def _setup_basis(self):
        self.basis = self.compiled_schedule[0].basis

    def _setup_collapse_ops(self):
        self.collapse_ops = self.compiled_schedule[0].collapse_ops

    def _setup_atomic_hamiltonian(self):
        self.H0 = self.compiled_schedule[0].H0

    def _setup_laser_hamiltonian(self):
        self.laser_hamiltonians = list(
            itertools.chain(
                *[system.laser_hamiltonians for system in self.compiled_schedule]
            )
        )
        self.laser_envelope_functions = list(
            itertools.chain(
                *[system.laser_envelope_functions for system in self.compiled_schedule]
            )
        )

    def _setup_measurement_ops(self):
        self.measurement_ops = self.compiled_schedule[0].measurement_ops
        self.measurement_ops_names = self.compiled_schedule[0].measurement_ops_names

    def solve(self, initial_state, t_list):
        result = qutip.mcsolve(
            [self.H0]
            + [
                [hamiltonian, envelope_function]
                for hamiltonian, envelope_function in zip(
                    self.laser_hamiltonians, self.laser_envelope_functions
                )
            ],
            initial_state,
            t_list,
            c_ops=self.collapse_ops,
            e_ops=self.measurement_ops,
            options={
                "store_states": True,
                # "nsteps": 50000,
                # "rtol": 1e-1,
                # "atol": 1e-16,
            },
            ntraj=1000,
        )
        return dict(zip(self.measurement_ops_names, result.expect)), result.final_state
