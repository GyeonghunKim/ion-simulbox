import os
import json
from typing import List, Dict
from collections import defaultdict
from itertools import combinations
import numpy as np
from .energy_level import EnergyLevel, FineStructure, HyperfineStructure
from .units import Units, Constants
from .utils import L_str_to_int, number_to_sympy, sympy_to_number
from sympy.physics.wigner import wigner_3j

class Ion:
    def __init__(self, species: str, mass_number: int):
        self.species = species
        self.mass_number = mass_number
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.library_name = os.path.join(
            current_dir,
            f"../ion_library/{self.species}_II/{self.species}-{self.mass_number}.json",
        )
        self.library = json.load(open(self.library_name))
        self.energy_levels: List[EnergyLevel] = []
        self.branching_ratios: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.decay_channels = {}
        self._load_energy_levels()
        self._load_branching_ratios()
        self._setup_decay_channels()
    @property
    def zeeman_levels(self):
        return {level: level.zeeman_levels for level in self.energy_levels}
    
    def get_level(self, name):
        for level in self.energy_levels:
            if level.name == name:
                return level
        raise ValueError(f"Level {name} not found")
    
    def _load_branching_ratios(self):
        for branching_ratio in self.library["branching_ratios"]:
            self.branching_ratios[branching_ratio["upper_level"]][
                self.get_level(branching_ratio["lower_level"])
            ] = branching_ratio["branching_ratio"]

    def _load_energy_levels(self):
        for level in self.library["energy_levels"]:
            if level["order"] == "FineStructure":
                self.energy_levels.append(
                    FineStructure(
                        str(level["n"]) + level["L"] + str(level["J"]),
                        level["energy_Hz"] * Constants.h,
                        level["n"],
                        self.library["I"],
                        L_str_to_int(level["L"]),
                        level["J"],
                        2 * np.pi * level["line_width_2_pi_Hz"],
                        self.branching_ratios[
                            str(level["n"]) + level["L"] + str(level["J"])
                        ],
                    )
                )
            elif level["order"] == "HyperfineStructure":
                self.energy_levels.append(
                    HyperfineStructure(
                        str(level["n"])
                        + level["L"]
                        + str(level["J"])
                        + str(level["F"]),
                        level["energy_Hz"] * Constants.h,
                        level["n"],
                        self.library["I"],
                        L_str_to_int(level["L"]),
                        level["J"],
                        level["F"],
                        2 * np.pi * level["line_width_2_pi_Hz"],
                        self.branching_ratios[
                            str(level["n"])
                            + level["L"]
                            + str(level["J"])
                            + str(level["F"])
                        ],
                    )
                )

    def apply_magnetic_field(self, B_field: float):
        self.B_field = B_field
        for level in self.energy_levels:
            level.apply_magnetic_field(B_field)

    
    def _setup_decay_channels(self):
        for level_from in self.energy_levels:
            for level_to, branching_ratio in self.branching_ratios[level_from.name].items():
                if abs(level_from.L - level_to.L) > 1:
                    continue
                elif np.isclose(level_from.L, 2) or np.isclose(level_from.L, 3):
                    continue
 
                for zeeman_level_from in level_from.zeeman_levels:
                    for zeeman_level_to in level_to.zeeman_levels:
                        cg_squared = (
                            abs(
                                sympy_to_number(
                                    np.sqrt(2 * zeeman_level_from.J + 1) * wigner_3j(
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
                            ] = branching_ratio * zeeman_level_from.line_width * cg_squared
        # # Setup decay channels
        # for level_from in self.levels:
        #     for level_to_name, branching_ratio in self.ion.branching_ratios[
        #         level_from.name
        #     ].items():
        #         if level_to_name in level_name_map:
        #             level_to = level_name_map[level_to_name]
        #             for zeeman_level_from in level_from.zeeman_levels:
        #                 for zeeman_level_to in level_to.zeeman_levels:
        #                     cg_coefficient = (
        #                         abs(
        #                             sympy_to_number(
        #                                 wigner_3j(
        #                                     number_to_sympy(zeeman_level_from.J),
        #                                     number_to_sympy(1),
        #                                     number_to_sympy(zeeman_level_to.J),
        #                                     number_to_sympy(zeeman_level_from.m),
        #                                     number_to_sympy(
        #                                         zeeman_level_to.m - zeeman_level_from.m
        #                                     ),
        #                                     number_to_sympy(-1 * zeeman_level_to.m),
        #                                 )
        #                             )
        #                         )
        #                         ** 2
        #                     )
        #                     self.decay_channels[
        #                         (zeeman_level_from, zeeman_level_to)
        #                     ] = (
        #                         branching_ratio
        #                         * zeeman_level_from.line_width
        #                         * cg_coefficient
        #                     )
        #                     L = np.sqrt(
        #                         self.decay_channels[
        #                             (zeeman_level_from, zeeman_level_to)
        #                         ]
        #                     ) * (
        #                         self.kets[zeeman_level_to]
        #                         * self.kets[zeeman_level_from].dag()
        #                     )
        #                     self.c_ops.append(L)
        #         else:
        #             level_to = self.total_level_name_map[level_to_name]
        #             for zeeman_level_from in level_from.zeeman_levels:
        #                 for zeeman_level_to in level_to.zeeman_levels:
        #                     cg_coefficient = (
        #                         abs(
        #                             sympy_to_number(
        #                                 wigner_3j(
        #                                     number_to_sympy(zeeman_level_from.J),
        #                                     number_to_sympy(1),
        #                                     number_to_sympy(zeeman_level_to.J),
        #                                     number_to_sympy(zeeman_level_from.m),
        #                                     number_to_sympy(
        #                                         zeeman_level_to.m - zeeman_level_from.m
        #                                     ),
        #                                     number_to_sympy(-1 * zeeman_level_to.m),
        #                                 )
        #                             )
        #                         )
        #                         ** 2
        #                     )
        #                     self.decay_channels[(zeeman_level_from, "dummy")] = (
        #                         branching_ratio
        #                         * zeeman_level_from.line_width
        #                         * cg_coefficient
        #                     )
        #                     L = np.sqrt(
        #                         self.decay_channels[(zeeman_level_from, "dummy")]
        #                     ) * (
        #                         self.kets["dummy"] * self.kets[zeeman_level_from].dag()
        #                     )
        #                     self.c_ops.append(L)


if __name__ == "__main__":
    ion = Ion("Ba", 138)
    for energy_level in ion.energy_levels:
        print(energy_level)
        for zeeman_level in energy_level.zeeman_levels:
            print(zeeman_level)
