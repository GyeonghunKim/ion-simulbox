import os
import json
from typing import List, Dict
from collections import defaultdict
import numpy as np
from .energy_level import EnergyLevel, FineStructure, HyperfineStructure
from .units import Units, Constants
from .utils import L_str_to_int


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
        self._load_branching_ratios()
        self._load_energy_levels()

    def _load_branching_ratios(self):
        for branching_ratio in self.library["branching_ratios"]:
            self.branching_ratios[branching_ratio["upper_level"]][
                branching_ratio["lower_level"]
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


if __name__ == "__main__":
    ion = Ion("Ba", 138)
    for energy_level in ion.energy_levels:
        print(energy_level)
        for zeeman_level in energy_level.zeeman_levels:
            print(zeeman_level)
