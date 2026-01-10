from sympy import S
import numpy as np
from .units import Constants
from .energy_level import EnergyLevel


def L_str_to_int(L_str: str) -> int:
    if L_str == "S":
        return 0
    elif L_str == "P":
        return 1
    elif L_str == "D":
        return 2
    elif L_str == "F":
        return 3
    else:
        raise ValueError(f"Invalid L value: {L_str}")


def number_to_sympy(number: float):
    if isinstance(number, int):
        return S(number)
    elif np.isclose(number, np.ceil(number)):
        return S(np.ceil(number))
    elif np.isclose(number, np.floor(number)):
        return S(np.floor(number))
    elif np.isclose(number % 1, 0.5):
        return np.floor(number) + S(1) / 2
    else:
        raise ValueError(f"Invalid number: {number}")


def sympy_to_number(sympy_number):
    if sympy_number.is_integer:
        return int(sympy_number)
    else:
        return float(sympy_number)


def get_resonant_frequency(level_1: EnergyLevel, level_2: EnergyLevel):
    return abs(level_1.energy - level_2.energy) / Constants.h
