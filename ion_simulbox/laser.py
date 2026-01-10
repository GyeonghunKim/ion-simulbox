from .units import Constants
import numpy as np


class Polarization:
    def __init__(self, k_hat: np.ndarray, epsilon_0: complex, epsilon_1: complex):
        """
        polarization_vector is a unit vector in the direction of the polarization of the laser.
        epsilon_0 is polarization component in same plane with B_field and k vector.
        epsilon_1 is polarization component in perpendicular to B_field and k vector plane. Since B field is set to z direction,
        this vector should on the xy plane
        """
        self.epsilon_0 = epsilon_0
        self.epsilon_1 = epsilon_1
        self.k_hat = k_hat / np.linalg.norm(k_hat)
        z_hat = np.array([0, 0, 1])
        if np.isclose(np.dot(self.k_hat, z_hat), 1):
            self.epsilon_0_unit_vector = np.array([1, 0, 0])
            self.epsilon_1_unit_vector = np.array([0, 1, 0])
        else:
            self.epsilon_0_unit_vector = z_hat - np.dot(z_hat, self.k_hat) * self.k_hat
            self.epsilon_0_unit_vector = self.epsilon_0_unit_vector / np.linalg.norm(
                self.epsilon_0_unit_vector
            )

        self.epsilon_1_unit_vector = np.cross(self.k_hat, self.epsilon_0_unit_vector)
        self.epsilon_in_cartesian = (
            self.epsilon_0 * self.epsilon_0_unit_vector
            + self.epsilon_1 * self.epsilon_1_unit_vector
        )
        self.epsilon_in_spherical_tensor = np.array(
            [
                (self.epsilon_in_cartesian[0] - 1j * self.epsilon_in_cartesian[1])
                / np.sqrt(2),
                self.epsilon_in_cartesian[2],
                -(self.epsilon_in_cartesian[0] + 1j * self.epsilon_in_cartesian[1])
                / np.sqrt(2),
            ]
        )  # in order of q = -1, q = 0, q = 1


class Laser:
    def __init__(
        self,
        name: str,
        frequency: float,
        intensity: float,
        line_width: float,
        polarization: Polarization,
    ):
        self.name = name
        self.frequency = frequency
        self.wavelength = Constants.c / frequency
        self.intensity = intensity
        self.line_width = line_width
        self.polarization = polarization
        self.k_hat = polarization.k_hat

    def get_frequency(self):
        return Constants.c / self.wavelength

    def get_electric_field_amplitude(self):
        return np.sqrt(2 * self.intensity / (Constants.c * Constants.epsilon_0))
