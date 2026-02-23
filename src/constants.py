import numpy as np

# Define constants
E = 1.602176634e-19  # Elementary charge [C]
HBAR = 1.054571817e-34  # Reduced Planck constant [J·s]
C = 299792458  # Speed of light [m/s]
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity [F/m]

K_CONST: float = E / np.sqrt(C * HBAR * EPSILON_0)  # Constant prefactor, dimensionless

LORENTIAN_BROADENING: float = 1e-2  # Broadening factor for LDOS calculations
