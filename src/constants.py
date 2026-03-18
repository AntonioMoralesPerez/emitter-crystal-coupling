import numpy as np

# Define constants
e = 1.602176634e-19  # Elementary charge [C]
H_BAR = 1.054571817e-34  # Reduced Planck constant [J·s]
c = 299792458  # Speed of light [m/s]
ε_0 = 8.8541878128e-12  # Vacuum permittivity [F/m]

K_CONST: float = e / np.sqrt(c * H_BAR * ε_0)  # Constant prefactor, dimensionless

LORENTIAN_BROADENING: float = 1e-2  # Broadening factor for LDOS calculations
