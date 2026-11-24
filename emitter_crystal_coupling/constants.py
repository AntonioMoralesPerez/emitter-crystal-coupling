import numpy as np

# Physical constants
E_CHARGE: float = 1.602176634e-19  # Elementary charge [C]
H_BAR: float = 1.054571817e-34  # Reduced Planck constant [J·s]
C_LIGHT: float = 299792458.0  # Speed of light in vacuum [m/s]
EPSILON_0: float = 8.8541878128e-12  # Vacuum permittivity [F/m]

# Derived constants
K_CONST: float = E_CHARGE / np.sqrt(
    C_LIGHT * H_BAR * EPSILON_0
)  # Dimensionless prefactor
# Broadening for LDOS calculations, in angular units ω a/c (the 2π keeps the
# width fixed relative to the band when frequencies are angular).
LORENTZIAN_BROADENING: float = 2 * np.pi * 1e-2
