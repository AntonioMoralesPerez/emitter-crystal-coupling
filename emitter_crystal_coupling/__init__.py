"""
Public API for `emitter-crystal-coupling`.

Exposes core utilities, sampling methods, Hamiltonian construction,
entropy measures, and field computations.
"""

__all__ = [
    "save",
    "load",
    "arrange_data",
    "find_reciprocal_lattice_vectors",
    "read_epsilon_file",
    "array_to_Vector3",
    "meshgrid_points",
    "convert_epsilon",
    "find_positions",
    "marker_size",
    "draw_arrow",
    "unit_dip",
    "free_spontaneous_emission",
    "monkhorst_pack_sampling",
    "compute_real_space_coupling",
    "compute_reciprocal_space_coupling",
    "compute_decay_rate",
    "make_light_matter_Hamiltonian",
    "evolve_diagonalisation",
    "binary_entropy",
    "quadrant_entropy",
    "compute_fr0rs",
    "safe_sqrt",
    "build_finite_grid",
]

# I/O and data utilities
from .utils import save, load, arrange_data, read_epsilon_file, convert_epsilon

# Geometry and spatial utilities
from .utils import (
    find_reciprocal_lattice_vectors,
    array_to_Vector3,
    meshgrid_points,
    find_positions,
)

# Visualization helpers
from .utils import marker_size, draw_arrow, unit_dip

# Numerical utilities
from .utils import safe_sqrt, build_finite_grid

# Physics / optical utilities
from .utils import free_spontaneous_emission
from .mpb_fields import compute_fr0rs

# Sampling
from .monkhorst_pack import monkhorst_pack_sampling

# Coupling and decay rate
from .coupling import (
    compute_real_space_coupling,
    compute_reciprocal_space_coupling,
    compute_decay_rate,
)

# Hamiltonian
from .hamiltonian import make_light_matter_Hamiltonian

# Time evolution
from .time_evolution import evolve_diagonalisation

# Entropy
from .entropy import binary_entropy, quadrant_entropy
