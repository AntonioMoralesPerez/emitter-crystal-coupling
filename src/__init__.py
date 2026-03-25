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
    "compute_ldos",
    "make_light_matter_Hamiltonian",
    "binary_entropy",
    "quadrant_entropy",
    "compute_ERr0",
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

# Physics / optical utilities
from .utils import free_spontaneous_emission
from .mpb_fields import compute_fr0rs

# Sampling
from .monkhorst_pack import monkhorst_pack_sampling

# Coupling and LDOS
from .coupling import compute_real_space_coupling, compute_ldos

# Hamiltonian
from .hamiltonian import make_light_matter_Hamiltonian

# Entropy
from .entropy import binary_entropy, quadrant_entropy
