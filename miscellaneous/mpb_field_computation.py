"""
    Docstring for mpb_field_computation.py

This file is just a helper script for just computing the electric at a certain
position in the unit cell field using MPB and storing it for later use.

This is usually the most time-consuming computation, so we made a dedicated script
for it. This way it can be computed isolated (or even on a cluster) and then use
the result within our formalism.
"""

import numpy as np
import meep as mp  # needed for the numerical simulation of the photonic crystal
from meep import mpb  # same case as the previous package

import src as mf  # our package functions to work with

# %% Define the main constants and parameters for the current simulation
NUM_BANDS = 2

EPSILON_INPUT_FILE = "data/kimble/epsilon.h5"
DIRNAME_OUTPUT = "data/kimble/"  # directory to save outputs

TOLERANCE = 1e-12  # tolerance for the MPB calculations

NK = 64  # number of points per direction for the Monkhorst-Pack grid

# Position of the quantum emitter in units of the lattice constant 'a'
# list of positions are also allowed
R0_LIST = [[0.5, 0.5]]

# ---------------------------------------------------------------------------- #

# Read some data from the epsilon.h5 file
d = mf.read_epsilon_file(EPSILON_INPUT_FILE)
resolution = d["resolution"]
lattice_vecs = d["lattice_vectors"]
lattice_size = d["lattice_size"]

# Print summary of parameters
print("\n")
print("Number of bands      = " + str(NUM_BANDS))
print("Epsilon file used    = " + EPSILON_INPUT_FILE)
print("Resolution           = " + str(resolution))
print("Lattice size         = " + str(lattice_size))
print("Lattice vectors      = " + str(lattice_vecs))
print("Tolerance            = " + str(TOLERANCE))

# ------------------------------------------------------------------------ #
# Monkhorst–Pack sampling of the Brillouin zone
# ------------------------------------------------------------------------ #

# Compute reciprocal lattice vectors (without the 2π factor, since MPB expects that)
b1, b2, b3 = mf.find_reciprocal_lattice_vectors(lattice_vecs, twopi=False)

# Generate k-point grid (in lattice coordinates by default)
k_points_array = mf.monkhorst_pack_sampling(N=NK, b1=b1, b2=b2, coordinates="lattice")

# Convert k-points into Meep/MPB Vector3 objects
k_points = mf.array_to_Vector3(k_points_array)

# ------------------------------------------------------------------------ #
# Configure the MPB geometry lattice
# ------------------------------------------------------------------------ #
geometry_lattice = mp.Lattice(
    size=lattice_size,
    basis1=lattice_vecs[0],
    basis2=lattice_vecs[1],
    # basis3 omitted for 2D computation; MPB treats it implicitly
)

# ------------------------------------------------------------------------ #
# Build the MPB mode solver
# ------------------------------------------------------------------------ #
ms = mpb.ModeSolver(
    num_bands=NUM_BANDS,
    resolution=mp.Vector3(*resolution),  # type: ignore[arg-type]
    epsilon_input_file="./data/kimble/epsilon.h5",
    mesh_size=1,
    tolerance=TOLERANCE,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
)

# Map all k-points into the first Brillouin zone (important for symmetry)
ms.k_points = [ms.first_brillouin_zone(k) for k in ms.k_points]

# ------------------------------------------------------------------------ #
# Field extraction callback
# ------------------------------------------------------------------------ #
fields_MPBarr = []

# Ensure R0 is a 2D array (1 row if single point)
r0pts_in = np.atleast_2d(R0_LIST)  # returns array of shape (1,:) if R0 was 1D


def get_efields(ms, band):
    """
    Extract the electric-field Bloch periodic functions for the given band.
    """
    ms.get_efield(band, bloch_phase=False)
    for R0 in r0pts_in:
        fields_MPBarr.append(ms.get_bloch_field_point(mp.Vector3(*R0)))


# ------------------------------------------------------------------------ #
# Run the MPB simulation (TE modes)
# ------------------------------------------------------------------------ #
ms.run_te(mpb.fix_hfield_phase, get_efields)  # ensure consistent phase conventions

# ------------------------------------------------------------------------ #
# Process and reorganize MPB output data
# ------------------------------------------------------------------------ #

# Convert MPB results into NumPy arrays
fields_in = np.asarray(fields_MPBarr)  # shape: (Nb*Nk*Nr, 3)
kpts_in = np.asarray(ms.k_points)  # Meep Vector3 → lattice coords
freqs_in = np.asarray(ms.all_freqs)  # Frequencies for all bands

# Reorganize the raw MPB output into a structured dictionary
d = mf.arrange_data(
    fields_in=fields_in,
    kpts_in=kpts_in,
    r0pts_in=r0pts_in,
    freqs_in=freqs_in,
    lattice_vecs=lattice_vecs,
)

# Extract cleaned and arranged fields
freqs = d["frequencies"]
Ek = d["Eks"]  # shape: (Nr, Nb, Nk, 3)

kpts = d["kpts_cartesian"]
r0pts = d["r0pts_cartesian"]

# ------------------------------------------------------------------------ #
# Package results and save to disk
# ------------------------------------------------------------------------ #
output = {
    "freqs": freqs,
    "Eks": Ek,
    "kpts": kpts,
    "r0pts": r0pts,
}

mf.save(
    output,
    filename=f"{NK}_meshgrid_at_{R0_LIST[0]}.pkl",
    dirname="./data/kimble/fields/",
)

# ---------------------------------------------------------------------------- #
