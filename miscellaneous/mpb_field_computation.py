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
NUM_BANDS = 1

EPSILON_INPUT_FILE = "data/kimble/epsilon.h5"
DIRNAME_OUTPUT = "data/kimble/fields/"  # directory to save outputs

NK = 128  # number of points per direction for the Monkhorst-Pack grid

# Positions of the quantum emitters in units of the lattice constant 'a'.
# Sample every emitter position within the SAME MPB run so all u_k(r) share a
# consistent k-dependent global phase (gauge). One file per position is then
# saved, sliced from this single run.
R0_LIST = [[0, 0], [-5, 5]]

# ------------------------------------------------------------------------------

# Read some data from the epsilon.h5 file
epsilon_data = mf.read_epsilon_file(EPSILON_INPUT_FILE)
resolution = epsilon_data["resolution"]
lattice_vecs = epsilon_data["lattice_vectors"]
lattice_size = epsilon_data["lattice_size"]

# Print summary of parameters
print("\n")
print(f"Number of bands      = {NUM_BANDS}")
print(f"Epsilon file used    = {EPSILON_INPUT_FILE}")
print(f"Resolution           = {resolution}")
print(f"Lattice size         = {lattice_size}")
print(f"Lattice vectors      = {lattice_vecs}")

# ------------------------------------------------------------------------------
# Monkhorst–Pack sampling of the Brillouin zone
# ------------------------------------------------------------------------------

# Compute reciprocal lattice vectors (without the 2π factor, since MPB expects that)
b1, b2, _ = mf.find_reciprocal_lattice_vectors(lattice_vecs, twopi=False)

# Generate k-point grid (in lattice coordinates by default)
k_points_array = mf.monkhorst_pack_sampling(
    n_points=NK, b1=b1, b2=b2, coordinates="lattice"
)

# Convert k-points into Meep/MPB Vector3 objects
k_points = mf.array_to_Vector3(k_points_array)

# ------------------------------------------------------------------------------
# Configure the MPB geometry lattice
# ------------------------------------------------------------------------------
geometry_lattice = mp.Lattice(
    size=lattice_size,
    basis1=lattice_vecs[0],
    basis2=lattice_vecs[1],
    # basis3 omitted for 2D computation; MPB treats it implicitly
)

# ------------------------------------------------------------------------------
# Build the MPB mode solver
# ------------------------------------------------------------------------------
ms = mpb.ModeSolver(
    num_bands=NUM_BANDS,
    resolution=mp.Vector3(*resolution),
    epsilon_input_file=EPSILON_INPUT_FILE,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
)

# Map all k-points into the first Brillouin zone (important for symmetry)
ms.k_points = [ms.first_brillouin_zone(k) for k in ms.k_points]

# ------------------------------------------------------------------------------
# Field extraction callback
# ------------------------------------------------------------------------------

# Ensure R0 is a 2D array (1 row if single point)
r0pts_in = np.atleast_2d(R0_LIST)  # returns array of shape (1,:) if R0 was 1D


def make_efield_collector(r0pts):
    collected = []

    def get_efields(ms, band):
        ms.get_efield(band, bloch_phase=False)
        for R0 in r0pts:
            collected.append(ms.get_bloch_field_point(mp.Vector3(*R0)))

    return get_efields, collected


get_efields, fields_MPBarr = make_efield_collector(r0pts_in)

# ------------------------------------------------------------------------------
# Run the MPB simulation (TE modes)
# ------------------------------------------------------------------------------
ms.run_te(mpb.fix_hfield_phase, get_efields)  # ensure consistent phase conventions

# ------------------------------------------------------------------------------
# Process and reorganize MPB output data
# ------------------------------------------------------------------------------

# Convert MPB results into NumPy arrays
fields_in = np.asarray(fields_MPBarr)  # shape: (Nb*Nk*Nr, 3)
kpts_in = np.asarray(ms.k_points)  # Meep Vector3 → lattice coords
freqs_in = np.asarray(ms.all_freqs)  # Frequencies for all bands

# Reorganize the raw MPB output into a structured dictionary
arranged_data = mf.arrange_data(
    fields_in=fields_in,
    kpts_in=kpts_in,
    r0pts_in=r0pts_in,
    freqs_in=freqs_in,
    lattice_vecs=lattice_vecs,
)

# Extract cleaned and arranged fields
freqs = arranged_data["frequencies"]
Ek = arranged_data["Eks"]  # shape: (Nr, Nb, Nk, 3)

kpts = arranged_data["kpts_cartesian"]
r0pts = arranged_data["r0pts_cartesian"]

# ------------------------------------------------------------------------------
# Package results and save one file per r0, all sliced from the same MPB run
# so the relative gauge of u_k(r) across positions is preserved.
# ------------------------------------------------------------------------------
for i, R0 in enumerate(R0_LIST):
    output = {
        "freqs": freqs,
        "Eks": Ek[i : i + 1],  # keep (Nr=1, Nb, Nk, 3) layout
        "kpts": kpts,
        "r0pts": r0pts[i : i + 1],
    }
    mf.save(
        output,
        filename=f"{NK}_meshgrid_at_{list(R0)}.pkl",
        dirname=DIRNAME_OUTPUT,
    )

# ------------------------------------------------------------------------------
