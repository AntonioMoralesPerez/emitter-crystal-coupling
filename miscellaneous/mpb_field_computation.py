"""
Compute the dielectric function and per-k-point Bloch electric fields for the
2D photonic crystal at a set of emitter positions.

The geometry is a square lattice of air cylinders (radius `RADIUS`, ε=`EPS_CYL`)
in a dielectric background (ε=`EPS_BG`). All positions in `R0_LIST` are sampled
within a single MPB run so the k-dependent gauge of u_k(r) is consistent
across positions. Outputs:

  - `./data/kimble/{RESOLUTION}-epsilon.h5`: dielectric tensor at the chosen
    spatial resolution, written via MPB's `output_epsilon`. Consumed by
    `kimble.py` and other scripts that rebuild a ModeSolver from
    `epsilon_input_file=`.
  - `./data/kimble/fields/{NK}_meshgrid_at_{r0}.pkl`: one pickle per emitter
    position with frequencies, k-points, and E_{n,k}(r0).

Run from the `miscellaneous/` directory.
"""

import meep as mp
import numpy as np
from meep import mpb

import emitter_crystal_coupling as mf

# %% Geometry: square lattice of air cylinders in dielectric background
RADIUS = 103 / 290  # cylinder radius [a]
EPS_BG = 4  # dielectric background
EPS_CYL = 1  # cylinder material (air)
LATTICE_SIZE = [1.0, 1.0, 0.0]
LATTICE_VECS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# %% MPB simulation parameters
NUM_BANDS = 1
NK = 81  # k-points per direction (Monkhorst-Pack)
RESOLUTION = 128  # spatial grid points per lattice unit
TOLERANCE = 1e-12  # eigensolver tolerance

# %% Output paths
EPSILON_OUTPUT_DIR = "../data/kimble/"
EPSILON_OUTPUT_FILE = f"{EPSILON_OUTPUT_DIR}{RESOLUTION}-epsilon.h5"
DIRNAME_OUTPUT = "../data/kimble/fields/"

# %% Emitter positions [units of lattice constant a].
# All positions sampled in the same MPB run for a consistent gauge.
R0_LIST = [
    [0, 0],
    [-5, 5],
    [5, 5],
    [-10, 10],
    [-15, 15],
    [-20, 20],
    [-25, 25],
    [-30, 30],
]

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
print(f"\nNumber of bands      = {NUM_BANDS}")
print(f"Resolution           = {RESOLUTION}")
print(f"k-grid (NK × NK)     = {NK} × {NK}")
print(f"Tolerance            = {TOLERANCE:g}")
print(f"Lattice size         = {LATTICE_SIZE}")
print(f"Lattice vectors      = {LATTICE_VECS}")
print(f"Cylinder radius      = {RADIUS:.6f} a")
print(f"ε background / cyl   = {EPS_BG} / {EPS_CYL}")

# ------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------
geometry = [
    mp.Cylinder(
        center=mp.Vector3(0, 0, 0),
        radius=RADIUS,
        material=mp.Medium(epsilon=EPS_CYL),
    )
]

geometry_lattice = mp.Lattice(
    size=mp.Vector3(*LATTICE_SIZE),
    basis1=mp.Vector3(*LATTICE_VECS[0]),
    basis2=mp.Vector3(*LATTICE_VECS[1]),
)

# ------------------------------------------------------------------------------
# Monkhorst–Pack sampling of the Brillouin zone
# ------------------------------------------------------------------------------

# Compute reciprocal lattice vectors (without the 2π factor, since MPB expects that)
b1, b2, _ = mf.find_reciprocal_lattice_vectors(LATTICE_VECS, twopi=False)

# Generate k-point grid (in lattice coordinates by default)
k_points_array = mf.monkhorst_pack_sampling(
    n_points=NK, b1=b1, b2=b2, coordinates="lattice"
)

# Convert k-points into Meep/MPB Vector3 objects
k_points = mf.array_to_Vector3(k_points_array)

# ------------------------------------------------------------------------------
# Build the MPB mode solver
# ------------------------------------------------------------------------------
ms = mpb.ModeSolver(
    num_bands=NUM_BANDS,
    resolution=RESOLUTION,
    default_material=mp.Medium(epsilon=EPS_BG),
    geometry=geometry,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    tolerance=TOLERANCE,
)

# Map all k-points into the first Brillouin zone (important for symmetry)
ms.k_points = [ms.first_brillouin_zone(k) for k in ms.k_points]

# ------------------------------------------------------------------------------
# Dump dielectric tensor for downstream consumers
# (kimble.py, mpb_fields_plot.py, …). MPB writes "<prefix>-epsilon.h5"
# with the full dielectric tensor including subpixel-smoothing components.
# ------------------------------------------------------------------------------
ms.init_params(p=mp.TE, reset_fields=True)
# MPB writes "<filename_prefix>-epsilon.h5", so a prefix ending in the
# resolution gives "<dir>/{RESOLUTION}-epsilon.h5".
ms.filename_prefix = f"{EPSILON_OUTPUT_DIR}{RESOLUTION}"
ms.output_epsilon()
ms.filename_prefix = ""
print(f"Wrote dielectric tensor to {EPSILON_OUTPUT_FILE}")

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
    lattice_vecs=np.asarray(LATTICE_VECS),
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
