"""
    Docstring for coupling_plot.py

This file is used for plotting the coupling in real space for the already computed
fields at such position. The main purpose is to see where the maximum coupling
strength is located in real space and its overall shape.

NOTE: use the file `mpb_field_computation.py` for obtaining the fields before
executing this file.
"""

import numpy as np

# some imports for plotting utilities
import matplotlib.pyplot as plt

import src as mf  # our package functions to work with

# %% Define the main constants and parameters for the current simulation
NK = 64  # number of points per direction for the Monkhorst-Pack grid

# angles for the dipole moments of the quantum emitters and its dimensionless value
THETA = np.deg2rad(90)
LAT_PARAM = 300e-9  # lattice parameter [m]
DIP_STRENGTH = 50  # [D]
P_DIMLESS = DIP_STRENGTH / mf.unit_dip(a=LAT_PARAM)

# Position of the quantum emitter in units of the lattice constant 'a'
R0 = [0.0, 0.0]

BAND_INDEX = 0  # index of the band to consider (0-based)

# ---------------------------------------------------------------------------- #
# Compute g_r(r0)
# ---------------------------------------------------------------------------- #

# Load the fields and k-points from the MPB calculation
# This file was generated previously
data = mf.load(f"./data/kimble/fields/{NK}_meshgrid_at_{R0}.pkl")

# Extract relevant quantities
omegas = np.asarray(data["freqs"])  # Frequencies (Nk, Nb)
Eks = np.asarray(data["Eks"])  # Electric fields, shape: (Nr, Nb, Nk, 3)
kpts = np.asarray(data["kpts"])  # k-points in cartesian coordinates, shape: (Nk, 3)

# Define the dipole moment for the emitter
dipole = P_DIMLESS * np.array([np.cos(THETA), np.sin(THETA), 0])

# take the field at position r0 from the possible list. Here, only one point
# was computed
E_k_r0 = Eks[0]  # Shape: (Nb, Nk, 3)

# ---------------------------------------------------------------------------- #
# Define the real-space grid for interpolation
# ---------------------------------------------------------------------------- #
num_r = 10
xjg = np.arange(
    -(num_r // 2) + int(np.floor(R0[0])), num_r // 2 + 1 + int(np.floor(R0[0])), 1
)
yjg = np.arange(
    -(num_r // 2) + int(np.floor(R0[1])), num_r // 2 + 1 + int(np.floor(R0[1])), 1
)
rjpts = mf.meshgrid_points(xjg, yjg, indexing="xy")

# ---------------------------------------------------------------------------- #
# Compute g_r(r0), shape (Nr)
# ---------------------------------------------------------------------------- #
gr0rjs = mf.compute_real_space_coupling(E_k_r0, dipole, omegas, kpts, rjpts)[
    BAND_INDEX, :
]

# ------------------------------------------------------------------------------
# plot the coupling
# ------------------------------------------------------------------------------
fig, ax = plt.subplots()

# normalize the coupling to the maximum
cbar_max = np.abs(gr0rjs).max()

# draw a small arrow representing the dipole
arrow_length = 2

sc = ax.scatter(
    rjpts[:, 0],
    rjpts[:, 1],
    c=np.abs(gr0rjs),
    vmin=0,
    vmax=cbar_max,
    marker="s",
    s=0.71 * mf.marker_size(len(rjpts)),
)
mf.draw_arrow(ax, R0, THETA, length=arrow_length)
ax.set(
    title=rf"$\theta = {np.rad2deg(THETA):.0f}º$",
    xlabel=r"$x/a$",
    ylabel=r"$y/a$",
    aspect="equal",
)
cbar = fig.colorbar(sc, ax=ax, label=r"$|g_\mathbf{r}(\mathbf{r}_0)|$")
cbar.formatter.set_powerlimits((0, 0))

fig.tight_layout()
fig.savefig(f"./figures/miscellaneous/{R0}_coupling_angle_{np.rad2deg(THETA):.0f}.png")
plt.show()
