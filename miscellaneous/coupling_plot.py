"""
    Docstring for coupling_plot.py

This file is used for plotting the coupling in real space for the already computed
fields at such position. The main purpose is to see where the maximum coupling
strength is located in real space and its overall shape.

NOTE: use the file `mpb_field_computation.py` for obtaining the fields before
executing this file.
"""

import numpy as np
import meep as mp  # needed for the numerical simulation of the photonic crystal
from meep import mpb  # same case as the previous package
import pythtb as tb  # for easy simulation of the tight-binding model

# some imports for plotting utilities
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import src as mf  # our package functions to work with

# %% Some set-ups for the plotting of the figures
mpl.rcParams.update(
    {
        # Figure
        "figure.dpi": 300,  # higher resolution
        # Font
        "font.family": "serif",  # switch to serif (journal-style)
        "font.size": 11,  # base font size
        "mathtext.fontset": "stix",  # match math to Times-like font
        # Axes
        "axes.labelsize": 11,  # axis label font size
        "axes.titlesize": 12,  # title font size
        "axes.linewidth": 1,  # thickness of axes spines
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Tick length/width
        "xtick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.size": 4,
        "ytick.major.width": 1,
        # Grid (off by default, but can be enabled)
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        # Legend
        "legend.fontsize": 10,
        "legend.frameon": False,
    }
)

# %% Define the main constants and parameters for the current simulation
NK = 64  # number of points per direction for the Monkhorst-Pack grid

# angles for the dipole moments of the quantum emitters and its dimensionless value
THETA = np.deg2rad(45)
P_DIMLESS = 1e0  # TODO: modify when you know which value should we put
DETUNING_FREQ = 0.274587  # [c/a]

# Position of the quantum emitter in units of the lattice constant 'a'
# list of positions are also allowed
R0 = [-4.5, 4.5]

BAND_INDEX = 0  # index of the band to consider (0-based)

# ---------------------------------------------------------------------------- #

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

# Define the dipole moments for the emitters
dipole = P_DIMLESS * np.array([np.cos(THETA), np.sin(THETA), 0])

# take the field in the position r0 from the possible list. Here, only one point
# was computed
E_k_r0 = Eks[0, :, :, :]  # Shape: (Nb, Nk, 3)
omega_k = omegas  # Shape: (Nk, Nb)

# ---------------------------------------------------------------------------- #
# Define the real-space grid for interpolation
# ---------------------------------------------------------------------------- #
num_r = 5
xjg = np.arange(-(num_r // 2) + R0[0] // 1, num_r // 2 + 1 + R0[0] // 1, 1)
yjg = np.arange(-(num_r // 2) + R0[1] // 1, num_r // 2 + 1 + R0[1] // 1, 1)
rjpts = mf.meshgrid_points(xjg, yjg, indexing="xy")

# ---------------------------------------------------------------------------- #
# Compute g_r(r0) for each dipole moment, shape (Nr)
# ---------------------------------------------------------------------------- #
gr0rjs = mf.compute_gr0r(E_k_r0, dipole, omega_k, kpts, rjpts)[BAND_INDEX, :]

# ------------------------------------------------------------------------------
# plot the different couplings for the different dipole moments
# ------------------------------------------------------------------------------
fig, ax = plt.subplots()

# normalize the coupling to the maximum
cbar_max = np.abs(gr0rjs).max()

# draw a small arrow representing the dipole
arrow_length = 0.4

sc = ax.scatter(
    rjpts[:, 0],
    rjpts[:, 1],
    c=np.abs(gr0rjs),
    vmin=0,
    vmax=cbar_max,
    marker="s",
    s=0.71 * mf.marker_size(len(rjpts[:, 0])),
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
plt.show()
fig.savefig(f"./figures/miscellaneous/{R0}_coupling_angle_{np.rad2deg(THETA):.0f}.png")
