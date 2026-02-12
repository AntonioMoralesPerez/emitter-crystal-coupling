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
NUM_BANDS = 2

EPSILON_INPUT_FILE = "data/kimble/epsilon.h5"
DIRNAME_OUTPUT = "data/miscellaneous/"  # directory to save outputs

TOLERANCE = 1e-12  # tolerance for the MPB calculations

N_CELLS = 41  # number of unit cells in each direction

NK = 64  # number of points per direction for the Monkhorst-Pack grid

# angles for the dipole moments of the quantum emitters and its dimensionless value
THETA = (90) * np.pi / 180
P_DIMLESS = 1e-2  # TODO: modify when you know which value should we put
DETUNING_FREQ = 0.274587  # [c/a]

# Position of the quantum emitter in units of the lattice constant 'a'
# list of positions are also allowed
N_EMITTERS = 1
R0 = [-10, 10]

# small inputs for the tight-binding model
LAT = [[1, 0], [0, 1]]
ORB = [[0, 0]]

BAND_INDEX = 0  # index of the band to consider (0-based)

# coefficients for the tight-binding model
TAU_0 = 0.07755639
TAU_1 = -0.01993698
TAU_2 = 0.00053966

T0 = 0  # initial time for the field snapshots
TF = 100  # final time for the field snapshots
T_STEPS = 100  # number of time steps for the field snapshots
TIME_INDEX = 30  # time index to consider for the field snapshots

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
xg = np.linspace(-15, -5, 101)
yg = np.linspace(15, 5, 101)
rpts = mf.meshgrid_points(xg, yg)  # Shape: (Nx*Ny, 2)

# ---------------------------------------------------------------------------- #
# Compute g_r(r0) for each dipole moment, shape (Nr)
# ---------------------------------------------------------------------------- #
gr0rs = mf.compute_gr0r(E_k_r0, dipole, omega_k, kpts, rpts)[BAND_INDEX, :]

# ------------------------------------------------------------------------------
# plot the different couplings for the different dipole moments
# ------------------------------------------------------------------------------
fig, ax = plt.subplots()

vmax = np.abs(gr0rs).max()

sc = ax.scatter(
    rpts[:, 0],
    rpts[:, 1],
    c=np.abs(gr0rs),
    s=mf.marker_size(len(rpts)),
    marker="o",
    cmap="inferno",
    vmin=0,
    vmax=vmax,
)
ax.set(
    title=rf"$\theta$ = {THETA*180/np.pi:.0f}",
    xlabel=r"$x/a$",
    ylabel=r"$y/a$",
    aspect="equal",
)

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{R0}_coupling.png")
