"""
    Docstring for parameter_sweep.py

This is a script for internal check on how polarization affects the emitter
dynamics. An emitter position is fixed and a sweep for different polarizations is
perform.
"""

# %% Import necessary packages
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
THETAS = np.linspace(90, 140, 10) * np.pi / 180
P_DIMLESS = 1e-2  # TODO: modify when you know which value should we put
DETUNING_FREQ = 0.274587  # [c/a]

# Position of the quantum emitter in units of the lattice constant 'a'
# list of positions are also allowed
N_EMITTERS = 1
R0 = [0.25, 0.5]

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
dipoles = [P_DIMLESS * np.array([np.cos(theta), np.sin(theta), 0]) for theta in THETAS]

# take the field in the position r0 from the possible list. Here, only one point
# was computed
E_k_r0 = Eks[0, :, :, :]  # Shape: (Nb, Nk, 3)
omega_k = omegas  # Shape: (Nk, Nb)

# ---------------------------------------------------------------------------- #
# Define the real-space grid for interpolation
# ---------------------------------------------------------------------------- #
xg = np.linspace(-5, 5, 101)
yg = np.linspace(-5, 5, 101)
rpts = mf.meshgrid_points(xg, yg)  # Shape: (Nx*Ny, 2)

# ---------------------------------------------------------------------------- #
# Compute g_r(r0) for each dipole moment, shape (len(THETAS), Nr)
# ---------------------------------------------------------------------------- #
gr0rs = [
    mf.compute_gr0r(E_k_r0, dipole, omega_k, kpts, rpts)[BAND_INDEX, :]
    for dipole in dipoles
]

# ------------------------------------------------------------------------------
# plot the different couplings for the different dipole moments
# ------------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(THETAS))

vmax = np.abs(gr0rs).max()

for i, theta in enumerate(THETAS):
    sc = axs[i].scatter(
        rpts[:, 0],
        rpts[:, 1],
        c=np.abs(gr0rs[i]),
        s=mf.marker_size(len(rpts)),
        marker="o",
        cmap="inferno",
        vmin=0,
        vmax=vmax,
    )
    axs[i].set(
        title=rf"$\theta$ = {theta*180/np.pi:.0f}",
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
    )

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{R0}_polarization_sweep_couplings.png")

# ------------------------------------------------------------------------------
# Compute the time evolution of the emitter and the bath
# ------------------------------------------------------------------------------

# -------------------------------
# Normalize parameters
# -------------------------------
alpha_norm = TAU_0 / np.abs(TAU_1)
j1_norm = TAU_1 / np.abs(TAU_1)
j2_norm = TAU_2 / np.abs(TAU_1)
detuning_freq_norm = DETUNING_FREQ / np.sqrt(np.abs(TAU_1))

# -------------------------------
# Initialize tight-binding model
# -------------------------------
infiniteModel = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)

# -------------------------------
# Onsite term
# -------------------------------
infiniteModel.set_onsite([alpha_norm])

# -------------------------------
# First nearest neighbors (1st nn)
# -------------------------------
first_nn_vectors = np.array([[1, 0], [0, 1]])
for vec in first_nn_vectors:
    infiniteModel.set_hop(j1_norm, 0, 0, vec)

# -------------------------------
# Second nearest neighbors (2nd nn)
# -------------------------------
second_nn_vectors = np.array([[1, 1], [1, -1]])
for vec in second_nn_vectors:
    infiniteModel.set_hop(j2_norm, 0, 0, vec)

temporalModel = infiniteModel.cut_piece(
    N_CELLS, fin_dir=0, glue_edgs=False
)  # cutout along 1st direction
finiteModel = temporalModel.cut_piece(
    N_CELLS, fin_dir=1, glue_edgs=False
)  # cutout along 2nd direction

# %% Compute g(r0,rj) on the finite grid

if N_CELLS % 2 == 0:
    xjg = np.arange(-N_CELLS / 2, N_CELLS / 2, 1)
    yjg = np.arange(-N_CELLS / 2, N_CELLS / 2 + 1, 1)
else:
    xjg = np.arange(-(N_CELLS // 2), N_CELLS // 2 + 1, 1)
    yjg = np.arange(-(N_CELLS // 2), N_CELLS // 2 + 1, 1)
rjpts = mf.meshgrid_points(xjg, yjg, indexing="xy")

gr0rjs = [
    mf.compute_gr0r(
        E_k_r0=E_k_r0,
        dipole_moment=dipole,
        omega_k=omega_k,
        kpts=kpts,
        rpts=rjpts,
    )[BAND_INDEX, :]
    for dipole in dipoles
]

# we want to normalize the couplings by the 1-st nn coupling strength, since TAU_1 has units
# of frequency² while g_rj(r0) has units of frequency, we need to divide by sqrt(|TAU_1|)
gr0rjs_norm = [gr0rj / np.sqrt(np.abs(TAU_1)) for gr0rj in gr0rjs]

# %% Build the light-matter Hamiltonian

indexes = np.arange(0, len(gr0rjs_norm[0]), 1)
couplings = [[[list(gr0rj_norm), list(indexes)]] for gr0rj_norm in gr0rjs_norm]

# build the light matter Hamiltonian
hamiltonians = [
    mf.make_light_matter_Hamiltonian(finiteModel, detuning_freq_norm, coupling)[0]
    for coupling in couplings
]

# %% Compute time evolution

from scipy.sparse.linalg import expm_multiply

tvals = np.linspace(T0, TF, T_STEPS)

psi0 = np.zeros(hamiltonians[0].shape[0])
psi0[0] = 1

# Compute time evolution
psits = [
    expm_multiply(-1j * hamiltonian, psi0, T0, TF, T_STEPS, endpoint=True)
    for hamiltonian in hamiltonians
]

# get emitter component of the overall wavefunction
mod_Ce_squareds = [np.abs(psit[:, 0]) ** 2 for psit in psits]

# get positions in Cartesian coordinates (requires find_positions() defined above)
positions = mf.find_positions(finiteModel._orb, finiteModel._lat)

# get photonic component of the overall wave function at selected time, i.e, tvals[time_index]
mod_Cr_squareds = [np.abs(psit[TIME_INDEX, N_EMITTERS:]) ** 2 for psit in psits]

x = positions[:, 0]
y = positions[:, 1]

# ------------------------------------------------------------------------------
# plot the time evolutions for the different polarizations
# ------------------------------------------------------------------------------
fig, ax = plt.subplots()

for i, theta in enumerate(THETAS):
    ax.plot(tvals, mod_Ce_squareds[i], label=rf"$\theta = {theta*180/np.pi:.0f}^\circ$")

ax.legend()
ax.set(xlabel=r"$t [\tau_1]$", ylabel=r"$|C_e(t)|^2$", xscale="log", yscale="log")

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{R0}_polarization_sweep_emit_population.png")

fig, axs = plt.subplots(1, len(THETAS))

vmax = np.abs(mod_Cr_squareds).max()

for i, theta in enumerate(THETAS):
    axs[i].scatter(
        x,
        y,
        c=mod_Cr_squareds[i],
        s=0.01 * mf.marker_size(len(x)),
        cmap="inferno",
        vmin=0,
        vmax=vmax,
        marker="o",
    )
    axs[i].set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        title=rf"$\theta = {theta*180/np.pi:.0f}^\circ$",
        aspect="equal",
    )

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{R0}_polarization_sweep_bath_population.png")
