"""
    Docstring for position_sweep.py

This is a script for internal check on how the emitter position affects the
emitter dynamics. A dipole polarization is fixed and a sweep over different
emitter positions is performed.
"""

# %% Import necessary packages
import numpy as np
import pythtb as tb  # for easy simulation of the tight-binding model

# some imports for plotting utilities
import matplotlib.pyplot as plt

import emitter_crystal_coupling as mf  # our package functions to work with

# %% Define the main constants and parameters for the current simulation
NUM_BANDS = 2

EPSILON_INPUT_FILE = "data/kimble/epsilon.h5"
DIRNAME_OUTPUT = "data/miscellaneous/"  # directory to save outputs

TOLERANCE = 1e-12  # tolerance for the MPB calculations

N_CELLS = 41  # number of unit cells in each direction

NK = 64  # number of points per direction for the Monkhorst-Pack grid

# angles for the dipole moments of the quantum emitters and its dimensionless value
THETA = 130 * np.pi / 180
P_DIMLESS = 1e-2  # TODO: modify when you know which value should we put
DETUNING_FREQ = 0.274587  # [c/a]

# Position of the quantum emitter in units of the lattice constant 'a'
# list of positions are also allowed
N_EMITTERS = 1
R0_LIST = [[0, 0], [0.25, 0], [0.5, 0], [0.25, 0.25], [0.5, 0.5]]

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
datas = [mf.load(f"./data/kimble/fields/{NK}_meshgrid_at_{R0}.pkl") for R0 in R0_LIST]

# Extract relevant quantities
omegas = [np.asarray(data["freqs"]) for data in datas]
Eks = [np.asarray(data["Eks"]) for data in datas]
kpts = [np.asarray(data["kpts"]) for data in datas]

# Define the dipole moments for the emitters
dipole = P_DIMLESS * np.array([np.cos(THETA), np.sin(THETA), 0])

# take the field in the position r0 from the possible list. Here, only one point
# was computed
E_k_r0s = [Ek[0, :, :, :] for Ek in Eks]

# ---------------------------------------------------------------------------- #
# Define the real-space grid for interpolation
# ---------------------------------------------------------------------------- #
xg = np.linspace(-5, 5, 101)
yg = np.linspace(-5, 5, 101)
rpts = mf.meshgrid_points(xg, yg)

# ---------------------------------------------------------------------------- #
# Compute g_r(r0) for each dipole moment, shape (len(THETAS), Nr)
# ---------------------------------------------------------------------------- #
gr0rs = [
    mf.compute_real_space_coupling(E_k_r0s[i], dipole, omegas[i], kpts[i], rpts)[
        BAND_INDEX, :
    ]
    for i in range(len(R0_LIST))
]

# ------------------------------------------------------------------------------
# plot the different couplings for the different dipole moments
# ------------------------------------------------------------------------------
fig, axs = plt.subplots(1, len(R0_LIST))

vmax = np.abs(gr0rs).max()

for i, r0 in enumerate(R0_LIST):
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
        title=rf"$R_0 = {r0}$",
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
    )

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{THETA}_position_sweep_couplings.png")

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

xjg, yjg, rjpts = mf.build_finite_grid(N_CELLS)

gr0rjs = [
    mf.compute_real_space_coupling(E_k_r0s[i], dipole, omegas[i], kpts[i], rjpts)[
        BAND_INDEX, :
    ]
    for i in range(len(R0_LIST))
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

for i, r0 in enumerate(R0_LIST):
    ax.plot(tvals, mod_Ce_squareds[i], label=rf"$R_0/a = {r0}$")

ax.legend()
ax.set(xlabel=r"$t [\tau_1]$", ylabel=r"$|C_e(t)|^2$", xscale="log", yscale="log")

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{THETA}_position_sweep_emit_population.png")

fig, axs = plt.subplots(1, len(R0_LIST))

vmax = np.abs(mod_Cr_squareds).max()

for i, r0 in enumerate(R0_LIST):
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
        title=rf"$R_0/a = {r0}$",
        aspect="equal",
    )

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{THETA}_position_sweep_bath_population.png")
