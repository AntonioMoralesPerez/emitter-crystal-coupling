"""
    Docstring for kimble.py

This file uses our formalism to compute the coupling between a quantum emitter
and a photonic crystal. We made use of a tight-binding model to simulate the
crystal and assume the emitter as a two-level system.

Additionally, we compute the time evolution of both the emitter and the photonic
bath for different polarizations of the emitter and compare them.

NOTE: this file assumes that the fields are already computed for the points where
the emitters are located. If not, consider running `mpb_field_computation.py`.
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
COMPUTEFIELDS = False  # tells the code to compute the fields using MPB or not
DIRNAME_OUTPUT = "data/kimble/"  # directory to save outputs

TOLERANCE = 1e-12  # tolerance for the MPB calculations

N_CELLS = 51  # number of unit cells in each direction

NK = 64  # number of points per direction for the Monkhorst-Pack grid

# angles for the dipole moments of the quantum emitters and its dimensionless value
THETA_1 = (90) * np.pi / 180
THETA_2 = (135) * np.pi / 180
P_DIMLESS = (
    1e-2  # TODO: not the value specified in the article. Consider a possible change
)
N_EMITTERS = 1
DETUNING_FREQ = 0.274587  # [c/a]

# Position of the quantum emitter in units of the lattice constant 'a'
# list of positions are also allowed
R0 = [0.0, 0.0]

# small inputs for the tight-binding model
LAT = [[1, 0], [0, 1]]
ORB = [[0, 0]]

BAND_INDEX = 0  # index of the band to consider (0-based)

# coefficients for the tight-binding model
TAU_0 = 0.07755639
TAU_1 = -0.01993698
TAU_2 = 0.00053966

# path in the Brillouin zone for band structure calculation
K_PATH = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
# define HSPs labels
K_LABEL = [r"$\Gamma$", r"X", r"M", r"$\Gamma$"]

T0 = 0  # initial time for the field snapshots
TF = 80  # final time for the field snapshots
T_STEPS = 100  # number of time steps for the field snapshots
TIME_INDEX = 20  # time index to consider for the field snapshots

# ---------------------------------------------------------------------------- #

data = mf.load(f"./data/kimble/fields/{NK}_meshgrid_at_{R0}.pkl")

# Extract relevant quantities
omegas = np.asarray(data["freqs"])  # Frequencies (Nk, Nb)
Eks = np.asarray(data["Eks"])  # Electric fields, shape: (Nr, Nb, Nk, 3)
kpts = np.asarray(data["kpts"])  # k-points in cartesian coordinates, shape: (Nk, 3)

# Define the dipole moments for the emitters
thetas = [THETA_1, THETA_2]
dipoles = [P_DIMLESS * np.array([np.cos(theta), np.sin(theta), 0]) for theta in thetas]

# ---------------------------------------------------------------------------- #
# Select the band of interest and position r0
# ---------------------------------------------------------------------------- #

E_k_r0 = Eks[0, :, :, :]  # Shape: (Nb, Nk, 3)

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

# %% Finite model definition

# Cut the system
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
        omega_k=omegas,
        kpts=kpts,
        rpts=rjpts,
    )
    for dipole in dipoles
]

# we want to normalize the couplings by the 1-st nn coupling strength, since TAU_1 has units
# of frequency² while g_rj(r0) has units of frequency, we need to divide by sqrt(|TAU_1|)
gr0rjs_norm = [gr0rj / np.sqrt(np.abs(TAU_1)) for gr0rj in gr0rjs]

# %% Build the light-matter Hamiltonian

# non-local couplings
values = [gr0rj_norm[BAND_INDEX, :] for gr0rj_norm in gr0rjs_norm]

# small check before proceeding
assert len(values[0]) == len(
    values[1]
), "Unexpected different lengths for the couplings!"

indexes = np.arange(0, len(values[0]), 1)
couplings_1 = [[list(values[0]), list(indexes)]]
couplings_2 = [[list(values[1]), list(indexes)]]

# build the light-matter Hamiltonian
H_total_1, omegas_1 = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq_norm, couplings_1
)
H_total_2, omegas_2 = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq_norm, couplings_2
)

# %% Compute time evolution

from scipy.sparse.linalg import expm_multiply

tvals = np.linspace(T0, TF, T_STEPS, endpoint=True)

# Define initial condition
assert (
    H_total_1.shape == H_total_2.shape
), "Unexpected different shapes for the Hamiltonians!"

psi0 = np.zeros(H_total_1.shape[0])
psi0[0] = 1

# Compute time evolution
psit_1 = expm_multiply(-1j * H_total_1, psi0, T0, TF, T_STEPS, endpoint=True)
psit_2 = expm_multiply(-1j * H_total_2, psi0, T0, TF, T_STEPS, endpoint=True)

# ------------------------------------------------------------------------------
# place another emitter at [-5,5] to see if it is exited when θ = 90º and
# not when θ = 130º. The angle of the new emitter is selected to be θ = 45°, since
# it should experience the most coupling
# ------------------------------------------------------------------------------
new_tf = 50
new_tvals = np.linspace(T0, new_tf, T_STEPS, endpoint=True)
new_R = [-4.5, 4.5]
new_theta = np.deg2rad(45)
new_dipole = 1e-1 * np.array([np.cos(new_theta), np.sin(new_theta), 0])

# we also need to provide a detuning frequency, we are going to pick the one
# associated to a state localized near the position of the emitter
target = [-5, 5]
idx = np.where(np.all(np.isclose(rjpts, target), axis=1))[0][0]
new_detuning_1 = omegas_1[idx]
new_detuning_2 = omegas_2[idx]

add_data = mf.load(f"./data/kimble/fields/{NK}_meshgrid_at_{new_R}.pkl")
new_omegas = np.asarray(add_data["freqs"])  # (Nk, Nb)
E_k_R = np.asarray(add_data["Eks"])[0, :, :, :]  # (Nb, Nk, 3)
new_kpts = np.asarray(add_data["kpts"])

gRrjs_norm = mf.compute_gr0r(E_k_R, new_dipole, new_omegas, new_kpts, rjpts) / np.sqrt(
    np.abs(TAU_1)
)  # (Nb, Nr)

new_value = gRrjs_norm[BAND_INDEX, :]

new_couplings_1 = [[list(values[0]), list(indexes)], [list(new_value), list(indexes)]]
new_couplings_2 = [[list(values[1]), list(indexes)], [list(new_value), list(indexes)]]

# build the light-matter Hamiltonian
new_H_1, new_omegas_1 = mf.make_light_matter_Hamiltonian(
    finiteModel, [detuning_freq_norm, new_detuning_1], new_couplings_1
)
new_H_2, new_omegas_2 = mf.make_light_matter_Hamiltonian(
    finiteModel, [detuning_freq_norm, new_detuning_2], new_couplings_2
)

new_psi0 = np.zeros(new_H_1.shape[0])
new_psi0[0] = 1

# Compute time evolution
new_psit_1 = expm_multiply(-1j * new_H_1, new_psi0, T0, new_tf, T_STEPS, endpoint=True)
new_psit_2 = expm_multiply(-1j * new_H_2, new_psi0, T0, new_tf, T_STEPS, endpoint=True)

# ------------------------------------------------------------------------------
# Prepare data to be plotted
# ------------------------------------------------------------------------------

# get emitter component of the overall wavefunction
mod_Ce_squareds = [np.abs(psit_1[:, 0]) ** 2, np.abs(psit_2[:, 0]) ** 2]

# get photonic component of the overall wave function at selected time, i.e, tvals[time_index]
mod_Cr_squareds = [
    np.abs(psit_1[TIME_INDEX, N_EMITTERS:]) ** 2,
    np.abs(psit_2[TIME_INDEX, N_EMITTERS:]) ** 2,
]

zs = [mod_Cr_squared.reshape(N_CELLS, N_CELLS) for mod_Cr_squared in mod_Cr_squareds]

# get all the emitters dynamics
new_mod_Ces = [np.abs(new_psit_1[:, 0]) ** 2, np.abs(new_psit_2[:, 0]) ** 2]
new_mod_Ce_probes = [np.abs(new_psit_1[:, 1]) ** 2, np.abs(new_psit_2[:, 1]) ** 2]

# compute the concurrence
concurrences = [
    2 * np.abs(new_mod_Ces[0] * np.conj(new_mod_Ce_probes[0])),
    2 * np.abs(new_mod_Ces[1] * np.conj(new_mod_Ce_probes[1])),
]

# ------------------------------------------------------------------------------
# let us plot the time evolution of the emitter for both polarizations
# ------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2)

dip_colors = ["#1f77b4", "#ff7f0e"]  # [dipole 1, dipole 2]

for i, theta in enumerate(thetas):
    ax = axs[i]
    ax.plot(tvals, mod_Ce_squareds[i], c="k")
    ax.set(
        xlabel=r"$t [\tau_1^{-1/2}]$",
        ylabel=r"$|C_e(t)|^2$",
        title=rf"$\theta = {np.rad2deg(theta):.0f}°$",
        yscale="log",
    )
    ax.title.set_color(dip_colors[i])

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/different_polarizations_decays_at_{R0}.png")

# ------------------------------------------------------------------------------
# let us plot the time evolution of the bath with one emitter coupled to it
# ------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2)

arrow_length = 1

cbar_max = max(zs[0].max(), zs[1].max())

for i, theta in enumerate(thetas):
    ax = axs[i]
    sc = ax.pcolormesh(xjg, yjg, zs[i], vmin=0, vmax=cbar_max)
    ax.set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        title=rf"$\theta = {np.rad2deg(theta):.0f}º$ for $t/\tau_1^{{1/2}} = {tvals[TIME_INDEX]:.0f}$",
        aspect="equal",
        xlim=(-9, -1),
        ylim=(1, 9),
    )
    ax.title.set_color(dip_colors[i])
    cbar = fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{r}|^2$", shrink=0.8)
    cbar.formatter.set_powerlimits((0, 0))
    mf.draw_arrow(ax, new_R, new_theta, length=arrow_length)

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{R0}_bath_population.png")

# ------------------------------------------------------------------------------
# let us plot the coupling strength of the new emitter to make sure it will couple
# to the bath mode the first emitter is producing
# ------------------------------------------------------------------------------
fig, ax = plt.subplots()

# normalize the coupling to the maximum
cbar_max = np.abs(new_value).max()

sc = ax.scatter(
    rjpts[:, 0],
    rjpts[:, 1],
    c=np.abs(new_value),
    vmin=0,
    vmax=cbar_max,
    marker="s",
    s=5e1 * mf.marker_size(len(rjpts[:, 0])),
)
mf.draw_arrow(ax, new_R, new_theta, length=arrow_length)
ax.set(
    title=rf"$\theta = {np.rad2deg(new_theta):.0f}º$",
    xlabel=r"$x/a$",
    ylabel=r"$y/a$",
    aspect="equal",
    xlim=(-9, -1),
    ylim=(1, 9),
)
cbar = fig.colorbar(sc, ax=ax, label=r"$|g_\mathbf{r}(\mathbf{R})|$")
cbar.formatter.set_powerlimits((0, 0))

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{new_R}_coupling_check.png")

# ------------------------------------------------------------------------------
# let us plot the time evolution of the emitters when two emitters are present
# ------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2)

for i, theta in enumerate(thetas):
    ax = axs[i]
    ax.plot(new_tvals, new_mod_Ces[i], c="k", label="Excited")
    ax.plot(new_tvals, new_mod_Ce_probes[i], c="b", ls="--", label="Probe")
    ax.axvline(tvals[TIME_INDEX], linewidth=0.5, color="k", ls="--")
    ax.set(
        title=rf"$\theta = {np.rad2deg(theta):.0f}°$",
        xlabel=r"$t [\tau_1^{-1/2}]$",
        ylabel=r"$|C_e(t)|^2$",
        yscale="log",
    )
    ax.title.set_color(dip_colors[i])
    ax.legend()

# Layout adjustment and save
fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/emit_pop_at_{new_R}_exciting_at_{R0}.png")
