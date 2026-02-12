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
TIME_INDEX = 50  # time index to consider for the field snapshots

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

# ---------------------------------------------------------------------------- #

infiniteModel = tb.tb_model(2, 2, LAT, ORB)

# onsite term
infiniteModel.set_onsite([TAU_0])

# 1st nn
infiniteModel.set_hop(TAU_1, 0, 0, [1, 0])
infiniteModel.set_hop(TAU_1, 0, 0, [0, 1])

# 2nd nn
infiniteModel.set_hop(TAU_2, 0, 0, [1, 1])
infiniteModel.set_hop(TAU_2, 0, 0, [1, -1])

# %% Compute bands for the infinite model

### Compute bands along high-symmetry path

# creates a path between the HSPs defined before with 500 points between them
k_vec, k_dist, k_node = infiniteModel.k_path(K_PATH, 500, report=False)

# solve the infinite model in the considered path
evals = infiniteModel.solve_all(k_vec)

## Select the detuning frequency
# we are picking the value of the frequency at the X point where there is the
# saddle point of the band and a high density of states
detuning_freq = np.sqrt(evals[BAND_INDEX][k_node[1] == k_dist].item())
print(f"Detuning frequency selected at X point: {detuning_freq:.6f} (2πc/a)")

# ---------------------------------------------------------------------------- #
# compute the bands with MPB for comparison

geometry_lattice = mp.Lattice(
    size=lattice_size, basis1=lattice_vecs[0], basis2=lattice_vecs[1]
)

ms = mpb.ModeSolver(
    num_bands=NUM_BANDS,
    resolution=mp.Vector3(*resolution),  # type: ignore[arg-type]
    epsilon_input_file=EPSILON_INPUT_FILE,
    tolerance=TOLERANCE,
    mesh_size=1,
    geometry_lattice=geometry_lattice,
    k_points=mf.array_to_Vector3(k_vec),
)

ms.run_te()

# all frequency bands
freqs = ms.all_freqs

# %% Read epsilon data for plotting
md = mpb.MPBData(rectify=False, periods=2)
d = mf.convert_epsilon("./data/kimble/epsilon.h5", md)

x1 = d["positions-lattice"][:, 0]
y1 = d["positions-lattice"][:, 1]
z1 = d["epsilon_values"]

# ---------------------------------------------------------------------------- #
# PERFORM THE FOURIER TRANSFORMATION OF THE COUPLING FUNCTION g(r0,r)
# ---------------------------------------------------------------------------- #

# Load the fields and k-points from the MPB calculation
# This file was generated previously in `kimble.ipynb`
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

# ---------------------------------------------------------------------------- #
# Define the real-space grid, we are going to take some lattice sites
# ---------------------------------------------------------------------------- #
num_r = 13
xjg = np.arange(-(num_r // 2), num_r // 2 + 1, 1)
yjg = np.arange(-(num_r // 2), num_r // 2 + 1, 1)
rjpts = mf.meshgrid_points(xjg, yjg, indexing="xy")

# ---------------------------------------------------------------------------- #
# Compute g_r(r0) for each dipole moment, shape (Nr,)
# ---------------------------------------------------------------------------- #
gr0rj_matrices = [
    mf.compute_gr0r(
        E_k_r0=E_k_r0,
        dipole_moment=dipole,
        omega_k=omegas,
        kpts=kpts,
        rpts=rjpts,
    )[BAND_INDEX, :].reshape(num_r, num_r)
    for dipole in dipoles
]

# ---------------------------------------------------------------------------- #
fig, axs = plt.subplots(2, 2)  # two-column width
panel_labels = ["(a)", "(b)", "(c)", "(d)"]

# ---------------------------------------------------------------------------- #
# Global dipole settings
# ---------------------------------------------------------------------------- #
x0, y0 = R0
dip_colors = ["#1f77b4", "#ff7f0e"]  # [dipole 1, dipole 2]
arrow_length = 0.4

# ==============================================================================
# --- First quadrant: dielectric structure (a)
# ==============================================================================
ax = axs[0, 0]
sc1 = ax.scatter(x1, y1, c=z1, marker="o", cmap="copper_r", s=mf.marker_size(len(x1)))
fig.colorbar(sc1, ax=ax, label=r"$\epsilon(\mathbf{r})$")
ax.set_title(r"$\epsilon(\mathbf{r})$")
ax.set_xlabel(r"$x/a$")
ax.set_ylabel(r"$y/a$")
ax.set_aspect("equal")
ax.text(-0.2, 1.1, panel_labels[0], transform=ax.transAxes)

for i, theta in enumerate(thetas):
    mf.draw_arrow(ax, R0, theta, dip_colors[i], rf"$\theta_{i+1}$", length=arrow_length)

# ==============================================================================
# --- Second quadrant: TE bands (b)
# ==============================================================================
ax = axs[0, 1]
for n in evals:
    # Ensure no negative values due to numerical errors
    n_safe = np.clip(np.real(n), 0, None)
    ax.plot(k_dist, np.sqrt(n_safe), color="r", linestyle="--")
ax.plot(k_dist, freqs, color="k")
ax.set_xlim(k_node[0], k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(K_LABEL)
for k in k_node:
    ax.axvline(x=k, linewidth=0.5, color="k")
ax.set_ylabel(r"$\omega \, [c/a]$")
ax.set_title("TE modes")
ax.text(-0.2, 1.05, panel_labels[1], transform=ax.transAxes)

# ==============================================================================
# --- Third and fourth quadrant: g_r(r0) for both dipoles (c-d)
# ==============================================================================
# define a maximum for the color scale
vmax = np.abs(gr0rjs).max()

for i, theta in enumerate(thetas):
    ax = axs[1, i]

    sc2 = ax.pcolormesh(
        xjg,
        yjg,
        np.abs(gr0rj_matrices[i]),  # take absolute value for plotting
        cmap="inferno",
        vmin=0,
        vmax=vmax,
    )
    cbar1 = fig.colorbar(sc2, ax=ax, label=r"$|g_\mathbf{r}(\mathbf{r}_0)| \, [c/a]$")
    cbar1.formatter.set_powerlimits((0, 0))

    ax.set(
        title=rf"$\theta_{i+1} = {theta*180/np.pi:.0f}º$",
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
    )
    ax.title.set_color(dip_colors[i])
    ax.text(-0.2, 1.1, panel_labels[2], transform=ax.transAxes)

# --- Layout and save ---
fig.tight_layout()
plt.show()
fig.savefig(f"./figures/figure2.png")

# ------------------------------------------------------------------------------
# Compute the LDOS to see how the emitters couple to the photonic bath
# ------------------------------------------------------------------------------
omega_k = omegas[:, BAND_INDEX]  # Shape: (Nk,)

# Frequency range for LDOS
omega_eval = np.linspace(np.min(omega_k), np.max(omega_k), 500)

# Fields for the selected band
E_k_r0_band = E_k_r0[BAND_INDEX, :, :]  # Shape: (Nk, 3)

# -----------------------------
# Compute LDOS for each dipole
# -----------------------------
ldos_list = [
    mf.compute_ldos(
        omega_eval=omega_eval, omega_k=omega_k, E_k_r0=E_k_r0_band, p_vec=dip
    )
    for dip in dipoles
]

# -----------------------------
# Plot LDOS side by side
# -----------------------------
fig, ax = plt.subplots()

rel_difference = np.abs(ldos_list[0] - ldos_list[1]) / np.abs(np.max(ldos_list[0]))

ax.scatter(omega_eval, rel_difference)
ax.set_xlabel(r"$\omega \, [c/a]$")
ax.set_ylabel(f"Relative difference [1]")
ax.set_title(
    rf"$R_0/a$={R0}, $\theta_1$={np.degrees(THETA_1):.0f}º, $\theta_2$={np.degrees(THETA_2):.0f}º"
)
ax.grid(True)
ax.axvline(
    x=detuning_freq,
    color="k",
    linestyle="--",
    label="Detuning \n frequency",
    linewidth=1,
)
ax.legend()

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/ldos_difference.png")

# ------------------------------------------------------------------------------
# Compute the time evolution of the emitter and the bath
# ------------------------------------------------------------------------------

# -------------------------------
# Normalize parameters
# -------------------------------
alpha_norm = TAU_0 / np.abs(TAU_1)
j1_norm = TAU_1 / np.abs(TAU_1)
j2_norm = TAU_2 / np.abs(TAU_1)
detuning_freq_norm = detuning_freq / np.sqrt(np.abs(TAU_1))

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

# -------------------------------------------------
# --- Generate 2D k-point grid in the Brillouin zone ---
# -------------------------------------------------
kx = np.linspace(-0.5, 0.5, NK)
ky = np.linspace(-0.5, 0.5, NK)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

# Flatten grid for evaluation
k_points = np.column_stack((kx_grid.ravel(), ky_grid.ravel()))

# -------------------------------------------------
# --- Compute band energies for the selected band ---
# -------------------------------------------------
band_energies = np.array(
    [infiniteModel.solve_one(k)[BAND_INDEX] for k in k_points],  # type: ignore[index]
    dtype=float,
)

# -------------------------------------------------
# --- Clip all negative values (remove numerical noise) ---
# -------------------------------------------------
band_energies_clipped = np.clip(band_energies, a_min=0.0, a_max=None)

# -------------------------------------------------
# --- Reshape and compute omega = sqrt(E) ---
# -------------------------------------------------
omega = np.sqrt(band_energies_clipped).reshape(NK, NK)

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
# of frequency² while g(r0,rj) has units of frequency, we need to divide by sqrt(|TAU_1|)
gr0rjs_norm = [gr0rj / np.sqrt(np.abs(TAU_1)) for gr0rj in gr0rjs]

# %% Build the light-matter Hamiltonian

# ... non-local couplings
values = [gr0rj_norm[BAND_INDEX, :] for gr0rj_norm in gr0rjs_norm]

# small check before proceeding
assert len(values[0]) == len(
    values[1]
), "Unexpected different lengths for the couplings!"

indexes = np.arange(0, len(values[0]), 1)
couplings_1 = [[list(values[0]), list(indexes)]]
couplings_2 = [[list(values[1]), list(indexes)]]

# build the light-matter Hamiltonian
H_total_1, _ = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq_norm, couplings_1
)
H_total_2, _ = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq_norm, couplings_2
)

# %% Compute time evolution

from scipy.sparse.linalg import expm_multiply

tvals = np.linspace(T0, TF, T_STEPS)

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
# place another emitter at [-0.5,0.5] to see if it is exited when θ = 90º and
# not when θ = 130º
# ------------------------------------------------------------------------------
new_R = [-10, 10]
new_dipole = P_DIMLESS * np.array([0, 1, 0])

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
new_H_1, _ = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq_norm, new_couplings_1
)
new_H_2, _ = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq_norm, new_couplings_2
)

new_psi0 = np.zeros(new_H_1.shape[0])
new_psi0[0] = 1

# Compute time evolution
new_psit_1 = expm_multiply(-1j * new_H_1, new_psi0, T0, TF, T_STEPS, endpoint=True)
new_psit_2 = expm_multiply(-1j * new_H_2, new_psi0, T0, TF, T_STEPS, endpoint=True)

# ------------------------------------------------------------------------------
# Prepare data to be plotted
# ------------------------------------------------------------------------------

# get emitter component of the overall wavefunction
mod_Ce_squared_1 = np.abs(psit_1[:, 0]) ** 2
mod_Ce_squared_2 = np.abs(psit_2[:, 0]) ** 2

# compute the entanglement for both cases
entropy_1 = mf.binary_entropy(mod_Ce_squared_1)
entropy_2 = mf.binary_entropy(mod_Ce_squared_2)

# get positions in Cartesian coordinates (requires find_positions() defined above)
positions = mf.find_positions(finiteModel._orb, finiteModel._lat)

# get photonic component of the overall wave function at selected time, i.e, tvals[time_index]
mod_Cr_squared_1 = np.abs(psit_1[TIME_INDEX, N_EMITTERS:]) ** 2
mod_Cr_squared_2 = np.abs(psit_2[TIME_INDEX, N_EMITTERS:]) ** 2

x = positions[:, 0] - N_CELLS // 2  # shift the crystal to place emitter at the center
y = positions[:, 1] - N_CELLS // 2
z1 = mod_Cr_squared_1
z2 = mod_Cr_squared_2

# get all the emitters dynamics
new_mod_Ce_1 = np.abs(new_psit_1[:, 0]) ** 2
new_mod_Ce_2 = np.abs(new_psit_2[:, 0]) ** 2
new_mod_Ce_probe_1 = np.abs(new_psit_1[:, 1]) ** 2
new_mod_Ce_probe_2 = np.abs(new_psit_2[:, 1]) ** 2

# compute the concurrence
concurrence_1 = 2 * np.abs(new_mod_Ce_1 * np.conj(new_mod_Ce_probe_1))
concurrence_2 = 2 * np.abs(new_mod_Ce_2 * np.conj(new_mod_Ce_probe_2))

# ------------------------------------------------------------------------------

fig, axs = plt.subplots(3, 2)  # no figsize unless absolutely needed
panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# ------------------------------------------------------------------------------
# First quadrant (a)
# ------------------------------------------------------------------------------
ax = axs[0, 0]
c = ax.contourf(kx_grid, ky_grid, omega, levels=12, cmap="viridis")
fig.colorbar(c, ax=ax, label=r"$\omega_\mathbf{k}\,[\tau_1^{-1/2}]$", pad=0.07)

ax.contour(
    kx_grid, ky_grid, omega, levels=[detuning_freq_norm], colors="k", linewidths=2.0
)

ax.set(
    xlabel=r"$k_x$",
    ylabel=r"$k_y$",
    title=rf"$\omega_0/\tau_1^{-1/2} = {detuning_freq:.2f}$",
)
ax.set_aspect("equal")
# ax.text(-0.2, 1.1, panel_labels[0], transform=ax.transAxes)

# ------------------------------------------------------------------------------
# second quadrant (b)
# ------------------------------------------------------------------------------

ax1 = axs[0, 1]

(l1,) = ax1.plot(
    tvals,
    mod_Ce_squared_1,
    lw=3,
    color=dip_colors[0],
    label=r"$|C_e(t)|^2,\;\theta=90^\circ$",
)
(l2,) = ax1.plot(
    tvals,
    mod_Ce_squared_2,
    lw=2,
    color=dip_colors[1],
    alpha=0.9,
    label=r"$|C_e(t)|^2,\;\theta=135^\circ$",
)

ax1.set(
    xlabel=r"$t\,[\tau_1^{-1/2}]$",
    ylabel=r"$|C_e(t)|^2$",
    title=rf"$\omega_0/\tau_1^{-1/2} = {detuning_freq:.2f}$",
    xscale="log",
    yscale="log",
)

ax2 = ax1.twinx()
(l3,) = ax2.plot(
    tvals,
    entropy_1,
    ls="--",
    color=dip_colors[0],
    lw=3,
    label=r"$S(t),\;\theta=90^\circ$",
)
(l4,) = ax2.plot(
    tvals,
    entropy_2,
    ls="--",
    color=dip_colors[1],
    alpha=0.9,
    lw=2.0,
    label=r"$S(t),\;\theta=135^\circ$",
)
ax2.set(ylabel=r"$S(t)$")

# --- combined legend ---
lines = [l1, l2, l3, l4]
labels = [line.get_label() for line in lines]

# ax1.legend(lines, labels, loc="upper right", fontsize=7, frameon=False)
# ax1.text(-0.2, 1.05, panel_labels[1], transform=ax1.transAxes)

# ------------------------------------------------------------------------------
# Third and fourth quadrant (c-d)
# ------------------------------------------------------------------------------

vmax = max(z1.max(), z2.max())

for ax, z, theta, color, label in zip(
    axs[1], [z1, z2], [90, 135], dip_colors, panel_labels[2:4]
):
    sc = ax.scatter(
        x,
        y,
        c=z,
        s=0.05 * mf.marker_size(len(x)),
        cmap="inferno",
        vmin=0,
        vmax=vmax,
        marker="o",
    )
    ax.set(
        xlabel=r"$x/a$", ylabel=r"$y/a$", title=rf"$\theta = {theta}$º", aspect="equal"
    )
    ax.title.set_color(color)
    ax.set_aspect("equal")
    # ax.text(-0.2, 1.1, label, transform=ax.transAxes)
    cbar = fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{r}(t)|^2$")
    cbar.formatter.set_powerlimits((0, 0))

# ------------------------------------------------------------------------------
# Fith and sixth quadrant (e-f)
# ------------------------------------------------------------------------------
for i, theta in enumerate(thetas):
    ax = axs[2, i]
    ax.plot(tvals, new_mod_Ce_1, c="k")
    ax.plot(tvals, new_mod_Ce_probe_1, c="k", ls="--")
    ax2 = ax.twinx()
    ax2.plot(tvals, concurrence_1, c="r")
    ax.set(
        title=rf"$\theta = {theta*180/np.pi}$º",
        xlabel=r"$t [\tau_1^{-1/2}]$",
        ylabel=r"$|C_e(t)|^2$",
    )
    ax.title.set_color(dip_colors[i])
    ax2.set_ylabel(r"$C(t)$", color="r")
    ax2.tick_params(axis="y", colors="r")
    ax2.spines["right"].set_color("r")

# Layout adjustment and save
fig.tight_layout(h_pad=2)
plt.show()
fig.savefig(rf"./figures/figure3.png")

# ------------------------------------------------------------------------------
# compute the photon-photon entropy for each cuadrant to show that the one with
# four chanels is equal to log(4) and the one with two chanels is equatl to log(2)
# ------------------------------------------------------------------------------
amplitudes_1 = psit_1[:, N_EMITTERS:]
amplitudes_2 = psit_2[:, N_EMITTERS:]

S_1, p_1 = mf.quadrant_entropy(x, y, amplitudes_1)
S_2, p_2 = mf.quadrant_entropy(x, y, amplitudes_2)

check_1 = np.allclose(S_1[1:], np.log(4), atol=1e-3)
check_2 = np.allclose(S_2[1:], np.log(2), atol=1e-3)

print(f"Is S_dir equal to log(4) for θ = {THETA_1*180/np.pi:.0f}º?", check_1)

print(f"Is S_dir equal to log(2) for θ = {THETA_2*180/np.pi:.0f}º?", check_2)
