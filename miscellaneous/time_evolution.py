"""

    Docstring for time_evolution.py

This file uses our formalism to compute the coupling between a quantum emitter
and a photonic crystal. We made use of a tight-binding model to simulate the
crystal and assume the emitter as a two-level system.

The excited emitter is located at [-4.5, 4.5] with an orientation of 60°. A
probe emitter at [0, 0] is then added to detect bath-mediated emission, and its
response is compared for two probe orientations (90° and 135°).

NOTE: this file assumes that the fields are already computed for the points where
the emitters are located. If not, consider running `mpb_field_computation.py`.
"""

# %% Import necessary packages
import numpy as np
import meep as mp  # needed for the numerical simulation of the photonic crystal
from meep import mpb  # same case as the previous package
import pythtb as tb  # for easy simulation of the tight-binding model
from scipy.sparse.linalg import expm_multiply

# some imports for plotting utilities
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import emitter_crystal_coupling as mf  # our package functions to work with

# %% Define the main constants and parameters for the current simulation
NUM_BANDS = 2

EPSILON_INPUT_FILE = "data/kimble/epsilon.h5"
COMPUTEFIELDS = False  # tells the code to compute the fields using MPB or not
DIRNAME_OUTPUT = "data/kimble/"  # directory to save outputs

TOLERANCE = 1e-12  # tolerance for the MPB calculations

LAT_PARAM = 300e-9  # lattice parameter [m]
N_CELLS = 41  # number of unit cells per direction
N_PML = 8  # absorbing layer thickness [unit cells per edge]
GAMMA_MAX = 3e-2  # maximum damping rate at the hard wall [c/a]

NK = 64  # number of k-points per direction for the Monkhorst-Pack grid

# Excited emitter parameters
THETA = 60 * np.pi / 180  # orientation [rad]
DIP_STRENGTH = 50  # dipole moment strength [Debye]
P_A = mf.unit_dip(a=LAT_PARAM)  # unit dipole moment [Debye]
R0 = [-4.5, 4.5]  # position in lattice units
dipole = (DIP_STRENGTH / P_A) * np.array([np.cos(THETA), np.sin(THETA), 0])

# Probe emitter parameters
PROBE_R = [0.0, 0.0]
PROBE_THETA_1 = 90 * np.pi / 180
PROBE_THETA_2 = 135 * np.pi / 180
PROBE_DIP_STRENGTH = 100  # [Debye]
probe_thetas = [PROBE_THETA_1, PROBE_THETA_2]
probe_dipoles = [
    (PROBE_DIP_STRENGTH / P_A) * np.array([np.cos(t), np.sin(t), 0])
    for t in probe_thetas
]
probe_colors = ["#1f77b4", "#ff7f0e"]

# tight-binding model parameters
LAT = [[1, 0], [0, 1]]
ORB = [[0, 0]]
BAND_INDEX = 0

# hopping coefficients [units: (c/a)²]
TAU_0 = 0.07755639
TAU_1 = -0.01993698
TAU_2 = 0.00053966

# path in the Brillouin zone for the detuning frequency computation
K_PATH = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]

# Time evolution [units: 1/Γ₀]
T0_PHYS = 0
TF_PHYS = 5e-5
NEW_TF_PHYS = 5e-5
T_STEPS = 100
TIME_INDEX = 50  # snapshot index for the single-emitter bath plot
NEW_TIME_INDEX = 25  # snapshot index for the two-emitter bath plot

N_EMITTERS = 1  # single excited emitter
NEW_N_EMITTERS = 2  # excited + probe

# ---------------------------------------------------------------------------- #
# Build the infinite tight-binding model (unnormalized, matching kimble.py)
# ---------------------------------------------------------------------------- #

infiniteModel = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
infiniteModel.set_onsite([TAU_0])
for vec in np.array([[1, 0], [0, 1]]):
    infiniteModel.set_hop(TAU_1, 0, 0, vec)
for vec in np.array([[1, 1], [1, -1]]):
    infiniteModel.set_hop(TAU_2, 0, 0, vec)

# Compute detuning frequency at the X point of the Brillouin zone
k_vec, k_dist, k_node = infiniteModel.k_path(K_PATH, 100, report=False)
tb_evals = infiniteModel.solve_all(k_vec)
x_mask = np.isclose(k_dist, k_node[1])
omega_0 = mf.safe_sqrt(tb_evals[BAND_INDEX][x_mask]).item()

# Time conversion: from 1/Γ₀ to a/c
gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / mf.constants.C_LIGHT * gamma_0
T0 = T0_PHYS / _t_conv
TF = TF_PHYS / _t_conv
NEW_TF = NEW_TF_PHYS / _t_conv

tvals = np.linspace(T0, TF, T_STEPS, endpoint=True)
tvals_norm = _t_conv * tvals  # [1/Γ₀]
new_tvals = np.linspace(T0, NEW_TF, T_STEPS, endpoint=True)
new_tvals_norm = _t_conv * new_tvals  # [1/Γ₀]

# %% Build the finite tight-binding model
temporalModel = infiniteModel.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finiteModel = temporalModel.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)

# %% Build the finite real-space grid and PML dampings
xjg, yjg, rjpts = mf.build_finite_grid(N_CELLS)

half = N_CELLS // 2
dist_x = half - np.abs(rjpts[:, 0])
dist_y = half - np.abs(rjpts[:, 1])
dist = np.minimum(dist_x, dist_y)
pml_dampings = np.where(dist < N_PML, GAMMA_MAX * ((N_PML - dist) / N_PML) ** 2, 0.0)

# %% Load field data for the excited emitter at R0=[-4.5, 4.5]
data = mf.load(f"./data/kimble/fields/{NK}_meshgrid_at_{R0}.pkl")
omegas = np.asarray(data["freqs"])  # (Nk, Nb)
Eks = np.asarray(data["Eks"])  # (Nr, Nb, Nk, 3)
kpts = np.asarray(data["kpts"])  # (Nk, 3)
E_k_r0 = Eks[0, :, :, :]  # (Nb, Nk, 3)

# Compute real-space coupling for the excited emitter
gr0rj = mf.compute_real_space_coupling(E_k_r0, dipole, omegas, kpts, rjpts)
value = gr0rj[BAND_INDEX, :]
indexes = np.arange(0, len(value), 1)
couplings = [[list(value), list(indexes)]]

# %% Build the single-emitter light-matter Hamiltonian
H_total, omega_sites = mf.make_light_matter_Hamiltonian(
    tb_model=finiteModel,
    omega_0s=omega_0,
    gr0s_rs=couplings,
    pml_dampings=pml_dampings,
)

# %% Compute time evolution (single excited emitter)
psi0 = np.zeros(H_total.shape[0])
psi0[0] = 1

psit = expm_multiply(-1j * H_total, psi0, T0, TF, T_STEPS, endpoint=True)

# ---------------------------------------------------------------------------- #
# Probe emitter at [0, 0]
# ---------------------------------------------------------------------------- #

# Load field data for the probe at PROBE_R=[0.0, 0.0]
probe_data = mf.load(f"./data/kimble/fields/{NK}_meshgrid_at_{PROBE_R}.pkl")
probe_omegas = np.asarray(probe_data["freqs"])
probe_Eks = np.asarray(probe_data["Eks"])
probe_kpts = np.asarray(probe_data["kpts"])
E_k_probe = probe_Eks[0, :, :, :]  # (Nb, Nk, 3)

probe_detuning = omega_0

# Compute couplings for the probe dipoles (two orientations)
probe_values = [
    mf.compute_real_space_coupling(E_k_probe, pd, probe_omegas, probe_kpts, rjpts)[
        BAND_INDEX, :
    ]
    for pd in probe_dipoles
]

# Build two-emitter Hamiltonians (one per probe orientation)
two_em_couplings = [
    [[list(value), list(indexes)], [list(pv), list(indexes)]] for pv in probe_values
]

new_Hs = [
    mf.make_light_matter_Hamiltonian(
        tb_model=finiteModel,
        omega_0s=[omega_0, probe_detuning],
        gr0s_rs=c,
        pml_dampings=pml_dampings,
    )[0]
    for c in two_em_couplings
]

# Initial state: only the excited emitter populated
new_psi0 = np.zeros(new_Hs[0].shape[0])
new_psi0[0] = 1

# Compute time evolutions for both probe orientations
new_psits = [
    expm_multiply(-1j * H, new_psi0, T0, NEW_TF, T_STEPS, endpoint=True) for H in new_Hs
]

# ---------------------------------------------------------------------------- #
# Prepare data to be plotted
# ---------------------------------------------------------------------------- #

# Single-emitter data
mod_Ce_squared = np.abs(psit[:, 0]) ** 2
mod_Cr_squared = np.abs(psit[TIME_INDEX, N_EMITTERS:]) ** 2
z = mod_Cr_squared.reshape(N_CELLS, N_CELLS)

# Two-emitter data
new_mod_Ces = [np.abs(npt[:, 0]) ** 2 for npt in new_psits]
new_mod_Ce_probes = [np.abs(npt[:, 1]) ** 2 for npt in new_psits]

# ---------------------------------------------------------------------------- #
# Plot 1: Emitter decay (single excited emitter)
# ---------------------------------------------------------------------------- #
fig, ax = plt.subplots()

ax.plot(tvals_norm, mod_Ce_squared, c="k")
ax.set(
    xlabel=r"$t\Gamma_0$",
    ylabel=r"$|C_e(t)|^2$",
    title=rf"$\theta = {np.rad2deg(THETA):.0f}°$",
    yscale="log",
)

fig.tight_layout()
plt.show()
fig.savefig(
    f"./figures/miscellaneous/emitter_decay_at_{R0}_theta_{np.rad2deg(THETA):.0f}deg.png"
)

# ---------------------------------------------------------------------------- #
# Plot 2: Bath population snapshot (single emitter)
# ---------------------------------------------------------------------------- #
fig, ax = plt.subplots()

sc = ax.pcolormesh(xjg, yjg, z, vmin=0)
mf.draw_arrow(ax, R0, THETA, length=3)
ax.set(
    xlabel=r"$x/a$",
    ylabel=r"$y/a$",
    title=rf"$\theta = {np.rad2deg(THETA):.0f}°$, $t\Gamma_0 = {tvals_norm[TIME_INDEX]:.2e}$",
    aspect="equal",
)
cbar = fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{r}|^2$", shrink=0.8)
cbar.formatter.set_powerlimits((0, 0))

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{R0}_bath_population.png")

# ---------------------------------------------------------------------------- #
# Plot 3: Probe coupling strength for each probe orientation
# ---------------------------------------------------------------------------- #
fig, axs = plt.subplots(1, 2)

for i, (pv, ptheta, color) in enumerate(zip(probe_values, probe_thetas, probe_colors)):
    ax = axs[i]
    cbar_max = np.abs(pv).max()
    sc = ax.scatter(
        rjpts[:, 0],
        rjpts[:, 1],
        c=np.abs(pv),
        vmin=0,
        vmax=cbar_max,
        marker="s",
        s=0.71 * mf.marker_size(len(rjpts[:, 0])),
    )
    mf.draw_arrow(ax, PROBE_R, ptheta, length=10)
    ax.set(
        title=rf"$\theta_p = {np.rad2deg(ptheta):.0f}°$",
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
    )
    ax.title.set_color(color)
    cbar = fig.colorbar(sc, ax=ax, label=r"$|g_\mathbf{r}(\mathbf{R})|$")
    cbar.formatter.set_powerlimits((0, 0))

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{PROBE_R}_coupling_check.png")

# ---------------------------------------------------------------------------- #
# Plot 4: Two-emitter time evolution (excited at R0, probe at PROBE_R)
# ---------------------------------------------------------------------------- #
fig, axs = plt.subplots(1, 2)

for i, (ptheta, color) in enumerate(zip(probe_thetas, probe_colors)):
    ax = axs[i]
    ax.plot(new_tvals_norm, new_mod_Ces[i], c="k", label="Excited")
    ax.plot(new_tvals_norm, new_mod_Ce_probes[i], c=color, ls="--", label="Probe")
    ax.set(
        title=rf"$\theta_p = {np.rad2deg(ptheta):.0f}°$",
        xlabel=r"$t\Gamma_0$",
        ylabel=r"$|C_e(t)|^2$",
        yscale="log",
    )
    ax.title.set_color(color)
    ax.legend()

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/emit_pop_at_{PROBE_R}_exciting_at_{R0}.png")
