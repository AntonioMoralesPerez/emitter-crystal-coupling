"""
    Docstring for polarization_sweep.py

This is a script for internal check on how polarization affects the emitter
dynamics. An emitter position is fixed and a sweep for different polarizations is
performed.
"""

# %% Import necessary packages
from scipy.sparse.linalg import expm_multiply

import numpy as np
import pythtb as tb
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import emitter_crystal_coupling as mf

# %% Define the main constants and parameters for the current simulation
BAND_IDX = 0

EPSILON_INPUT_FILE = "data/kimble/epsilon.h5"
DIRNAME_OUTPUT = "data/miscellaneous/"

N_CELLS = 41
N_PML: int = 8
GAMMA_MAX: float = 3e-2

N_K = 101

# angles for the dipole moments of the quantum emitters and its dimensionless value
THETAS = np.linspace(90, 140, 6) * np.pi / 180
LAT_PARAM: float = 300e-9  # lattice parameter [m]
DIP_STRENGTH: float = 7000  # dipole moment strength [Debye]
DETUNING_FREQ = 0.274587  # [c/a]

N_EMITTERS = 1
R_0 = [0, 0]

LAT = [[1, 0], [0, 1]]
ORB = [[0, 0]]

# Hopping coefficients up to 2nd nearest neighbours [units: (c/a)²]
J_0 = 0.07755639
J_1 = -0.01993698
J_2 = 0.00053966

T_0 = 0
T_F = 1e-2
T_STEPS = 100
SNAP_IDX = 30

# Real-space grid
N_R: int = 13  # number of grid points per direction

DIP_STRENGTH_NORM = DIP_STRENGTH / mf.unit_dip(a=LAT_PARAM)
DIPOLES = [
    DIP_STRENGTH_NORM * np.array([np.cos(theta), np.sin(theta), 0]) for theta in THETAS
]

# ---------------------------------------------------------------------------- #
# Load the fields and k-points from the MPB calculation
# ---------------------------------------------------------------------------- #

data = mf.load(f"./data/kimble/fields/{N_K}_meshgrid_at_{R_0}.pkl")

omega_ks_n = np.asarray(data["freqs"])  # (Nk, Nb)
ks = np.asarray(data["kpts"])  # (Nk, 3)
fr0_n_ks = np.asarray(data["Eks"])[0]  # (Nb, Nk, 3)

# ---------------------------------------------------------------------------- #
# Define the real-space grid for the coupling visualization
# ---------------------------------------------------------------------------- #
xg = np.arange(-(N_R // 2), N_R // 2 + 1)
yg = np.arange(-(N_R // 2), N_R // 2 + 1)
rpts = mf.meshgrid_points(xg, yg, indexing="xy")

# ---------------------------------------------------------------------------- #
# Compute g_r(r_0) for each dipole, shape (len(THETAS), Nr)
# ---------------------------------------------------------------------------- #
gr0rs = [
    mf.compute_real_space_coupling(
        fr0_n_ks=fr0_n_ks,
        dipole_moment=dipole,
        omega_ks_n=omega_ks_n,
        ks=ks,
        rs=rpts,
    )[BAND_IDX, :]
    for dipole in DIPOLES
]

# ------------------------------------------------------------------------------
# Plot the different couplings for the different dipole moments
# ------------------------------------------------------------------------------
theta_colors = [f"C{i}" for i in range(len(THETAS))]
cbar_min = np.abs(gr0rs).min()
cbar_max = np.abs(gr0rs).max()

n_cols = 3
n_rows = int(np.ceil(len(THETAS) / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)

for i, theta in enumerate(THETAS):
    ax = axs[i // n_cols, i % n_cols]
    sc = ax.pcolormesh(
        xg,
        yg,
        np.abs(gr0rs[i]).reshape(len(yg), len(xg)),
        norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max),
    )
    mf.draw_arrow(ax, R_0, theta, color=theta_colors[i], length=4, arrow_offset=0.4)
    ax.set(
        title=rf"$\theta = {np.rad2deg(theta):.0f}$°",
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
    )
    ax.title.set_color(theta_colors[i])

fig.colorbar(
    sc, ax=axs.ravel().tolist(), label=r"$|g_\mathbf{R}(\mathbf{r}_0)| \, [c/a]$"
)
fig.savefig(f"./figures/miscellaneous/g_r({R_0})_polarization_sweep_couplings.png")
plt.close(fig)

# ------------------------------------------------------------------------------
# Build the tight-binding model
# ------------------------------------------------------------------------------


def build_tight_binding_model() -> tb.tb_model:
    model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
    model.set_onsite([J_0])
    for vec in np.array([[1, 0], [0, 1]]):
        model.set_hop(J_1, 0, 0, vec)
    for vec in np.array([[1, 1], [1, -1]]):
        model.set_hop(J_2, 0, 0, vec)
    return model


def build_finite_model(infinite_model: tb.tb_model) -> tb.tb_model:
    temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
    return temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    half = n_cells // 2
    dist_x = half - np.abs(rs[:, 0])
    dist_y = half - np.abs(rs[:, 1])
    dist = np.minimum(dist_x, dist_y)
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


infinite_model = build_tight_binding_model()
finite_model = build_finite_model(infinite_model)
xs_finite, ys_finite, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)

# %% Compute g(r0, rj) on the finite grid

gr0rjs = [
    mf.compute_real_space_coupling(
        fr0_n_ks=fr0_n_ks,
        dipole_moment=dipole,
        omega_ks_n=omega_ks_n,
        ks=ks,
        rs=rs_finite,
    )[BAND_IDX, :]
    for dipole in DIPOLES
]

# %% Build the light-matter Hamiltonian

idxs = np.arange(len(gr0rjs[0]))
interactions = [[[list(gr0rj), list(idxs)]] for gr0rj in gr0rjs]

hamiltonians = [
    mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=DETUNING_FREQ,
        gr0s_rs=interaction,
        pml_dampings=pml_dampings,
    )[0]
    for interaction in interactions
]

# %% Compute time evolution

gamma_0 = mf.free_spontaneous_emission(
    omega_0=DETUNING_FREQ, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / mf.constants.C_LIGHT * gamma_0
_T_0_ac = T_0 / _t_conv  # [a/c]
_T_F_ac = T_F / _t_conv  # [a/c]

ts_norm = np.linspace(T_0, T_F, T_STEPS, endpoint=True)  # [1/Γ₀]

psi0 = np.zeros(hamiltonians[0].shape[0])
psi0[0] = 1

psits = [
    expm_multiply(-1j * hamiltonian, psi0, _T_0_ac, _T_F_ac, T_STEPS, endpoint=True)
    for hamiltonian in hamiltonians
]

mod_Ce_squareds = [np.abs(psit[:, 0]) ** 2 for psit in psits]

positions = mf.find_positions(finite_model._orb, finite_model._lat)
mod_Cr_squareds = [np.abs(psit[SNAP_IDX, N_EMITTERS:]) ** 2 for psit in psits]

x = positions[:, 0]
y = positions[:, 1]

# ------------------------------------------------------------------------------
# Plot the time evolutions for the different polarizations
# ------------------------------------------------------------------------------
fig, ax = plt.subplots()

for i, theta in enumerate(THETAS):
    ax.plot(
        ts_norm, mod_Ce_squareds[i], label=rf"$\theta = {theta*180/np.pi:.0f}^\circ$"
    )

ax.legend()
ax.set(xlabel=r"$t \Gamma_0$", ylabel=r"$|C_e(t)|^2$", xscale="log", yscale="log")

fig.tight_layout()
plt.show()
fig.savefig(f"./figures/miscellaneous/{R_0}_polarization_sweep_emit_population.png")

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
fig.savefig(f"./figures/miscellaneous/{R_0}_polarization_sweep_bath_population.png")
