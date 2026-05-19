"""
PML diagnostic: sweep over N_PML.

Runs the single-emitter time evolution for a range of N_PML values
(absorbing layer thickness in unit cells) while keeping GAMMA_MAX fixed.
Plots |C_e(t)|^2 over time for both dipole orientations to identify
the N_PML range that minimises reflective re-excitation without
consuming too much of the active crystal region.

Note: the active crystal width per direction is N_CELLS - 2 * N_PML.
For N_CELLS = 41, N_PML = 10 leaves 21 active sites, which is still
reasonable. Avoid N_PML > N_CELLS // 2.
"""

# Standard library
import matplotlib.pyplot as plt

# Third-party
import numpy as np
import pythtb as tb

# Local
import emitter_crystal_coupling as mf
from emitter_crystal_coupling.constants import C_LIGHT

# ------------------------------------------------------------------------------
# Fixed simulation parameters (matching kimble.py)
# ------------------------------------------------------------------------------

LAT_PARAM: float = 30e-9  # lattice parameter [m]
N_BANDS: int = 2
BAND_IDX: int = 0
N_K: int = 64
RESOLUTION: int = 64  # must match the precomputed field files
R_0: list = [0.0, 0.0]

THETA_1: float = 90 * np.pi / 180
THETA_2: float = 135 * np.pi / 180
DIP_STRENGTH: float = 1000  # [Debye]
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 41

J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
N_K_PATH: int = 100

T_0: float = 0  # [1/Γ₀]
T_F: float = 5e-2  # [1/Γ₀]
T_STEPS: int = 100

# --- Sweep parameters ---
GAMMA_MAX: float = 3e-2  # fixed maximum damping rate [c/a]
N_PML_VALUES: list = [2, 3, 4, 5, 6, 8, 10]

THETAS: list = [THETA_1, THETA_2]
DIPOLES: list = [
    DIP_STRENGTH_NORM * np.array([np.cos(theta), np.sin(theta), 0]) for theta in THETAS
]
DIP_COLORS: list = ["#1f77b4", "#ff7f0e"]


# ------------------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------------------


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    half = n_cells // 2
    dist = np.minimum(half - np.abs(rs[:, 0]), half - np.abs(rs[:, 1]))
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


# ------------------------------------------------------------------------------
# Build TB model and detuning frequency omega_0
# ------------------------------------------------------------------------------

infinite_model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
infinite_model.set_onsite([J_0])
for vec in np.array([[1, 0], [0, 1]]):
    infinite_model.set_hop(J_1, 0, 0, vec)
for vec in np.array([[1, 1], [1, -1]]):
    infinite_model.set_hop(J_2, 0, 0, vec)

k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals = infinite_model.solve_all(k_vec)
omega_0 = mf.safe_sqrt(tb_evals[BAND_IDX][np.isclose(k_dist, k_node[1])]).item()
print(f"omega_0 = {omega_0:.6f} c/a")

temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)
_, _, rs_finite = mf.build_finite_grid(N_CELLS)

# ------------------------------------------------------------------------------
# Load field data and compute couplings (independent of PML — done once)
# ------------------------------------------------------------------------------

fields_data = mf.load(
    f"./data/kimble/fields/res_{RESOLUTION}/nk_{N_K}/meshgrid_at_{R_0}.pkl"
)
omega_ks_n = np.asarray(fields_data["freqs"])
ks = np.asarray(fields_data["kpts"])
fr0_n_ks = np.asarray(fields_data["Eks"])[0]

couplings = [
    mf.compute_real_space_coupling(
        fr0_n_ks=fr0_n_ks,
        dipole_moment=dipole,
        omega_ks_n=omega_ks_n,
        ks=ks,
        rs=rs_finite,
    )[BAND_IDX, :]
    for dipole in DIPOLES
]
idxs = np.arange(len(couplings[0]))

# Time axis (convert T_F from 1/Γ₀ to a/c for the time evolution, then back for plot)
gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / C_LIGHT * gamma_0
t_f_internal = T_F / _t_conv
ts_norm = _t_conv * np.linspace(0.0, t_f_internal, T_STEPS, endpoint=True)

# Initial state: emitter fully excited
psi_0 = None

# ------------------------------------------------------------------------------
# Sweep over N_PML
# ------------------------------------------------------------------------------

# results[i][j] = |C_e(t)|^2 for n_pml index i, dipole index j
results = []

for n_pml in N_PML_VALUES:
    active_width = N_CELLS - 2 * n_pml
    print(f"  N_PML = {n_pml}  (active width = {active_width} sites) ...", flush=True)
    pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, n_pml, GAMMA_MAX)

    pops = []
    for coupling in couplings:
        H, _ = mf.make_light_matter_Hamiltonian(
            tb_model=finite_model,
            omega_0s=omega_0,
            gr0s_rs=[(list(coupling), list(idxs))],
            pml_dampings=pml_dampings,
        )
        if psi_0 is None:
            psi_0 = np.zeros(H.shape[0])
            psi_0[0] = 1
        psi_t = mf.evolve_diagonalisation(H, psi_0, np.linspace(0.0, t_f_internal, T_STEPS))
        pops.append(np.abs(psi_t[:, 0]) ** 2)

    results.append(pops)

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

cmap = plt.get_cmap("viridis")
sweep_colors = [cmap(i / (len(N_PML_VALUES) - 1)) for i in range(len(N_PML_VALUES))]

fig, axs = plt.subplots(1, 2, sharey=True)

for j, (theta, ax, dip_color) in enumerate(zip(THETAS, axs, DIP_COLORS)):
    for i, (n_pml, color) in enumerate(zip(N_PML_VALUES, sweep_colors)):
        ax.plot(
            ts_norm,
            results[i][j],
            color=color,
            label=rf"$N_\mathrm{{PML}} = {n_pml}$",
        )
    ax.set(
        xlabel=r"$t\Gamma_0$",
        ylabel=r"$|C_e(t)|^2$" if j == 0 else "",
        yscale="log",
        title=rf"$\theta = {np.rad2deg(theta):.0f}$°",
    )
    ax.title.set_color(dip_color)
    ax.legend()

fig.suptitle(
    rf"$N_\mathrm{{PML}}$ sweep, $\Gamma_\mathrm{{max}} = {GAMMA_MAX:.0e}$ $c/a$"
)
fig.savefig(f"./figures/miscellaneous/pml_n_pml_sweep_gamma_max_{GAMMA_MAX:.0e}.png")
plt.close(fig)
print(f"Saved ./figures/miscellaneous/pml_n_pml_sweep_gamma_max_{GAMMA_MAX:.0e}.png")
