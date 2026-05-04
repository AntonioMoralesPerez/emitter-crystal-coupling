"""
Bath excitation orientation sweep.

Fixes a single emitter at EMITTER_R=[-4.5, 4.5] with dipole strength 100 D and
sweeps its orientation from 0° to 180°. For each orientation the time evolution
is computed and the bath population snapshot at TIME_INDEX is plotted.

Parameters and conventions match time_evolution.py: same PML settings, same
time units (1/Γ₀), same TB model.

NOTE: run mpb_field_computation.py at EMITTER_R before executing.
"""

import numpy as np
import matplotlib.pyplot as plt
import pythtb as tb
from scipy.sparse.linalg import expm_multiply

import emitter_crystal_coupling as mf

# %% Parameters

LAT_PARAM: float = 300e-9
DIP_STRENGTH: float = 100  # [Debye]
P_A: float = mf.unit_dip(a=LAT_PARAM)

BAND_IDX: int = 0
N_K: int = 64

EMITTER_R: list = [0.0, 0.0]
N_THETAS: int = 13
THETAS: np.ndarray = np.linspace(0, 180, N_THETAS) * np.pi / 180

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 41
N_PML: int = 8
GAMMA_MAX: float = 3e-2

J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]

T_F: float = 5e-5  # end time [1/Γ₀]
T_STEPS: int = 100
TIME_INDEX: int = 50

# %% Tight-binding model and omega_0

infinite_model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
infinite_model.set_onsite([J_0])
for vec in [[1, 0], [0, 1]]:
    infinite_model.set_hop(J_1, 0, 0, vec)
for vec in [[1, 1], [1, -1]]:
    infinite_model.set_hop(J_2, 0, 0, vec)

k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, 100, report=False)
tb_evals = infinite_model.solve_all(k_vec)
omega_0 = mf.safe_sqrt(tb_evals[BAND_IDX][np.isclose(k_dist, k_node[1])]).item()
print(f"omega_0 = {omega_0:.6f} c/a")

temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)
xjg, yjg, rs_finite = mf.build_finite_grid(N_CELLS)

half = N_CELLS // 2
dist_x = half - np.abs(rs_finite[:, 0])
dist_y = half - np.abs(rs_finite[:, 1])
dist = np.minimum(dist_x, dist_y)
pml_dampings = np.where(dist < N_PML, GAMMA_MAX * ((N_PML - dist) / N_PML) ** 2, 0.0)
indexes = np.arange(len(rs_finite))

# %% Time conversion (1/Γ₀ → a/c for internal use)

gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / mf.constants.C_LIGHT * gamma_0
t_f_internal = T_F / _t_conv
tvals_norm = np.linspace(0.0, T_F, T_STEPS, endpoint=True)

# %% Load emitter fields

data = mf.load(f"./data/kimble/fields/{N_K}_meshgrid_at_{EMITTER_R}.pkl")
omega_ks_n = np.asarray(data["freqs"])
fr0_n_ks = np.asarray(data["Eks"])[0]
ks = np.asarray(data["kpts"])

# %% Orientation sweep — compute all bath snapshots

zs = []
for i, theta in enumerate(THETAS):
    dipole = (DIP_STRENGTH / P_A) * np.array([np.cos(theta), np.sin(theta), 0])
    coupling = mf.compute_real_space_coupling(
        fr0_n_ks=fr0_n_ks,
        dipole_moment=dipole,
        omega_ks_n=omega_ks_n,
        ks=ks,
        rs=rs_finite,
    )[BAND_IDX, :]

    H, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=omega_0,
        gr0s_rs=[[list(coupling), list(indexes)]],
        pml_dampings=pml_dampings,
    )

    psi0 = np.zeros(H.shape[0])
    psi0[0] = 1
    psit = expm_multiply(-1j * H, psi0, 0.0, t_f_internal, T_STEPS, endpoint=True)
    z = np.abs(psit[TIME_INDEX, 1:]) ** 2
    zs.append(z.reshape(N_CELLS, N_CELLS))
    print(f"θ = {np.rad2deg(theta):.0f}° done")

vmax = max(z.max() for z in zs)

# %% Plot

N_COLS = 5
N_ROWS = int(np.ceil(N_THETAS / N_COLS))

fig, axs = plt.subplots(N_ROWS, N_COLS, figsize=(3 * N_COLS, 3.2 * N_ROWS))

for i, (theta, z) in enumerate(zip(THETAS, zs)):
    ax = axs.flat[i]
    sc = ax.pcolormesh(xjg, yjg, z, vmin=0, vmax=vmax)
    mf.draw_arrow(ax, EMITTER_R, theta, length=10)
    ax.set(
        title=rf"$\theta = {np.rad2deg(theta):.0f}°$",
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
    )

for ax in axs.flat[N_THETAS:]:
    ax.set_visible(False)

fig.colorbar(
    sc, ax=axs.flat[:N_THETAS].tolist(), label=r"$|C_\mathbf{r}|^2$", shrink=0.6
)
fig.suptitle(
    rf"$R_0 = {EMITTER_R}$,  $p = {DIP_STRENGTH:.0f}$ D,  $t\Gamma_0 = {tvals_norm[TIME_INDEX]:.2e}$",
)

fig.tight_layout()
fig.savefig(f"./figures/miscellaneous/bath_excitation_sweep_at_{EMITTER_R}.png")
plt.show()
