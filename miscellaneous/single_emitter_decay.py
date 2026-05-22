"""
Single-emitter decay at fixed parameters, for quick testing.

θ = 90°, ω₀ = ω_X (BZ X-point), DIP_STRENGTH = 7000 D. Other settings
match kimble.py: N_CELLS = 50, N_PML = 10, GAMMA_MAX = 1e-2,
T_F = 5e-2 (in 1/Γ₀ units). Plots |C_e(t)|² vs t·Γ₀ and saves the figure.
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
# Parameters
# ------------------------------------------------------------------------------

LAT_PARAM: float = 300e-9  # lattice parameter [m]
N_BANDS: int = 2
BAND_IDX: int = 0
N_K: int = 81
RESOLUTION: int = 64
R_0: list = [0, 0]

THETA: float = 90 * np.pi / 180
DIP_STRENGTH: float = 10000  # [Debye]
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIPOLE: np.ndarray = (DIP_STRENGTH / P_A) * np.array([np.cos(THETA), np.sin(THETA), 0])

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 50
N_PML: int = 10
GAMMA_MAX: float = 1e-2

J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
N_K_PATH: int = 100

T_0: float = 0  # [1/Γ₀]
T_F: float = 3e-2  # [1/Γ₀]
T_STEPS: int = 200


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
# Build TB model and find ω_X
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
print(f"omega_0 (ω_X) = {omega_0:.6f} c/a")

temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)
_, _, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)

# ------------------------------------------------------------------------------
# Load field data and compute coupling
# ------------------------------------------------------------------------------

fields_data = mf.load(
    f"../data/kimble/fields/res_{RESOLUTION}/nk_{N_K}/meshgrid_at_{R_0}.pkl"
)
omega_ks_n = np.asarray(fields_data["freqs"])
ks = np.asarray(fields_data["kpts"])
fr0_n_ks = np.asarray(fields_data["Eks"])[0]

coupling = mf.compute_real_space_coupling(
    fr0_n_ks=fr0_n_ks,
    dipole_moment=DIPOLE,
    omega_ks_n=omega_ks_n,
    ks=ks,
    rs=rs_finite,
)[BAND_IDX, :]
idxs = np.arange(len(coupling))

# ------------------------------------------------------------------------------
# Hamiltonian and time evolution
# ------------------------------------------------------------------------------

H, _ = mf.make_light_matter_Hamiltonian(
    tb_model=finite_model,
    omega_0s=omega_0,
    gr0s_rs=[(list(coupling), list(idxs))],
    pml_dampings=pml_dampings,
)

gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / C_LIGHT * gamma_0
t_f_internal = T_F / _t_conv
ts_norm = np.linspace(0.0, T_F, T_STEPS, endpoint=True)

psi_0 = np.zeros(H.shape[0], dtype=complex)
psi_0[0] = 1
psi_t, cond_V = mf.evolve_diagonalisation(
    H,
    psi_0,
    np.linspace(0.0, t_f_internal, T_STEPS),
    return_cond=True,
)
print(f"cond(V) = {cond_V:.3e}")
mod_Ce_sq = np.abs(psi_t[:, 0]) ** 2

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(ts_norm, mod_Ce_sq, color="k")
ax.set(
    xlabel=r"$t\Gamma_0$",
    ylabel=r"$|C_e(t)|^2$",
    yscale="log",
    title=(
        rf"$\theta = {np.rad2deg(THETA):.0f}$°, "
        rf"$\omega_0 = \omega_X$, "
        rf"$p = {DIP_STRENGTH:.0f}$ D"
    ),
)

out_path = (
    f"../figures/miscellaneous/single_emitter_decay_"
    f"theta_{np.rad2deg(THETA):.0f}deg_omegaX_{DIP_STRENGTH:.0f}D.png"
)
fig.savefig(out_path)
plt.close(fig)
print(f"Saved {out_path}")
