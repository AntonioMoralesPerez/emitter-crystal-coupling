"""
Single-emitter decay comparing two resonances: ω₀ = ω_X (BZ X-point)
vs ω₀ = ω_Γmax (frequency that maximizes Γ(ω)).

θ = 90°, DIP_STRENGTH = 10000 D. Other settings match kimble.py:
N_CELLS = 50, N_PML = 10, GAMMA_MAX = 1e-2, T_F = 3e-2 (in 1/Γ₀ units).

Produces two figures:
  (a) |C_e(t)|² vs t·Γ₀ overlaying both resonances.
  (b) bath population snapshot (at SNAP_IDX = 5) for each resonance.
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
SNAP_IDX: int = 5  # bath snapshot time index (matches kimble.py)

# Resonance for Γ(ω) maximum (from kimble.py)
OMEGA_GAMMA_MAX: float = 0.262631  # c/a


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    half = n_cells // 2
    dist = np.minimum(half - np.abs(rs[:, 0]), half - np.abs(rs[:, 1]))
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


def run_evolution(omega_0: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Build H at the given resonance, evolve, and return (|C_e|², ψ(t), Γ₀)."""
    H, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=omega_0,
        gr0s_rs=[(list(coupling), list(idxs))],
        pml_dampings=pml_dampings,
    )
    gamma_0 = mf.free_spontaneous_emission(
        omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
    )
    t_conv = LAT_PARAM / C_LIGHT * gamma_0
    t_f_internal = T_F / t_conv

    psi_0 = np.zeros(H.shape[0], dtype=complex)
    psi_0[0] = 1
    psi_t, cond_V = mf.evolve_diagonalisation(
        H,
        psi_0,
        np.linspace(0.0, t_f_internal, T_STEPS),
        return_cond=True,
    )
    print(f"  cond(V) = {cond_V:.3e}, Γ₀ = {gamma_0:.3e} c/a")
    return np.abs(psi_t[:, 0]) ** 2, psi_t, gamma_0


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
OMEGA_X: float = mf.safe_sqrt(
    tb_evals[BAND_IDX][np.isclose(k_dist, k_node[1])]
).item()
print(f"ω_X        = {OMEGA_X:.6f} c/a")
print(f"ω_Γmax     = {OMEGA_GAMMA_MAX:.6f} c/a")

temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)
xs_finite, ys_finite, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)

# ------------------------------------------------------------------------------
# Load field data and compute coupling (resonance-independent)
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
# Evolve at both resonances
# ------------------------------------------------------------------------------

ts_norm = np.linspace(0.0, T_F, T_STEPS, endpoint=True)

print("Evolving at ω₀ = ω_X ...")
mod_Ce_sq_X, psi_t_X, _ = run_evolution(OMEGA_X)
print("Evolving at ω₀ = ω_Γmax ...")
mod_Ce_sq_G, psi_t_G, _ = run_evolution(OMEGA_GAMMA_MAX)

# ------------------------------------------------------------------------------
# Figure (a): emitter population overlay
# ------------------------------------------------------------------------------

fig, ax = plt.subplots()
ax.plot(ts_norm, mod_Ce_sq_X, label=r"$\omega_0 = \omega_X$", color="k")
ax.plot(
    ts_norm,
    mod_Ce_sq_G,
    label=r"$\omega_0 = \omega_{\Gamma_{\max}}$",
    color="#d62728",
)
ax.set(
    xlabel=r"$t\Gamma_0$",
    ylabel=r"$|C_e(t)|^2$",
    yscale="log",
    title=(
        rf"$\theta = {np.rad2deg(THETA):.0f}$°, "
        rf"$p = {DIP_STRENGTH:.0f}$ D"
    ),
)
ax.legend()
out_path_a = (
    f"../figures/miscellaneous/single_emitter_decay_compare_"
    f"theta_{np.rad2deg(THETA):.0f}deg_{DIP_STRENGTH:.0f}D.png"
)
fig.savefig(out_path_a)
plt.close(fig)
print(f"Saved {out_path_a}")

# ------------------------------------------------------------------------------
# Figure (b): bath snapshot at each resonance (single time index)
# ------------------------------------------------------------------------------

n_emitters = 1
pos_x = np.unique(rs_finite[:, 0])
pos_y = np.unique(rs_finite[:, 1])

z_X = (np.abs(psi_t_X[SNAP_IDX, n_emitters:]) ** 2).reshape(len(pos_x), len(pos_y))
z_G = (np.abs(psi_t_G[SNAP_IDX, n_emitters:]) ** 2).reshape(len(pos_x), len(pos_y))
cbar_max = max(z_X.max(), z_G.max())

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
for ax, z, label in zip(
    axes,
    [z_X, z_G],
    [r"$\omega_0 = \omega_X$", r"$\omega_0 = \omega_{\Gamma_{\max}}$"],
):
    sc = ax.pcolormesh(pos_x, pos_y, z, vmin=0, vmax=cbar_max)
    ax.set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
        title=label,
    )
    fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{R}(t)|^2$", shrink=0.8)
fig.suptitle(
    rf"Bath snapshot at $t\Gamma_0 = {ts_norm[SNAP_IDX]:.3e}$ "
    rf"($\theta = {np.rad2deg(THETA):.0f}$°, $p = {DIP_STRENGTH:.0f}$ D)"
)
out_path_b = (
    f"../figures/miscellaneous/single_emitter_bath_snapshot_compare_"
    f"theta_{np.rad2deg(THETA):.0f}deg_{DIP_STRENGTH:.0f}D.png"
)
fig.savefig(out_path_b)
plt.close(fig)
print(f"Saved {out_path_b}")
