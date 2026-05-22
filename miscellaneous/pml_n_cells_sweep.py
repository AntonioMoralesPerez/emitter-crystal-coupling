"""
PML diagnostic: sweep over N_CELLS.

Discriminates between a physical revival and a PML-reflection artifact in
the single-emitter time evolution. With N_PML and GAMMA_MAX held fixed,
the active crystal width grows linearly with N_CELLS. A genuine bulk
revival should sit at the same t·Γ₀ across sizes (modulo finite-size
shifts at small N_CELLS); a reflection-driven revival should drift to
later times as (N_CELLS - 2·N_PML) / v_g.

Plots |C_e(t)|^2 over time for a single orientation (θ = 90°).
"""

# Standard library
import matplotlib.pyplot as plt

# Third-party
import numpy as np
import pythtb as tb
import scipy.linalg
from scipy.sparse.linalg import expm_multiply

# Local
import emitter_crystal_coupling as mf
from emitter_crystal_coupling.constants import C_LIGHT

# ------------------------------------------------------------------------------
# Fixed simulation parameters
# ------------------------------------------------------------------------------

LAT_PARAM: float = 300e-9  # lattice parameter [m]
N_BANDS: int = 2
BAND_IDX: int = 0
N_K: int = 81
RESOLUTION: int = 64  # must match the precomputed field files
R_0: list = [0, 0]

THETA: float = 90 * np.pi / 180
DIP_STRENGTH: float = 1000  # [Debye]
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A
DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array(
    [np.cos(THETA), np.sin(THETA), 0]
)
DIP_COLOR: str = "#1f77b4"

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]

J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
N_K_PATH: int = 100

T_0: float = 0  # [1/Γ₀]
T_F: float = 1e-1  # [1/Γ₀]
T_STEPS: int = 100

# --- Sweep parameters ---
N_PML: int = 10  # fixed PML thickness [unit cells]
GAMMA_MAX: float = 1e-2  # fixed maximum damping rate [c/a]
N_CELLS_VALUES: list = [35, 41, 50, 60]


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
# Build infinite TB model and detuning frequency omega_0 (N_CELLS-independent)
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

# ------------------------------------------------------------------------------
# Load field data (N_CELLS-independent — k-space data)
# ------------------------------------------------------------------------------

fields_data = mf.load(
    f"../data/kimble/fields/res_{RESOLUTION}/nk_{N_K}/meshgrid_at_{R_0}.pkl"
)
omega_ks_n = np.asarray(fields_data["freqs"])
ks = np.asarray(fields_data["kpts"])
fr0_n_ks = np.asarray(fields_data["Eks"])[0]

# Time axis (convert T_F from 1/Γ₀ to a/c). Γ₀ is N_CELLS-independent.
gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / C_LIGHT * gamma_0
t_f_internal = T_F / _t_conv
ts_norm = _t_conv * np.linspace(0.0, t_f_internal, T_STEPS, endpoint=True)

# ------------------------------------------------------------------------------
# Sweep over N_CELLS
# ------------------------------------------------------------------------------

results_diag = []
results_expm = []
cond_Vs = []

times_internal = np.linspace(0.0, t_f_internal, T_STEPS)

for n_cells in N_CELLS_VALUES:
    active_width = n_cells - 2 * N_PML
    print(
        f"  N_CELLS = {n_cells}  (active width = {active_width} sites) ...",
        flush=True,
    )

    temporal_model = infinite_model.cut_piece(n_cells, fin_dir=0, glue_edgs=False)
    finite_model = temporal_model.cut_piece(n_cells, fin_dir=1, glue_edgs=False)
    _, _, rs_finite = mf.build_finite_grid(n_cells)

    coupling = mf.compute_real_space_coupling(
        fr0_n_ks=fr0_n_ks,
        dipole_moment=DIPOLE,
        omega_ks_n=omega_ks_n,
        ks=ks,
        rs=rs_finite,
    )[BAND_IDX, :]
    idxs = np.arange(len(coupling))

    pml_dampings = compute_pml_dampings(rs_finite, n_cells, N_PML, GAMMA_MAX)

    H, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=omega_0,
        gr0s_rs=[(list(coupling), list(idxs))],
        pml_dampings=pml_dampings,
    )
    psi_0 = np.zeros(H.shape[0], dtype=complex)
    psi_0[0] = 1

    # (a) Diagonalisation path + condition number of eigenvector matrix
    lam, V = scipy.linalg.eig(H)
    cond_V = np.linalg.cond(V)
    cond_Vs.append(cond_V)
    print(f"    cond(V) = {cond_V:.3e}")
    c = np.linalg.solve(V, psi_0)
    phase = np.exp(-1j * np.outer(lam, times_internal))
    psi_t_diag = (V @ (phase * c[:, None])).T
    results_diag.append(np.abs(psi_t_diag[:, 0]) ** 2)

    # (b) Krylov path: expm_multiply, independent of cond(V)
    psi_t_expm = expm_multiply(
        -1j * H,
        psi_0,
        start=0.0,
        stop=t_f_internal,
        num=T_STEPS,
        endpoint=True,
    )
    results_expm.append(np.abs(psi_t_expm[:, 0]) ** 2)

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

cmap = plt.get_cmap("viridis")
sweep_colors = [
    cmap(i / (len(N_CELLS_VALUES) - 1)) for i in range(len(N_CELLS_VALUES))
]

fig, ax = plt.subplots()

for n_cells, color, pop_diag, pop_expm, cond_V in zip(
    N_CELLS_VALUES, sweep_colors, results_diag, results_expm, cond_Vs
):
    ax.plot(
        ts_norm,
        pop_diag,
        color=color,
        ls="-",
        label=rf"$N_\mathrm{{cells}} = {n_cells}$, $\kappa(V)={cond_V:.1e}$",
    )
    ax.plot(ts_norm, pop_expm, color=color, ls="--")

ax.set(
    xlabel=r"$t\Gamma_0$",
    ylabel=r"$|C_e(t)|^2$",
    yscale="log",
    title=rf"$\theta = {np.rad2deg(THETA):.0f}$° — solid: diag, dashed: expm_multiply",
)
ax.title.set_color(DIP_COLOR)
ax.legend(fontsize=7)

fig.suptitle(
    rf"$N_\mathrm{{cells}}$ sweep, $N_\mathrm{{PML}} = {N_PML}$, "
    rf"$\Gamma_\mathrm{{max}} = {GAMMA_MAX:.0e}$ $c/a$"
)
out_path = (
    f"../figures/miscellaneous/pml_n_cells_sweep_N_PML_{N_PML}_"
    f"gamma_max_{GAMMA_MAX:.0e}.png"
)
fig.savefig(out_path)
plt.close(fig)
print(f"Saved {out_path}")
