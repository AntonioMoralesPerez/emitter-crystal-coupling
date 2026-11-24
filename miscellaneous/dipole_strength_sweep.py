"""
DIP_STRENGTH sweep around the revival regime, with ω₀ = ω_X.

Runs the single-emitter time evolution (θ = 90°) for a fine sweep of
dipole strengths bracketing the regime where a single non-Markovian
revival emerges. Uses ω₀ = ω_X (band frequency at the BZ X point — a
saddle van Hove singularity), instead of argmax Γ(ω). Other parameters
(pre-2π kimble.py values): N_CELLS = 50, N_PML = 10, GAMMA_MAX = 2π·1e-2,
T_F = (2π)²·5e-2 (in 1/Γ₀ units).

Each curve is plotted vs t·Γ₀(self) so the Markovian baseline lies on
the same diagonal across the sweep and only the non-Markovian content
varies between curves.
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

LAT_PARAM: float = 300e-9  # lattice parameter [m]
N_BANDS: int = 2
BAND_IDX: int = 0
N_K: int = 81
RESOLUTION: int = 64  # must match the precomputed field files
R_0: list = [0, 0]

THETA: float = 90 * np.pi / 180  # single orientation for the sweep
P_A: float = float(mf.unit_dip(a=LAT_PARAM))

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 50
N_PML: int = 10
GAMMA_MAX: float = 2 * np.pi * 1e-2

J_0: float = 3.06180355
J_1: float = -0.78708042
J_2: float = 0.02130492

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
N_K_PATH: int = 100

T_0: float = 0  # [1/Γ₀]
T_F: float = (2 * np.pi) ** 2 * 5e-2  # [1/Γ₀]
T_STEPS: int = 200

# --- Sweep parameters ---
DIP_STRENGTH_VALUES: list = list(range(4300, 5100, 100))  # 4300, 4400, ..., 5000


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
# Build TB model (independent of DIP_STRENGTH)
# ------------------------------------------------------------------------------

infinite_model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
infinite_model.set_onsite([J_0])
for vec in np.array([[1, 0], [0, 1]]):
    infinite_model.set_hop(J_1, 0, 0, vec)
for vec in np.array([[1, 1], [1, -1]]):
    infinite_model.set_hop(J_2, 0, 0, vec)

temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)
_, _, rs_finite = mf.build_finite_grid(N_CELLS)

pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)

# ------------------------------------------------------------------------------
# Load field data (independent of DIP_STRENGTH)
# ------------------------------------------------------------------------------

fields_data = mf.load(
    f"../data/kimble/fields/res_{RESOLUTION}/nk_{N_K}/meshgrid_at_{R_0}.pkl"
)
omega_ks_n = 2 * np.pi * np.asarray(fields_data["freqs"])
ks = np.asarray(fields_data["kpts"])
fr0_n_ks = np.asarray(fields_data["Eks"])[0]

# ------------------------------------------------------------------------------
# Determine omega_0 = ω_X (saddle van Hove singularity at the X point of BZ).
# ------------------------------------------------------------------------------

k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals = infinite_model.solve_all(k_vec)
omega_0 = mf.safe_sqrt(tb_evals[BAND_IDX][np.isclose(k_dist, k_node[1])]).item()
print(f"omega_0 (ω_X) = {omega_0:.6f} c/a")

# ------------------------------------------------------------------------------
# Sweep over DIP_STRENGTH
# ------------------------------------------------------------------------------

times_internal_unit = np.linspace(0.0, 1.0, T_STEPS, endpoint=True)
# Per-curve t·Γ₀ axis is identical across the sweep, since each run uses its
# own _t_conv: ts_norm = _t_conv · linspace(0, T_F/_t_conv) = linspace(0, T_F).
ts_norm = np.linspace(0.0, T_F, T_STEPS, endpoint=True)

results = []

for dip_strength in DIP_STRENGTH_VALUES:
    print(f"  DIP_STRENGTH = {dip_strength} D ...", flush=True)
    dipole = (dip_strength / P_A) * np.array(
        [np.cos(THETA), np.sin(THETA), 0]
    )

    coupling = mf.compute_real_space_coupling(
        fr0_n_ks=fr0_n_ks,
        dipole_moment=dipole,
        omega_ks_n=omega_ks_n,
        ks=ks,
        rs=rs_finite,
    )[BAND_IDX, :]
    idxs = np.arange(len(coupling))

    H, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=omega_0,
        gr0s_rs=[(list(coupling), list(idxs))],
        pml_dampings=pml_dampings,
    )

    gamma_0 = mf.free_spontaneous_emission(
        omega_0=omega_0, dip_strength=dip_strength, a=LAT_PARAM
    )
    _t_conv = LAT_PARAM / C_LIGHT * gamma_0
    t_f_internal = T_F / _t_conv

    psi_0 = np.zeros(H.shape[0], dtype=complex)
    psi_0[0] = 1
    psi_t = mf.evolve_diagonalisation(
        H, psi_0, np.linspace(0.0, t_f_internal, T_STEPS)
    )
    results.append(np.abs(psi_t[:, 0]) ** 2)

# ------------------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------------------

cmap = plt.get_cmap("viridis")
sweep_colors = [
    cmap(i / (len(DIP_STRENGTH_VALUES) - 1))
    for i in range(len(DIP_STRENGTH_VALUES))
]

fig, ax = plt.subplots()

for dip_strength, color, pop in zip(DIP_STRENGTH_VALUES, sweep_colors, results):
    ax.plot(ts_norm, pop, color=color, label=rf"${dip_strength}$ D")

ax.set(
    xlabel=r"$t\Gamma_0$",
    ylabel=r"$|C_e(t)|^2$",
    yscale="log",
    ylim=(1e-10, 1.5),
    title=rf"$\theta = {np.rad2deg(THETA):.0f}$°",
)
ax.legend(fontsize=7, ncol=2)

fig.suptitle(
    rf"$p$ sweep, $N_\mathrm{{cells}} = {N_CELLS}$, "
    rf"$N_\mathrm{{PML}} = {N_PML}$, "
    rf"$\Gamma_\mathrm{{max}} = {GAMMA_MAX:.0e}$ $c/a$"
)
out_path = (
    f"../figures/miscellaneous/dipole_strength_sweep_"
    f"{DIP_STRENGTH_VALUES[0]}_to_{DIP_STRENGTH_VALUES[-1]}_D_omegaX.png"
)
fig.savefig(out_path)
plt.close(fig)
print(f"Saved {out_path}")
