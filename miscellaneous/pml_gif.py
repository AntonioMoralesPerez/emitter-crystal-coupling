"""
Creates a GIF comparing bath excitation with and without PMLs for 90° and 135°
polarizations. Used to verify that PML absorbing boundaries suppress reflections.

Layout: 2 rows (with PML / no PML) × 2 columns (90° / 135°).
Also saves a static plot of |C_e(t)|² for both cases to check for revivals.
"""

import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pythtb as tb
from scipy.sparse.linalg import expm_multiply

import src as mf

# ------------------------------------------------------------------------------
# Parameters (mirror kimble.py test values)
# ------------------------------------------------------------------------------
LAT_PARAM: float = 300e-9
BAND_IDX: int = 0
N_K: int = 64
R_0: list = [0.0, 0.0]

THETA_1: float = 90 * np.pi / 180
THETA_2: float = 135 * np.pi / 180
DIP_STRENGTH: float = 50  # [Debye]
DIP_STRENGTH_NORM: float = DIP_STRENGTH / mf.unit_dip(a=LAT_PARAM)
THETAS: list = [THETA_1, THETA_2]
DIPOLES: list = [
    DIP_STRENGTH_NORM * np.array([np.cos(t), np.sin(t), 0]) for t in THETAS
]
DIP_COLORS: list = ["#1f77b4", "#ff7f0e"]
ROW_LABELS: list = ["PML", "no PML"]

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 41
N_PML: int = 8
GAMMA_MAX: float = 0.1  # [c/a]
J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

T_0: float = 0.0
T_F: float = 1e-5  # [1/Γ₀]
T_STEPS: int = 100
N_EMITTERS: int = 1

# ------------------------------------------------------------------------------
# Load field data
# ------------------------------------------------------------------------------
fields_data = mf.load(f"./data/kimble/fields/{N_K}_meshgrid_at_{R_0}.pkl")
omega_ks_n = np.asarray(fields_data["freqs"])
ks = np.asarray(fields_data["kpts"])
fr0_n_ks = np.asarray(fields_data["Eks"])[0]

# ------------------------------------------------------------------------------
# Build tight-binding model and get omega_0
# ------------------------------------------------------------------------------
infinite_model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
infinite_model.set_onsite([J_0])
for vec in [[1, 0], [0, 1]]:
    infinite_model.set_hop(J_1, 0, 0, vec)
for vec in [[1, 1], [1, -1]]:
    infinite_model.set_hop(J_2, 0, 0, vec)

k_vec, k_dist, k_node = infinite_model.k_path(
    [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]], 100, report=False
)
tb_evals = infinite_model.solve_all(k_vec)
omega_0 = mf.safe_sqrt(tb_evals[BAND_IDX][np.isclose(k_dist, k_node[1])]).item()

# ------------------------------------------------------------------------------
# Build finite model and grid
# ------------------------------------------------------------------------------
finite_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = finite_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)
xs_f, ys_f, rs_f = mf.build_finite_grid(N_CELLS)


def compute_pml_dampings(rs, n_cells, n_pml, gamma_max):
    half = n_cells // 2
    dist = np.minimum(half - np.abs(rs[:, 0]), half - np.abs(rs[:, 1]))
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


pml_dampings = compute_pml_dampings(rs_f, N_CELLS, N_PML, GAMMA_MAX)

# ------------------------------------------------------------------------------
# Convert T_F from 1/Γ₀ to a/c
# ------------------------------------------------------------------------------
gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / mf.constants.C_LIGHT * gamma_0
t_0 = T_0 / _t_conv
t_f = T_F / _t_conv
ts_norm = np.linspace(T_0, T_F, T_STEPS, endpoint=True)

# ------------------------------------------------------------------------------
# Compute couplings (shared by both runs)
# ------------------------------------------------------------------------------
gr0_rs_list = [
    mf.compute_real_space_coupling(
        fr0_n_ks=fr0_n_ks,
        dipole_moment=dip,
        omega_ks_n=omega_ks_n,
        ks=ks,
        rs=rs_f,
    )[BAND_IDX]
    for dip in DIPOLES
]
idxs = list(np.arange(len(gr0_rs_list[0])))


def run_evolution(dampings):
    """Return psi_t for each polarization given pml dampings (or None)."""
    results = []
    for gr0_rs in gr0_rs_list:
        H, _ = mf.make_light_matter_Hamiltonian(
            tb_model=finite_model,
            omega_0s=omega_0,
            gr0s_rs=[[list(gr0_rs), idxs]],
            pml_dampings=dampings,
        )
        psi_0 = np.zeros(H.shape[0])
        psi_0[0] = 1
        results.append(expm_multiply(-1j * H, psi_0, t_0, t_f, T_STEPS, endpoint=True))
    return results


print("Running with PML...")
psi_ts_pml = run_evolution(pml_dampings)
print("Running without PML...")
psi_ts_nopml = run_evolution(None)


# ------------------------------------------------------------------------------
# Pre-compute bath populations: shape [2 cases][2 pols][T_STEPS]
# ------------------------------------------------------------------------------
def bath_populations(psi_ts):
    return [
        [
            (np.abs(psi_t[i, N_EMITTERS:]) ** 2).reshape(N_CELLS, N_CELLS)
            for i in range(T_STEPS)
        ]
        for psi_t in psi_ts
    ]


all_pops = [bath_populations(psi_ts_pml), bath_populations(psi_ts_nopml)]

vmax = max(
    all_pops[case][pol][t].max()
    for case in range(2)
    for pol in range(2)
    for t in range(T_STEPS)
)
vmin_log = min(
    p[p > 0].min()
    for case_pops in all_pops
    for pol_pops in case_pops
    for p in pol_pops
    if p[p > 0].size > 0
)
norm = mcolors.LogNorm(vmin=vmin_log, vmax=vmax)

# ------------------------------------------------------------------------------
# Static plot: emitter population |C_e(t)|² — look for revivals in no-PML case
# ------------------------------------------------------------------------------
fig_ce, axs_ce = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

for i, (ax, theta) in enumerate(zip(axs_ce, THETAS)):
    ce_pml = np.abs(psi_ts_pml[i][:, 0]) ** 2
    ce_nopml = np.abs(psi_ts_nopml[i][:, 0]) ** 2
    ax.plot(ts_norm, ce_pml, color=DIP_COLORS[i], label="PML")
    ax.plot(ts_norm, ce_nopml, color=DIP_COLORS[i], ls="--", alpha=0.6, label="no PML")
    ax.set(
        xlabel=r"$t\,\Gamma_0$",
        ylabel=r"$|C_e(t)|^2$",
        title=rf"$\theta = {np.rad2deg(theta):.0f}°$",
        yscale="log",
    )
    ax.title.set_color(DIP_COLORS[i])
    ax.legend()

fig_ce.tight_layout()
fig_ce.savefig(f"./figures/miscellaneous/pml_emitter_population_{R_0}.png", dpi=150)
print("Saved emitter population plot.")

# ------------------------------------------------------------------------------
# GIF: 2 rows (PML / no PML) × 2 columns (90° / 135°)
# ------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

meshes = []
titles = []
for row, (pops, row_label) in enumerate(zip(all_pops, ROW_LABELS)):
    for col, (ax, theta) in enumerate(zip(axs[row], THETAS)):
        sc = ax.pcolormesh(xs_f, ys_f, pops[col][0], norm=norm)
        fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{R}(t)|^2$")
        mf.draw_arrow(
            ax, R_0, theta, color=DIP_COLORS[col], length=14, arrow_offset=1.4
        )
        pml_boundary = N_CELLS // 2 - N_PML + 0.5
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (-pml_boundary, -pml_boundary),
                2 * pml_boundary,
                2 * pml_boundary,
                fill=False,
                edgecolor="white",
                linewidth=1.5,
                linestyle="--",
            )
        )
        ax.set(xlabel=r"$x/a$", ylabel=r"$y/a$", aspect="equal")
        title = ax.set_title(
            rf"{row_label} — $\theta = {np.rad2deg(theta):.0f}°$, "
            rf"$t\,\Gamma_0 = {ts_norm[0]:.2e}$"
        )
        title.set_color(DIP_COLORS[col])
        meshes.append(sc)
        titles.append(title)

fig.tight_layout()


def update(frame):
    for idx, (row, col) in enumerate([(r, c) for r in range(2) for c in range(2)]):
        meshes[idx].set_array(all_pops[row][col][frame].ravel())
        titles[idx].set_text(
            rf"{ROW_LABELS[row]} — $\theta = {np.rad2deg(THETAS[col]):.0f}°$, "
            rf"$t\,\Gamma_0 = {ts_norm[frame]:.2e}$"
        )
    return meshes + titles


anim = animation.FuncAnimation(fig, update, frames=T_STEPS, interval=80, blit=True)
gif_path = f"./figures/miscellaneous/pml_bath_evolution_{R_0}.gif"
anim.save(gif_path, writer="pillow", dpi=100)
print(f"Saved GIF to {gif_path}")
