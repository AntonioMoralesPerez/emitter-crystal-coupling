"""
Step B: NN tight-binding bath (√Θ Hamiltonian) with full MPB real-space coupling.

Intermediate step in the progression kimble_toy.py → kimble.py. Replaces the
local coupling G·δ(R, r_0) from Step A with the full position-dependent coupling
g_R(r_0) computed from MPB field data, while keeping the NN-only tight-binding
bath. Two dipole orientations are included since the coupling is now anisotropic.

Comparison ladder
-----------------
kimble_toy.py  →  Step A  →  Step B (this)  →  kimble.py
  Θ bath          √Θ bath    √Θ bath           √Θ bath
  local G         local G    MPB g_R           MPB g_R
  no Γ_0 norm     Γ_0 norm   Γ_0 norm          Γ_0 norm
  NN hops         NN hops    NN hops           2nd-NN hops
"""

# Standard library
from scipy.sparse.linalg import expm_multiply

# Third-party
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pythtb as tb

# Local
import src as mf


# ------------------------------------------------------------------------------
# Matplotlib configuration
# ------------------------------------------------------------------------------


def configure_matplotlib() -> None:
    """Configure Matplotlib rcParams for journal-quality figures."""
    mm_to_inches = 1 / 25.4
    mpl.rcParams.update(
        {
            "figure.figsize": (85 * mm_to_inches, 42.5 * mm_to_inches),
            "figure.dpi": 300,
            "font.family": "serif",
            "font.size": 13,
            "mathtext.fontset": "stix",
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.linewidth": 1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


configure_matplotlib()

# ------------------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------------------

# MPB field data source (same files as kimble.py)
LAT_PARAM: float = 300e-9  # lattice parameter [m]
N_BANDS: int = 2
BAND_IDX: int = 0
N_K: int = 64

# Emitter parameters
THETA_1: float = 90 * np.pi / 180  # [rad]
THETA_2: float = 135 * np.pi / 180  # [rad]
DIP_STRENGTH: float = 7000  # dipole moment [Debye]
P_A: float = mf.unit_dip(a=LAT_PARAM)
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A
N_EMITTERS: int = 1
R_0: list = [0.0, 0.0]

THETAS: list = [THETA_1, THETA_2]
DIPOLES: list = [
    DIP_STRENGTH_NORM * np.array([np.cos(theta), np.sin(theta), 0]) for theta in THETAS
]
DIP_COLORS: list = ["#1f77b4", "#ff7f0e"]

# Tight-binding model — nearest-neighbour only (no J_2)
LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 41
N_PML: int = 8
GAMMA_MAX: float = 3e-2  # max PML damping [c/a]
J_0: float = 0.07755639  # on-site energy [(c/a)²]
J_1: float = -0.01993698  # NN hopping [(c/a)²]

# Brillouin zone path
K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
K_LABELS: list = [r"$\Gamma$", r"X", r"M", r"$\Gamma$"]
N_K_PATH: int = 100

# Real-space grid (visualization only; full finite grid used for Hamiltonian)
N_R: int = 13

# Time evolution [units: 1/Γ₀]
T_0: float = 0
T_F: float = 3e-2
T_STEPS: int = 200
SNAP_IDX: int = 100

# Probe emitter
NEW_N_EMITTERS: int = 2
NEW_R: list = [-5, 5]
NEW_THETA: float = np.deg2rad(45)
NEW_DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array(
    [np.cos(NEW_THETA), np.sin(NEW_THETA), 0]
)
NEW_DIP_COLOR: str = "#2ca02c"
NEW_T_F: float = T_F  # [1/Γ₀]
NEW_SNAP_IDX: int = 50

# Colors
MY_RED: str = "#d62728"

# ------------------------------------------------------------------------------
# NN tight-binding band structure
# ------------------------------------------------------------------------------


def build_tight_binding_model() -> tb.tb_model:
    """Build the NN infinite tight-binding model (J_0, J_1 only; no J_2)."""
    model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
    model.set_onsite([J_0])
    for vec in np.array([[1, 0], [0, 1]]):
        model.set_hop(J_1, 0, 0, vec)
    return model


def compute_detuning_frequency(
    tb_evals: np.ndarray,
    k_dist: np.ndarray,
    k_node: np.ndarray,
) -> float:
    """Extract the detuning frequency ω_0 = √ε_X at the X point."""
    x_point_mask = np.isclose(k_dist, k_node[1])
    if not np.any(x_point_mask):
        raise ValueError("X point not found in k_dist — check K_PATH and N_K_PATH.")
    return mf.safe_sqrt(tb_evals[BAND_IDX][x_point_mask]).item()


def plot_band_structure(
    k_dist: np.ndarray,
    k_node: np.ndarray,
    tb_evals: np.ndarray,
    omega_0: float,
) -> None:
    """Plot the NN TB band structure (ω_k = √ε_k) and mark ω_0."""
    fig, ax = plt.subplots()
    for tb_eval in tb_evals:
        ax.plot(k_dist, mf.safe_sqrt(tb_eval), color="k")
    ax.hlines(
        omega_0,
        xmin=k_dist[0],
        xmax=k_dist[-1],
        color=MY_RED,
        linestyle="--",
        alpha=0.8,
        label=r"$\omega_0$",
    )
    ax.legend()
    ax.set(
        xlim=(k_node[0], k_node[-1]),
        xticks=k_node,
        xticklabels=K_LABELS,
        ylabel=r"$\omega_\mathbf{k} \, [c/a]$",
        title="NN TB band structure (√Θ)",
    )
    ax.grid(axis="x")
    fig.savefig("./figures/kimble_step_b/TB_bandstructure.svg")
    plt.close(fig)


def compute_and_plot_isofrequency(model: tb.tb_model, omega_0: float) -> None:
    """Plot the isofrequency contour ω_k = ω_0 for the NN TB band."""
    k_x = np.linspace(-0.5, 0.5, N_K_PATH)
    k_y = np.linspace(-0.5, 0.5, N_K_PATH)
    k_x_grid, k_y_grid = np.meshgrid(k_x, k_y, indexing="ij")

    ks_grid_flat = np.column_stack((k_x_grid.ravel(), k_y_grid.ravel()))
    tb_evals_grid = np.array([model.solve_one(k)[BAND_IDX] for k in ks_grid_flat])
    omega_grid = mf.safe_sqrt(tb_evals_grid).reshape(N_K_PATH, N_K_PATH)

    fig, ax = plt.subplots()
    c = ax.contourf(k_x_grid, k_y_grid, omega_grid, levels=12)
    cbar = fig.colorbar(c, ax=ax, label=r"$\omega_\mathbf{k} \, [c/a]$")
    cbar.formatter.set_powerlimits((0, 0))
    ax.contour(
        k_x_grid, k_y_grid, omega_grid, levels=[omega_0], colors=MY_RED, linewidths=2
    )
    ax.set(xlabel=r"$k_x a/2\pi$", ylabel=r"$k_y a/2\pi$", aspect="equal")
    fig.savefig("./figures/kimble_step_b/tb_isofrequency_contour.svg")
    plt.close(fig)


infinite_model = build_tight_binding_model()
k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals = infinite_model.solve_all(k_vec)
omega_0 = compute_detuning_frequency(tb_evals, k_dist, k_node)
print(f"Detuning frequency at X point (NN): omega_0 = {omega_0:.6f} c/a")

plot_band_structure(k_dist, k_node, tb_evals, omega_0)
compute_and_plot_isofrequency(infinite_model, omega_0)

# ------------------------------------------------------------------------------
# Real-space emitter-bath coupling g_R(r_0) — visualization on small grid
# ------------------------------------------------------------------------------


def load_field_data(n_k: int, r_0: list) -> dict:
    """Load precomputed MPB fields, frequencies, and k-points from disk."""
    return mf.load(f"./data/kimble/fields/{n_k}_meshgrid_at_{r_0}.pkl")


def compute_coupling_grid(
    fr0_n_ks: np.ndarray,
    omega_ks_n: np.ndarray,
    ks: np.ndarray,
    n_r: int,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Compute g_R(r_0) on a small square grid for visualization."""
    xs = np.arange(-(n_r // 2), n_r // 2 + 1)
    ys = np.arange(-(n_r // 2), n_r // 2 + 1)
    rs = mf.meshgrid_points(xs, ys, indexing="xy")

    gr0_rs_list = [
        mf.compute_real_space_coupling(
            fr0_n_ks=fr0_n_ks,
            dipole_moment=dipole,
            omega_ks_n=omega_ks_n,
            ks=ks,
            rs=rs,
        )[BAND_IDX, :].reshape(n_r, n_r)
        for dipole in DIPOLES
    ]
    return gr0_rs_list, xs, ys


def plot_real_space_coupling(
    gr0_rs_list: list,
    xs: np.ndarray,
    ys: np.ndarray,
    r_0: list,
) -> None:
    """Plot |g_R(r_0)| for each dipole orientation."""
    cbar_min = np.abs(gr0_rs_list).min()
    cbar_max = np.abs(gr0_rs_list).max()

    for i, theta in enumerate(THETAS, start=1):
        fig, ax = plt.subplots()
        sc = ax.pcolormesh(
            xs,
            ys,
            np.abs(gr0_rs_list[i - 1]),
            norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max),
        )
        fig.colorbar(sc, ax=ax, label=r"$|g_\mathbf{R}(\mathbf{r}_0)| \, [c/a]$")
        mf.draw_arrow(
            ax, r_0, theta, color=DIP_COLORS[i - 1], length=4, arrow_offset=0.4
        )
        ax.set(
            title=rf"$\theta_{i} = {np.rad2deg(theta):.0f}$°",
            xlabel=r"$x/a$",
            ylabel=r"$y/a$",
            aspect="equal",
        )
        ax.title.set_color(DIP_COLORS[i - 1])
        fig.savefig(
            f"./figures/kimble_step_b/g_r({r_0})_for_theta_{np.rad2deg(theta):.0f}deg.svg"
        )
        plt.close(fig)


fields_data = load_field_data(N_K, R_0)
omega_ks_n = np.asarray(fields_data["freqs"])  # (Nk, Nb)
ks = np.asarray(fields_data["kpts"])  # (Nk, 3)
fr0_n_ks = np.asarray(fields_data["Eks"])[0]  # (Nb, Nk, 3)

gr0_rs_list, xs_small, ys_small = compute_coupling_grid(fr0_n_ks, omega_ks_n, ks, N_R)
plot_real_space_coupling(gr0_rs_list, xs_small, ys_small, R_0)

# ------------------------------------------------------------------------------
# Finite tight-binding model and PML
# ------------------------------------------------------------------------------


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    """Quadratic PML damping profile in c/a."""
    half = n_cells // 2
    dist_x = half - np.abs(rs[:, 0])
    dist_y = half - np.abs(rs[:, 1])
    dist = np.minimum(dist_x, dist_y)
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


def build_finite_model(infinite_model: tb.tb_model) -> tb.tb_model:
    """Cut the infinite model to N_CELLS × N_CELLS."""
    temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
    return temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)


finite_model = build_finite_model(infinite_model)
xs_finite, ys_finite, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)

# ------------------------------------------------------------------------------
# Γ₀ and time conversion
# ------------------------------------------------------------------------------

gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / mf.constants.C_LIGHT * gamma_0
T_0 /= _t_conv
T_F /= _t_conv
NEW_T_F /= _t_conv
ts = np.linspace(T_0, T_F, T_STEPS, endpoint=True)
ts_norm = _t_conv * ts

print(f"\nΓ₀  = {gamma_0:.4e} s⁻¹")
print(f"T_F = {T_F:.1f} a/c  ({T_F * _t_conv:.2e} × Γ₀⁻¹)")

# ------------------------------------------------------------------------------
# Single-emitter time evolution
# ------------------------------------------------------------------------------


def compute_time_evolution(
    finite_model: tb.tb_model,
    omega_ks_n: np.ndarray,
    fr0_n_ks: np.ndarray,
    ks: np.ndarray,
    rs: np.ndarray,
    omega_0: float,
    t_0: float,
    t_f: float,
    t_steps: int,
    pml_dampings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Build H = [ω_0, g†; g, √Θ] with full MPB coupling and time-evolve both dipoles.

    Returns
    -------
    psi_t_1, psi_t_2 : np.ndarray, shape (t_steps, 1 + n_orbitals)
    couplings : list of np.ndarray, each shape (n_orbitals,)
    """
    gr0_n_rs_list = [
        mf.compute_real_space_coupling(
            fr0_n_ks=fr0_n_ks,
            dipole_moment=dipole,
            omega_ks_n=omega_ks_n,
            ks=ks,
            rs=rs,
        )
        for dipole in DIPOLES
    ]
    couplings = [gr0_n_rs[BAND_IDX, :] for gr0_n_rs in gr0_n_rs_list]

    idxs = np.arange(len(couplings[0]))
    interaction_1 = [[list(couplings[0]), list(idxs)]]
    interaction_2 = [[list(couplings[1]), list(idxs)]]

    H_1, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=omega_0,
        gr0s_rs=interaction_1,
        pml_dampings=pml_dampings,
    )
    H_2, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=omega_0,
        gr0s_rs=interaction_2,
        pml_dampings=pml_dampings,
    )

    psi_0 = np.zeros(H_1.shape[0])
    psi_0[0] = 1

    psi_t_1 = expm_multiply(-1j * H_1, psi_0, t_0, t_f, t_steps, endpoint=True)
    psi_t_2 = expm_multiply(-1j * H_2, psi_0, t_0, t_f, t_steps, endpoint=True)

    return psi_t_1, psi_t_2, couplings


def plot_emitter_time_evolution(
    ts: np.ndarray,
    psi_t_1: np.ndarray,
    psi_t_2: np.ndarray,
    r_0: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot emitter population and entanglement entropy for both dipole orientations."""
    mod_Ce_sq_t_1 = np.abs(psi_t_1[:, 0]) ** 2
    mod_Ce_sq_t_2 = np.abs(psi_t_2[:, 0]) ** 2

    S_t_1 = mf.binary_entropy(mod_Ce_sq_t_1)
    S_t_2 = mf.binary_entropy(mod_Ce_sq_t_2)

    fig, ax = plt.subplots()
    (l1,) = ax.plot(
        ts,
        mod_Ce_sq_t_1,
        lw=3,
        color=DIP_COLORS[0],
        label=r"$|C_e(t)|^2$, $\theta_1 = 90$°",
    )
    (l2,) = ax.plot(
        ts,
        mod_Ce_sq_t_2,
        lw=2,
        color=DIP_COLORS[1],
        alpha=0.9,
        label=r"$|C_e(t)|^2$, $\theta_2 = 135$°",
    )
    ax.set(xlabel=r"$t \Gamma_0$", ylabel=r"$|C_e(t)|^2$", yscale="log")

    ax2 = ax.twinx()
    (l3,) = ax2.plot(
        ts,
        S_t_1,
        ls="--",
        color=DIP_COLORS[0],
        lw=3,
        label=r"$S(t)$, $\theta_1 = 90$°",
    )
    (l4,) = ax2.plot(
        ts,
        S_t_2,
        ls="--",
        color=DIP_COLORS[1],
        alpha=0.9,
        lw=2,
        label=r"$S(t)$, $\theta_2 = 135$°",
    )
    ax2.set(ylabel=r"$S(t)$")

    lines = [l1, l4]
    ax.legend(lines, [line.get_label() for line in lines])

    fig.savefig(f"./figures/kimble_step_b/emitter_timeevolution_entropy_at_{r_0}.svg")
    plt.close(fig)
    return S_t_1, S_t_2


def plot_bath_snapshot(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    psi_t_1: np.ndarray,
    psi_t_2: np.ndarray,
    snap_idx: int,
    n_emitters: int,
    r_0: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot bath population snapshots for both dipole orientations."""
    mod_Cr_sq_1 = np.abs(psi_t_1[snap_idx, n_emitters:]) ** 2
    mod_Cr_sq_2 = np.abs(psi_t_2[snap_idx, n_emitters:]) ** 2

    z1 = mod_Cr_sq_1.reshape(len(pos_x), len(pos_y))
    z2 = mod_Cr_sq_2.reshape(len(pos_x), len(pos_y))
    cbar_max = max(z1.max(), z2.max())
    exp = int(np.floor(np.log10(cbar_max))) if cbar_max > 0 else 0

    for i, (theta, z) in enumerate(zip(THETAS, [z1, z2])):
        fig, ax = plt.subplots()
        sc = ax.pcolormesh(pos_x, pos_y, z, vmin=0, vmax=cbar_max)
        mf.draw_arrow(
            ax, r_0, theta, color=DIP_COLORS[i], length=17.5, arrow_offset=1.75
        )
        pml_boundary = N_CELLS // 2 - N_PML + 0.5
        ax.add_patch(
            mpl.patches.Rectangle(
                (-pml_boundary, -pml_boundary),
                2 * pml_boundary,
                2 * pml_boundary,
                fill=False,
                edgecolor="white",
                linewidth=1.5,
                linestyle="--",
            )
        )
        cbar = fig.colorbar(
            sc,
            ax=ax,
            label=rf"$|C_\mathbf{{R}}(t)|^2 \; [\times 10^{{{exp}}}]$",
            pad=0.1,
        )
        cbar.formatter = mpl.ticker.FuncFormatter(lambda x, _: f"{x / 10**exp:.1f}")
        cbar.update_ticks()
        ax.set(
            xlabel=r"$x/a$",
            ylabel=r"$y/a$",
            title=rf"$\mathbf{{r}}_0/a = {r_0}$",
            aspect="equal",
        )
        fig.savefig(
            f"./figures/kimble_step_b/bath_snapshot_at_{r_0}_theta_{np.rad2deg(theta):.0f}deg.svg"
        )
        plt.close(fig)

    return z1, z2


psi_t_1, psi_t_2, couplings = compute_time_evolution(
    finite_model=finite_model,
    omega_ks_n=omega_ks_n,
    fr0_n_ks=fr0_n_ks,
    ks=ks,
    rs=rs_finite,
    omega_0=omega_0,
    t_0=T_0,
    t_f=T_F,
    t_steps=T_STEPS,
    pml_dampings=pml_dampings,
)

S_t_1, S_t_2 = plot_emitter_time_evolution(ts_norm, psi_t_1, psi_t_2, R_0)

# NOTE: _orb and _lat are private PythTB attributes with no public equivalent.
positions = mf.find_positions(finite_model._orb, finite_model._lat)
pos_x = np.unique(positions[:, 0]) - N_CELLS // 2
pos_y = np.unique(positions[:, 1]) - N_CELLS // 2

z1, z2 = plot_bath_snapshot(pos_x, pos_y, psi_t_1, psi_t_2, SNAP_IDX, N_EMITTERS, R_0)

# ------------------------------------------------------------------------------
# Probe emitter: bath-mediated excitation transfer and concurrence
# ------------------------------------------------------------------------------


def compute_probe_emitter_evolution(
    finite_model: tb.tb_model,
    couplings: list,
    rs: np.ndarray,
    omega_0: float,
    t_0: float,
    t_f: float,
    t_steps: int,
    pml_dampings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-evolve the two-emitter system using full MPB coupling for both emitters.

    Loads probe field data at NEW_R, computes NEW_DIPOLE coupling over the full
    finite grid, then builds two two-emitter Hamiltonians (one per main dipole
    orientation). Initial state: excited emitter only.

    Returns
    -------
    new_psi_t_1, new_psi_t_2 : np.ndarray, shape (t_steps, 2 + n_orbitals)
        Index 0 = excited emitter, index 1 = probe emitter.
    """
    probe_fields_data = load_field_data(N_K, NEW_R)
    new_omega_ks_n = np.asarray(probe_fields_data["freqs"])
    new_fr0_n_ks = np.asarray(probe_fields_data["Eks"])[0]
    new_ks = np.asarray(probe_fields_data["kpts"])

    new_gr0_n_rs = mf.compute_real_space_coupling(
        fr0_n_ks=new_fr0_n_ks,
        dipole_moment=NEW_DIPOLE,
        omega_ks_n=new_omega_ks_n,
        ks=new_ks,
        rs=rs,
    )
    new_coupling = new_gr0_n_rs[BAND_IDX, :]
    idxs = np.arange(len(couplings[0]))

    new_interaction_1 = [
        [list(couplings[0]), list(idxs)],
        [list(new_coupling), list(idxs)],
    ]
    new_interaction_2 = [
        [list(couplings[1]), list(idxs)],
        [list(new_coupling), list(idxs)],
    ]

    new_H_1, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=[omega_0, omega_0],
        gr0s_rs=new_interaction_1,
        pml_dampings=pml_dampings,
    )
    new_H_2, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=[omega_0, omega_0],
        gr0s_rs=new_interaction_2,
        pml_dampings=pml_dampings,
    )

    new_psi_0 = np.zeros(new_H_1.shape[0])
    new_psi_0[0] = 1

    new_psi_t_1 = expm_multiply(
        -1j * new_H_1, new_psi_0, t_0, t_f, t_steps, endpoint=True
    )
    new_psi_t_2 = expm_multiply(
        -1j * new_H_2, new_psi_0, t_0, t_f, t_steps, endpoint=True
    )

    return new_psi_t_1, new_psi_t_2


def plot_probe_bath_snapshot(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    new_psi_t_1: np.ndarray,
    new_psi_t_2: np.ndarray,
    r_0: list,
) -> None:
    """Plot bath population snapshots for the two-emitter system."""
    new_mod_Cr_sq_1 = np.abs(new_psi_t_1[NEW_SNAP_IDX, NEW_N_EMITTERS:]) ** 2
    new_mod_Cr_sq_2 = np.abs(new_psi_t_2[NEW_SNAP_IDX, NEW_N_EMITTERS:]) ** 2

    new_z1 = new_mod_Cr_sq_1.reshape(len(pos_x), len(pos_y))
    new_z2 = new_mod_Cr_sq_2.reshape(len(pos_x), len(pos_y))
    cbar_max = max(new_z1.max(), new_z2.max())
    exp = int(np.floor(np.log10(cbar_max))) if cbar_max > 0 else 0

    for i, (theta, z) in enumerate(zip(THETAS, [new_z1, new_z2]), start=1):
        fig, ax = plt.subplots()
        sc = ax.pcolormesh(pos_x, pos_y, z, vmin=0, vmax=cbar_max)
        mf.draw_arrow(ax, NEW_R, NEW_THETA, NEW_DIP_COLOR, length=5, arrow_offset=0.5)
        mf.draw_arrow(ax, r_0, theta, DIP_COLORS[i - 1], length=5, arrow_offset=0.5)
        pml_boundary = N_CELLS // 2 - N_PML + 0.5
        ax.add_patch(
            mpl.patches.Rectangle(
                (-pml_boundary, -pml_boundary),
                2 * pml_boundary,
                2 * pml_boundary,
                fill=False,
                edgecolor="white",
                linewidth=1.5,
                linestyle="--",
            )
        )
        cbar = fig.colorbar(
            sc,
            ax=ax,
            label=rf"$|C_\mathbf{{r}}(t)|^2 \; [\times 10^{{{exp}}}]$",
        )
        cbar.formatter = mpl.ticker.FuncFormatter(lambda x, _: f"{x / 10**exp:.1f}")
        cbar.update_ticks()
        ax.set(
            xlabel=r"$x/a$",
            ylabel=r"$y/a$",
            aspect="equal",
            title=rf"$\theta_{i} = {np.rad2deg(theta):.0f}$°",
            xlim=(-8, 8),
            ylim=(-8, 8),
        )
        ax.title.set_color(DIP_COLORS[i - 1])
        fig.savefig(
            f"./figures/kimble_step_b/bath_snapshot_at_{r_0}_theta_{np.rad2deg(theta):.0f}deg_with_probe.svg"
        )
        plt.close(fig)


def plot_probe_time_evolution(
    ts: np.ndarray,
    new_psi_t_1: np.ndarray,
    new_psi_t_2: np.ndarray,
    r_0: list,
) -> None:
    """Plot emitter populations and concurrence for the two-emitter system."""
    mod_Ce_excited_sq = [
        np.abs(new_psi_t_1[:, 0]) ** 2,
        np.abs(new_psi_t_2[:, 0]) ** 2,
    ]
    mod_Ce_probe_sq = [
        np.abs(new_psi_t_1[:, 1]) ** 2,
        np.abs(new_psi_t_2[:, 1]) ** 2,
    ]
    C_ts = [
        2 * np.abs(mod_Ce_excited_sq[0] * np.conj(mod_Ce_probe_sq[0])),
        2 * np.abs(mod_Ce_excited_sq[1] * np.conj(mod_Ce_probe_sq[1])),
    ]
    C_max = max(C_ts[0].max(), C_ts[1].max())

    for i, theta in enumerate(THETAS):
        fig, ax = plt.subplots()
        ax.plot(ts, mod_Ce_excited_sq[i], c=DIP_COLORS[i])
        ax.plot(ts, mod_Ce_probe_sq[i], c=NEW_DIP_COLOR)

        ax2 = ax.twinx()
        ax2.plot(ts, C_ts[i], c=MY_RED, ls="--")

        ax.set(
            title=rf"$\theta_p = {np.rad2deg(NEW_THETA):.0f}$°",
            xlabel=r"$t\Gamma_0$",
            ylabel=r"$|C_e(t)|^2$",
            yscale="log",
        )
        ax.title.set_color(NEW_DIP_COLOR)
        ax2.set_ylabel(r"$C(t)$", color=MY_RED)
        ax2.tick_params(axis="y", colors=MY_RED)
        ax2.spines["right"].set_color(MY_RED)
        pad = C_max * 0.05
        ax2.set_ylim((-pad, C_max + pad))
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        fig.savefig(
            f"./figures/kimble_step_b/emitter_timeevolution_concurrence_at_{r_0}"
            f"_theta_{np.rad2deg(theta):.0f}deg_and_{NEW_R}.svg"
        )
        plt.close(fig)


new_ts = np.linspace(T_0, NEW_T_F, T_STEPS, endpoint=True)
new_ts_norm = _t_conv * new_ts

new_psi_t_1, new_psi_t_2 = compute_probe_emitter_evolution(
    finite_model=finite_model,
    couplings=couplings,
    rs=rs_finite,
    omega_0=omega_0,
    t_0=T_0,
    t_f=NEW_T_F,
    t_steps=T_STEPS,
    pml_dampings=pml_dampings,
)

plot_probe_bath_snapshot(pos_x, pos_y, new_psi_t_1, new_psi_t_2, R_0)
plot_probe_time_evolution(new_ts_norm, new_psi_t_1, new_psi_t_2, R_0)
