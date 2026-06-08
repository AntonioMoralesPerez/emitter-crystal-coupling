"""
Step A: NN tight-binding bath (√Θ Hamiltonian) with local emitter coupling.

Intermediate step in the progression kimble_toy.py → kimble.py. Isolates the
effect of replacing the bath Hamiltonian Θ → √Θ (i.e. using real photon
frequencies ω_k = √ε_k) while keeping a purely local coupling G·δ(R, r_0),
with G matched to the peak of the MPB real-space coupling at DIP_STRENGTH = 7000 D.

Comparison ladder
-----------------
kimble_toy.py  →  Step A (this)  →  Step B  →  kimble.py
  Θ bath          √Θ bath           √Θ bath     √Θ bath
  local G         local G           MPB g_R     MPB g_R
  no Γ_0 norm     Γ_0 norm          Γ_0 norm    Γ_0 norm
  NN hops         NN hops           NN hops     2nd-NN hops
"""

# Standard library
import matplotlib as mpl
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker

# Third-party
import numpy as np
import pythtb as tb

# Local
import emitter_crystal_coupling as mf
from emitter_crystal_coupling.constants import C_LIGHT

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
RESOLUTION: int = 64  # must match the precomputed field/epsilon files

# Emitter — single orientation used only to extract G from MPB fields
THETA: float = 90 * np.pi / 180  # [rad]
DIP_STRENGTH: float = 7000  # dipole moment [Debye]
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A
DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array([np.cos(THETA), np.sin(THETA), 0])
N_EMITTERS: int = 1
R_0: list = [0.0, 0.0]

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

# Real-space coupling grid (to extract G)
N_R: int = 13

# Time evolution [units: 1/Γ₀]
T_0: float = 0
T_F: float = 3e-2
T_STEPS: int = 200
SNAP_IDX: int = 100

# Probe emitter — on-lattice site nearest to kimble.py's [-4.5, 4.5]
NEW_N_EMITTERS: int = 2
NEW_R: list = [-5, 5]
NEW_T_F: float = T_F  # [1/Γ₀]
NEW_SNAP_IDX: int = 50

# Colors
MY_BLUE: str = "#1f77b4"
MY_GREEN: str = "#2ca02c"
MY_RED: str = "#d62728"

# ------------------------------------------------------------------------------
# NN tight-binding band structure
# ------------------------------------------------------------------------------


def build_tight_binding_model() -> tb.tb_model:
    """
    Build the NN infinite tight-binding model (J_0, J_1 only; no J_2).

    Returns
    -------
    tb.tb_model
        Tight-binding model with on-site and NN hopping terms.
    """
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
    """
    Extract the detuning frequency ω_0 = √ε_X at the X point.

    Parameters
    ----------
    tb_evals : np.ndarray, shape (Nb, Nk)
    k_dist : np.ndarray, shape (Nk,)
    k_node : np.ndarray

    Returns
    -------
    float
        ω_0 in c/a.
    """
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
    fig.savefig("./figures/kimble_step_a/TB_bandstructure.svg")
    plt.close(fig)


infinite_model = build_tight_binding_model()
k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals: np.ndarray = np.asarray(infinite_model.solve_all(k_vec))
omega_0 = compute_detuning_frequency(tb_evals, k_dist, k_node)
print(f"Detuning frequency at X point (NN): omega_0 = {omega_0:.6f} c/a")

plot_band_structure(k_dist, k_node, tb_evals, omega_0)

# ------------------------------------------------------------------------------
# Isofrequency contour
# ------------------------------------------------------------------------------


def compute_and_plot_isofrequency(model: tb.tb_model, omega_0: float) -> None:
    """Plot the isofrequency contour ω_k = ω_0 for the NN TB band."""
    k_x = np.linspace(-0.5, 0.5, N_K_PATH)
    k_y = np.linspace(-0.5, 0.5, N_K_PATH)
    k_x_grid, k_y_grid = np.meshgrid(k_x, k_y, indexing="ij")

    ks_grid_flat = np.column_stack((k_x_grid.ravel(), k_y_grid.ravel()))
    tb_evals_grid = np.array(
        [np.asarray(model.solve_one(k))[BAND_IDX] for k in ks_grid_flat]
    )
    omega_grid = mf.safe_sqrt(tb_evals_grid).reshape(N_K_PATH, N_K_PATH)

    fig, ax = plt.subplots()
    c = ax.contourf(k_x_grid, k_y_grid, omega_grid, levels=12)
    cbar = fig.colorbar(c, ax=ax, label=r"$\omega_\mathbf{k} \, [c/a]$")
    assert isinstance(cbar.formatter, mpl_ticker.ScalarFormatter)
    cbar.formatter.set_powerlimits((0, 0))
    ax.contour(
        k_x_grid, k_y_grid, omega_grid, levels=[omega_0], colors=MY_RED, linewidths=2
    )
    ax.set(xlabel=r"$k_x a/2\pi$", ylabel=r"$k_y a/2\pi$", aspect="equal")
    fig.savefig("./figures/kimble_step_a/tb_isofrequency_contour.svg")
    plt.close(fig)


compute_and_plot_isofrequency(infinite_model, omega_0)

# ------------------------------------------------------------------------------
# Local coupling G from MPB fields
# ------------------------------------------------------------------------------
# Load the same field data as kimble.py, compute the real-space coupling on a
# small grid, and use the peak value |g_R|_max as the local coupling strength G.
# This is the fairest comparison: G matches the dominant coupling scale that the
# real emitter achieves at DIP_STRENGTH = 7000 D.

fields_data = mf.load(
    f"./data/kimble/fields/res_{RESOLUTION}/nk_{N_K}/meshgrid_at_{R_0}.pkl"
)
omega_ks_n = np.asarray(fields_data["freqs"])  # (Nk, Nb)
ks = np.asarray(fields_data["kpts"])  # (Nk, 3)
fr0_n_ks = np.asarray(fields_data["Eks"])[0]  # (Nb, Nk, 3)

_xs = np.arange(-(N_R // 2), N_R // 2 + 1)
_ys = np.arange(-(N_R // 2), N_R // 2 + 1)
_rs = mf.meshgrid_points(_xs, _ys, indexing="xy")

_g_grid = mf.compute_real_space_coupling(
    fr0_n_ks=fr0_n_ks,
    dipole_moment=DIPOLE,
    omega_ks_n=omega_ks_n,
    ks=ks,
    rs=_rs,
)[BAND_IDX, :].reshape(N_R, N_R)

G: float = float(np.abs(_g_grid).max())
G_PROBE: float = G

print(f"\nLocal coupling G = |g_R|_max at DIP_STRENGTH = {DIP_STRENGTH} D:")
print(f"  G          = {G:.6e} c/a")
print(f"  G / √|J_1| = {G / np.sqrt(abs(J_1)):.4f}  (toy target: 0.1)")

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


def find_site_index(rs: np.ndarray, r: list) -> int:
    """Index of the bath site closest to position r."""
    dists = np.sum((rs - np.array(r, dtype=float)) ** 2, axis=1)
    return int(np.argmin(dists))


finite_model = build_finite_model(infinite_model)
xs_finite, ys_finite, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)

idx_r0 = find_site_index(rs_finite, R_0)
idx_new_r = find_site_index(rs_finite, NEW_R)
print(f"\nEmitter site index: {idx_r0}  →  r = {rs_finite[idx_r0]}")
print(f"Probe site index:   {idx_new_r}  →  r = {rs_finite[idx_new_r]}")

# ------------------------------------------------------------------------------
# Γ₀ and time conversion
# ------------------------------------------------------------------------------

gamma_0: float = float(
    mf.free_spontaneous_emission(
        omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
    )
)
_t_conv = LAT_PARAM / C_LIGHT * gamma_0
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
    rs: np.ndarray,
    omega_0: float,
    g: float,
    r_0: list,
    t_0: float,
    t_f: float,
    t_steps: int,
    pml_dampings: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build H = [ω_0, G; G†, √Θ] with local coupling and time-evolve.

    Parameters
    ----------
    g : float
        Local coupling strength [c/a].
    r_0 : list
        Emitter position; coupled to the single nearest bath site.

    Returns
    -------
    np.ndarray, shape (t_steps, 1 + n_orbitals)
    """
    idx = find_site_index(rs, r_0)
    H, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=omega_0,
        gr0s_rs=[([g], [idx])],
        pml_dampings=pml_dampings,
    )
    psi_0 = np.zeros(H.shape[0], dtype=complex)
    psi_0[0] = 1
    return mf.evolve_diagonalisation(H, psi_0, np.linspace(t_0, t_f, t_steps))


def plot_emitter_time_evolution(
    ts: np.ndarray,
    psi_t: np.ndarray,
    r_0: list,
) -> np.ndarray:
    """Plot emitter population and entanglement entropy."""
    mod_Ce_sq = np.abs(psi_t[:, 0]) ** 2
    S_t = mf.binary_entropy(mod_Ce_sq)

    fig, ax = plt.subplots()
    (l1,) = ax.plot(ts, mod_Ce_sq, lw=2, color=MY_BLUE, label=r"$|C_e(t)|^2$")
    ax.set(xlabel=r"$t\Gamma_0$", ylabel=r"$|C_e(t)|^2$", yscale="log")

    ax2 = ax.twinx()
    (l2,) = ax2.plot(ts, S_t, ls="--", color=MY_BLUE, lw=2, label=r"$S(t)$")
    ax2.set(ylabel=r"$S(t)$")
    ax.legend([l1, l2], [str(l1.get_label()), str(l2.get_label())])

    fig.savefig(f"./figures/kimble_step_a/emitter_timeevolution_entropy_at_{r_0}.svg")
    plt.close(fig)
    return S_t


def plot_bath_snapshot(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    psi_t: np.ndarray,
    snap_idx: int,
    n_emitters: int,
    r_0: list,
) -> np.ndarray:
    """Plot bath population snapshot at snap_idx."""
    mod_Cr_sq = np.abs(psi_t[snap_idx, n_emitters:]) ** 2
    z = mod_Cr_sq.reshape(len(pos_x), len(pos_y))
    cbar_max = z.max()
    exp = int(np.floor(np.log10(cbar_max))) if cbar_max > 0 else 0

    fig, ax = plt.subplots()
    sc = ax.pcolormesh(pos_x, pos_y, z, vmin=0, vmax=cbar_max)
    ax.plot(
        *r_0,
        "o",
        color=MY_BLUE,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    pml_boundary = N_CELLS // 2 - N_PML + 0.5
    ax.add_patch(
        mpl_patches.Rectangle(
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
        sc, ax=ax, label=rf"$|C_\mathbf{{R}}(t)|^2 \; [\times 10^{{{exp}}}]$", pad=0.1
    )
    cbar.formatter = mpl_ticker.FuncFormatter(lambda x, _: f"{x / 10**exp:.1f}")
    cbar.update_ticks()
    ax.set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        title=rf"$\mathbf{{r}}_0/a = {r_0}$, $G = {G:.2e}$ c/a",
        aspect="equal",
    )
    fig.savefig(f"./figures/kimble_step_a/bath_snapshot_at_{r_0}.svg")
    plt.close(fig)
    return z


psi_t = compute_time_evolution(
    finite_model=finite_model,
    rs=rs_finite,
    omega_0=omega_0,
    g=G,
    r_0=R_0,
    t_0=T_0,
    t_f=T_F,
    t_steps=T_STEPS,
    pml_dampings=pml_dampings,
)

S_t = plot_emitter_time_evolution(ts_norm, psi_t, R_0)

# NOTE: _orb and _lat are private PythTB attributes with no public equivalent.
positions = mf.find_positions(finite_model._orb, finite_model._lat)
pos_x = np.unique(positions[:, 0]) - N_CELLS // 2
pos_y = np.unique(positions[:, 1]) - N_CELLS // 2

plot_bath_snapshot(pos_x, pos_y, psi_t, SNAP_IDX, N_EMITTERS, R_0)

# ------------------------------------------------------------------------------
# Probe emitter: bath-mediated excitation transfer and concurrence
# ------------------------------------------------------------------------------


def compute_probe_emitter_evolution(
    finite_model: tb.tb_model,
    rs: np.ndarray,
    omega_0: float,
    g: float,
    r_0: list,
    g_probe: float,
    new_r: list,
    t_0: float,
    t_f: float,
    t_steps: int,
    pml_dampings: np.ndarray | None = None,
) -> np.ndarray:
    """
    Time-evolve the two-emitter system: excited emitter at r_0, probe at new_r.

    Both emitters use local coupling (G·δ). Initial state: excited emitter only.

    Returns
    -------
    np.ndarray, shape (t_steps, 2 + n_orbitals)
        Index 0 = excited emitter, index 1 = probe emitter.
    """
    idx_main = find_site_index(rs, r_0)
    idx_probe = find_site_index(rs, new_r)
    H, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=omega_0,
        gr0s_rs=[([g], [idx_main]), ([g_probe], [idx_probe])],
        pml_dampings=pml_dampings,
    )
    psi_0 = np.zeros(H.shape[0], dtype=complex)
    psi_0[0] = 1
    return mf.evolve_diagonalisation(H, psi_0, np.linspace(t_0, t_f, t_steps))


def plot_probe_bath_snapshot(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    new_psi_t: np.ndarray,
    r_0: list,
) -> None:
    """Plot bath population snapshot for the two-emitter system."""
    new_mod_Cr_sq = np.abs(new_psi_t[NEW_SNAP_IDX, NEW_N_EMITTERS:]) ** 2
    new_z = new_mod_Cr_sq.reshape(len(pos_x), len(pos_y))
    cbar_max = new_z.max()
    exp = int(np.floor(np.log10(cbar_max))) if cbar_max > 0 else 0

    fig, ax = plt.subplots()
    sc = ax.pcolormesh(pos_x, pos_y, new_z, vmin=0, vmax=cbar_max)
    ax.plot(
        *r_0,
        "o",
        color=MY_BLUE,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    ax.plot(
        *NEW_R,
        "o",
        color=MY_GREEN,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    pml_boundary = N_CELLS // 2 - N_PML + 0.5
    ax.add_patch(
        mpl_patches.Rectangle(
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
        sc, ax=ax, label=rf"$|C_\mathbf{{r}}(t)|^2 \; [\times 10^{{{exp}}}]$"
    )
    cbar.formatter = mpl_ticker.FuncFormatter(lambda x, _: f"{x / 10**exp:.1f}")
    cbar.update_ticks()
    ax.set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
        title=rf"$\mathbf{{r}}_0 = {r_0}$, probe at {NEW_R}",
        xlim=(-8, 8),
        ylim=(-8, 8),
    )
    fig.savefig(f"./figures/kimble_step_a/bath_snapshot_at_{r_0}_with_probe.svg")
    plt.close(fig)


def plot_probe_time_evolution(
    ts: np.ndarray,
    new_psi_t: np.ndarray,
    r_0: list,
) -> None:
    """Plot emitter populations and concurrence for the two-emitter system."""
    mod_Ce_excited_sq = np.abs(new_psi_t[:, 0]) ** 2
    mod_Ce_probe_sq = np.abs(new_psi_t[:, 1]) ** 2
    C_t = 2 * np.abs(mod_Ce_excited_sq * np.conj(mod_Ce_probe_sq))
    C_max = C_t.max()

    fig, ax = plt.subplots()
    ax.plot(ts, mod_Ce_excited_sq, c=MY_BLUE, label=rf"$|C_e(t)|^2$, r$_0={r_0}$")
    ax.plot(ts, mod_Ce_probe_sq, c=MY_GREEN, label=rf"$|C_p(t)|^2$, r$_p={NEW_R}$")

    ax2 = ax.twinx()
    ax2.plot(ts, C_t, c=MY_RED, ls="--", label=r"$C(t)$")

    ax.set(xlabel=r"$t\Gamma_0$", ylabel=r"$|C_e(t)|^2$", yscale="log")
    ax.legend(loc="upper right")
    ax2.set_ylabel(r"$C(t)$", color=MY_RED)
    ax2.tick_params(axis="y", colors=MY_RED)
    ax2.spines["right"].set_color(MY_RED)
    pad = C_max * 0.05
    ax2.set_ylim((-pad, C_max + pad))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.savefig(
        f"./figures/kimble_step_a/emitter_timeevolution_concurrence_at_{r_0}_and_{NEW_R}.svg"
    )
    plt.close(fig)


new_ts = np.linspace(T_0, NEW_T_F, T_STEPS, endpoint=True)
new_ts_norm = _t_conv * new_ts

new_psi_t = compute_probe_emitter_evolution(
    finite_model=finite_model,
    rs=rs_finite,
    omega_0=omega_0,
    g=G,
    r_0=R_0,
    g_probe=G_PROBE,
    new_r=NEW_R,
    t_0=T_0,
    t_f=NEW_T_F,
    t_steps=T_STEPS,
    pml_dampings=pml_dampings,
)

plot_probe_bath_snapshot(pos_x, pos_y, new_psi_t, R_0)
plot_probe_time_evolution(new_ts_norm, new_psi_t, R_0)
