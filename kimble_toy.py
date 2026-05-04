"""
Toy model: emitter coupled to a 2D tight-binding bath via a local coupling g.

Studies a single quantum emitter coupled to a 2D tight-binding bath via a
completely local coupling G that can be set manually. The bath is described by
the nearest-neighbor tight-binding Hamiltonian H_b = Θ directly (no square
root). The emitter-bath coupling is:

    g_R(r_0) = G * delta(R, r_0)

i.e. the emitter only talks to the single bath site at its position r_0. The
full Hamiltonian is:

    H = [ omega_0   g^T ]
        [ g         Θ   ]

where omega_0 and G are free parameters in units of |J_1|. Specifically,
this script:

  - Plots the tight-binding band structure and isofrequency contour.
  - Solves the time evolution of the emitter and the photonic bath.
  - Introduces a probe emitter to study bath-mediated excitation transfer
    and quantifies the entanglement between the two emitters via the concurrence.
"""

# Standard library
from scipy.sparse.linalg import expm_multiply  # time-dependent Schrödinger equation
from scipy.sparse import csr_matrix, diags, bmat

# Third-party
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pythtb as tb  # tight-binding model

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
            # Figure
            "figure.figsize": (85 * mm_to_inches, 42.5 * mm_to_inches),
            "figure.dpi": 300,
            # Font
            "font.family": "serif",
            "font.size": 13,
            "mathtext.fontset": "stix",
            # Axes
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.linewidth": 1,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            # Legend
            "legend.fontsize": 11,
            "legend.frameon": False,
            # Save
            "savefig.bbox": "tight",
        }
    )


configure_matplotlib()

# ------------------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------------------

# Tight-binding model
LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]  # orbital position in lattice units
N_CELLS: int = 41  # unit cells per direction
N_PML: int = 8  # absorbing layer thickness [unit cells per edge]
GAMMA_MAX: float = 2  # maximum damping rate at the hard wall [|J_1|]
# Hopping parameters normalized to |J_1| = 1, rotating frame at J_0
J_1: float = -1.0  # NN hopping [|J_1|]
J_0: float = 0.0  # on-site energy set to zero (rotating frame)

# Brillouin zone path for band structure
K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
K_LABELS: list = [r"$\Gamma$", r"X", r"M", r"$\Gamma$"]
N_K_PATH: int = 100  # number of k-points along the high-symmetry path

# Emitter
N_EMITTERS: int = 1
R_0: list = [0, 0]  # emitter position in lattice units (must be a lattice site)

# Local coupling strength [|J_1|] — adjust this freely
G: float = 0.1

# Time evolution [|J_1|⁻¹]
T_0: float = 0.0
T_F: float = 200
T_STEPS: int = 200
SNAP_IDX: int = 50

# Probe emitter
NEW_N_EMITTERS: int = 2
NEW_R: list = [-5, 5]  # probe position in lattice units (must be a lattice site)
G_PROBE: float = G  # probe coupling strength [|J_1|]
NEW_T_F: float = T_F  # [|J_1|⁻¹]
NEW_SNAP_IDX: int = SNAP_IDX

# Plot colors
MY_BLUE: str = "#1f77b4"
MY_GREEN: str = "#2ca02c"
MY_RED: str = "#d62728"


# ------------------------------------------------------------------------------
# Tight-binding band structure
# ------------------------------------------------------------------------------


def build_tight_binding_model() -> tb.tb_model:
    """
    Build the infinite tight-binding model for the photonic crystal bath.

    Returns
    -------
    tb.tb_model
        Tight-binding model with on-site and hopping terms up to 1st
        nearest neighbours.
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
    Extract the detuning frequency at the X point of the Brillouin zone.

    The X point is a saddle point of the band with a high local density
    of states, making it the natural choice for the emitter detuning.

    Parameters
    ----------
    tb_evals : np.ndarray, shape (Nb, Nk)
        Tight-binding eigenvalues along the k-path.
    k_dist : np.ndarray, shape (Nk,)
        Cumulative k-distances along the path.
    k_node : np.ndarray
        k-distances at each high-symmetry node.

    Returns
    -------
    float
        Detuning energy omega_0 at the X point in units of |J_1|.

    Raises
    ------
    ValueError
        If the X point is not found in k_dist.
    """
    x_point_mask = np.isclose(k_dist, k_node[1])
    if not np.any(x_point_mask):
        raise ValueError("X point not found in k_dist — check K_PATH and N_K_PATH.")
    return float(tb_evals[0][x_point_mask])


def plot_band_structure(
    k_dist: np.ndarray,
    k_node: np.ndarray,
    tb_evals: np.ndarray,
    omega_0: float,
) -> None:
    """
    Plot and save the tight-binding band structure along the high-symmetry path.

    Parameters
    ----------
    k_dist : np.ndarray, shape (Nk,)
        Cumulative k-distances along the path.
    k_node : np.ndarray
        k-distances at each high-symmetry node.
    tb_evals : np.ndarray, shape (Nb, Nk)
        Tight-binding eigenvalues along the k-path.
    omega_0 : float
        Detuning energy to mark on the plot [|J_1|].
    """
    fig, ax = plt.subplots()

    for tb_eval in tb_evals:
        ax.plot(k_dist, tb_eval, color="k")

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
        ylabel=r"$\varepsilon_\mathbf{k} \, [|J_1|]$",
        title="TB band structure (NN)",
    )
    ax.grid(axis="x")
    fig.savefig("./figures/kimble_toy/TB_bandstructure.svg")
    plt.close(fig)


# Run band structure computation
infinite_model = build_tight_binding_model()
k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals = infinite_model.solve_all(k_vec)

omega_0 = compute_detuning_frequency(tb_evals, k_dist, k_node)
print(f"Detuning energy at X point: omega_0 = {omega_0:.6f} |J_1|")
print(f"Local coupling:             G        = {G:.2e} |J_1|")

plot_band_structure(k_dist, k_node, tb_evals, omega_0)

# ------------------------------------------------------------------------------
# Isofrequency contour in the Brillouin zone
# ------------------------------------------------------------------------------


def compute_and_plot_isofrequency(model: tb.tb_model, omega_0: float) -> None:
    """
    Compute and plot the isofrequency contour of the tight-binding band.

    Parameters
    ----------
    model : tb.tb_model
        Infinite periodic tight-binding model.
    omega_0 : float
        Detuning energy to highlight as an isofrequency contour [|J_1|].
    """
    k_x = np.linspace(-0.5, 0.5, N_K_PATH)
    k_y = np.linspace(-0.5, 0.5, N_K_PATH)
    k_x_grid, k_y_grid = np.meshgrid(k_x, k_y, indexing="ij")

    ks_grid_flat = np.column_stack((k_x_grid.ravel(), k_y_grid.ravel()))
    tb_evals_grid = np.array([model.solve_one(k)[0] for k in ks_grid_flat])
    omega_grid = tb_evals_grid.reshape(N_K_PATH, N_K_PATH)

    fig, ax = plt.subplots()
    c = ax.contourf(k_x_grid, k_y_grid, omega_grid, levels=12)
    cbar = fig.colorbar(c, ax=ax, label=r"$\varepsilon_\mathbf{k} \, [|J_1|]$")
    cbar.formatter.set_powerlimits((0, 0))
    ax.contour(
        k_x_grid, k_y_grid, omega_grid, levels=[omega_0], colors=MY_RED, linewidths=2
    )
    ax.set(
        xlabel=r"$k_x a/2\pi$",
        ylabel=r"$k_y a/2\pi$",
        aspect="equal",
    )
    fig.savefig("./figures/kimble_toy/tb_isofrequency_contour.svg")
    plt.close(fig)


compute_and_plot_isofrequency(infinite_model, omega_0)

# ------------------------------------------------------------------------------
# Finite tight-binding model
# ------------------------------------------------------------------------------


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    """
    Compute per-orbital damping rates for the absorbing boundary layer.

    Uses a quadratic ramp from zero at the crystal-PML interface to
    ``gamma_max`` at the hard wall.

    Parameters
    ----------
    rs : np.ndarray, shape (n_orbitals, 2)
        Real-space grid points in lattice units, centered at the origin.
    n_cells : int
        Number of unit cells per direction.
    n_pml : int
        Thickness of the absorbing layer in unit cells.
    gamma_max : float
        Maximum damping rate at the outermost cell [|J_1|].

    Returns
    -------
    np.ndarray, shape (n_orbitals,)
        Per-orbital damping rates γ_j ≥ 0.
    """
    half = n_cells // 2
    dist_x = half - np.abs(rs[:, 0])
    dist_y = half - np.abs(rs[:, 1])
    dist = np.minimum(dist_x, dist_y)
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


def build_finite_model(infinite_model: tb.tb_model) -> tb.tb_model:
    """
    Cut the infinite tight-binding model into a finite 2D crystal.

    Parameters
    ----------
    infinite_model : tb.tb_model
        Infinite periodic tight-binding model.

    Returns
    -------
    tb.tb_model
        Finite model with N_CELLS unit cells in each direction.
    """
    temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
    return temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)


def find_site_index(rs: np.ndarray, r: list) -> int:
    """
    Find the index of the bath site closest to position r.

    Parameters
    ----------
    rs : np.ndarray, shape (n_orbitals, 2)
        Real-space grid points in lattice units.
    r : list
        Target position [x, y] in lattice units.

    Returns
    -------
    int
        Index of the closest site in rs.
    """
    dists = np.sum((rs - np.array(r, dtype=float)) ** 2, axis=1)
    return int(np.argmin(dists))


# Build finite model and its grid
finite_model = build_finite_model(infinite_model)
xs_finite, ys_finite, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)

# Bath site indices for emitter and probe
idx_r0 = find_site_index(rs_finite, R_0)
idx_new_r = find_site_index(rs_finite, NEW_R)
print(f"Emitter site index: {idx_r0}  →  r = {rs_finite[idx_r0]}")
print(f"Probe site index:   {idx_new_r}  →  r = {rs_finite[idx_new_r]}")

# Time arrays [|J_1|⁻¹]
ts = np.linspace(T_0, T_F, T_STEPS, endpoint=True)

# ------------------------------------------------------------------------------
# Toy Hamiltonian builder
# ------------------------------------------------------------------------------


def build_toy_hamiltonian(
    finite_model: tb.tb_model,
    omega_0s: float | list[float],
    couplings: list[tuple[float, int]],
    pml_dampings: np.ndarray | None = None,
) -> csr_matrix:
    """
    Build the full toy Hamiltonian H directly from the TB matrix (no sqrt).

    The bath block is H_b = Θ, the raw tight-binding Hamiltonian. The full
    Hamiltonian is:

        H = [ diag(omega_0s)   G^T ]
            [ G                Θ   ]

    where G is a sparse (n_emitters, n_orbitals) matrix with G_{e,j} = g_e
    if emitter e is coupled to bath site j, and 0 otherwise.

    Parameters
    ----------
    finite_model : tb.tb_model
        Finite tight-binding model. Must have no periodic boundary conditions.
    omega_0s : float or list of float
        Emitter on-site energies [|J_1|]. Broadcast if scalar.
    couplings : list of (g, bath_idx) pairs
        One entry per emitter. ``g`` is the local coupling strength [|J_1|]
        and ``bath_idx`` is the index in rs of the coupled bath site.
    pml_dampings : np.ndarray of shape (n_orbitals,) or None, optional
        Per-orbital PML damping rates [|J_1|]. Adds -i*diag(γ) to the bath
        block, making H non-Hermitian. Default is None.

    Returns
    -------
    H : scipy.sparse.csr_matrix
        Full (n_emitters + n_orbitals) × (n_emitters + n_orbitals) Hamiltonian.
    """
    n_orbitals = int(finite_model.get_num_orbitals())
    n_emitters = len(couplings)

    if np.isscalar(omega_0s):
        omega_0s = [omega_0s] * n_emitters

    # Build TB Hamiltonian Θ from private PythTB attributes (no public API)
    # NOTE: pin PythTB version if _site_energies or _hoppings are renamed.
    theta = np.zeros((n_orbitals, n_orbitals), dtype=complex)
    for i, val in enumerate(finite_model._site_energies):
        theta[i, i] += val
    for val, i, j in finite_model._hoppings:
        theta[i, j] += val
        theta[j, i] += np.conjugate(val)

    h_bath = csr_matrix(theta)

    if pml_dampings is not None:
        pml_dampings = np.asarray(pml_dampings, dtype=float)
        h_bath = h_bath - 1j * diags(pml_dampings, format="csr")

    h_emitter = diags(
        np.asarray(omega_0s, dtype=complex),
        offsets=0,
        shape=(n_emitters, n_emitters),
        format="csr",
    )

    g_vals = [g for g, _ in couplings]
    i_rows = list(range(n_emitters))
    j_cols = [idx for _, idx in couplings]

    h_int = csr_matrix(
        (g_vals, (i_rows, j_cols)),
        shape=(n_emitters, n_orbitals),
        dtype=complex,
    )

    return bmat([[h_emitter, h_int], [h_int.getH(), h_bath]], format="csr")


# ------------------------------------------------------------------------------
# Time evolution — single emitter
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
    Build the toy Hamiltonian with a local coupling and evolve in time.

    The emitter at r_0 couples only to the single bath site at its position
    with coupling strength g [|J_1|].

    Parameters
    ----------
    finite_model : tb.tb_model
        Finite tight-binding model of the photonic bath.
    rs : np.ndarray, shape (n_orbitals, 2)
        Real-space grid of bath sites in lattice units.
    omega_0 : float
        Emitter on-site energy [|J_1|].
    g : float
        Local coupling strength [|J_1|].
    r_0 : list
        Emitter position [x, y] in lattice units.
    t_0 : float
        Start time [|J_1|⁻¹].
    t_f : float
        End time [|J_1|⁻¹].
    t_steps : int
        Number of time steps.
    pml_dampings : np.ndarray of shape (n_orbitals,) or None, optional
        Per-orbital PML damping rates [|J_1|].

    Returns
    -------
    psi_t : np.ndarray, shape (t_steps, 1 + n_orbitals)
        Time-evolved wave function. Index 0 is the emitter amplitude.
    """
    idx = find_site_index(rs, r_0)
    H = build_toy_hamiltonian(
        finite_model=finite_model,
        omega_0s=omega_0,
        couplings=[(g, idx)],
        pml_dampings=pml_dampings,
    )

    psi_0 = np.zeros(H.shape[0])
    psi_0[0] = 1

    return expm_multiply(-1j * H, psi_0, t_0, t_f, t_steps, endpoint=True)


def plot_emitter_time_evolution(
    ts: np.ndarray,
    psi_t: np.ndarray,
    r_0: list,
) -> np.ndarray:
    """
    Plot and save the emitter population and entanglement entropy over time.

    Parameters
    ----------
    ts : np.ndarray
        Time array [|J_1|⁻¹].
    psi_t : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function.
    r_0 : list
        Emitter position in lattice units, used in filename.

    Returns
    -------
    S_t : np.ndarray
        Entanglement entropy over time.
    """
    mod_Ce_sq = np.abs(psi_t[:, 0]) ** 2
    S_t = mf.binary_entropy(mod_Ce_sq)

    fig, ax = plt.subplots()
    (l1,) = ax.plot(ts, mod_Ce_sq, lw=2, color=MY_BLUE, label=r"$|C_e(t)|^2$")
    ax.set(xlabel=r"$t \, [|J_1|^{-1}]$", ylabel=r"$|C_e(t)|^2$", yscale="log")

    ax2 = ax.twinx()
    (l2,) = ax2.plot(ts, S_t, ls="--", color=MY_BLUE, lw=2, label=r"$S(t)$")
    ax2.set(ylabel=r"$S(t)$")

    ax.legend([l1, l2], [l1.get_label(), l2.get_label()])

    fig.savefig(f"./figures/kimble_toy/emitter_timeevolution_entropy_at_{r_0}.svg")
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
    """
    Plot and save a bath population snapshot at a given time index.

    Parameters
    ----------
    pos_x : np.ndarray
        Unique x-coordinates of the crystal sites.
    pos_y : np.ndarray
        Unique y-coordinates of the crystal sites.
    psi_t : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function.
    snap_idx : int
        Time index for the snapshot.
    n_emitters : int
        Number of emitters (used to slice the photonic component).
    r_0 : list
        Emitter position in lattice units, used in title and filename.

    Returns
    -------
    z : np.ndarray
        Bath population grid at snap_idx.
    """
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
        title=rf"$\mathbf{{r}}_0/a = {r_0}$, $G = {G}\,|J_1|$",
        aspect="equal",
    )
    fig.savefig(f"./figures/kimble_toy/bath_snapshot_at_{r_0}.svg")
    plt.close(fig)
    return z


# Run single-emitter time evolution
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

S_t = plot_emitter_time_evolution(ts, psi_t, R_0)

# Get crystal positions and plot bath snapshot
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
    Compute the time evolution for the two-emitter system with a probe.

    Both emitters couple to their respective bath sites locally. The excited
    emitter is at r_0 with coupling g; the probe is at new_r with coupling
    g_probe. The initial state has only the excited emitter populated.

    Parameters
    ----------
    finite_model : tb.tb_model
        Finite tight-binding model of the photonic bath.
    rs : np.ndarray, shape (n_orbitals, 2)
        Real-space grid of bath sites in lattice units.
    omega_0 : float
        Common detuning energy for both emitters [|J_1|].
    g : float
        Local coupling strength of the excited emitter [|J_1|].
    r_0 : list
        Position of the excited emitter in lattice units.
    g_probe : float
        Local coupling strength of the probe emitter [|J_1|].
    new_r : list
        Position of the probe emitter in lattice units.
    t_0 : float
        Start time [|J_1|⁻¹].
    t_f : float
        End time [|J_1|⁻¹].
    t_steps : int
        Number of time steps.
    pml_dampings : np.ndarray of shape (n_orbitals,) or None, optional
        Per-orbital PML damping rates [|J_1|].

    Returns
    -------
    new_psi_t : np.ndarray, shape (t_steps, 2 + n_orbitals)
        Time-evolved wave function. Index 0 is the excited emitter, index 1
        is the probe emitter.
    """
    idx_main = find_site_index(rs, r_0)
    idx_probe = find_site_index(rs, new_r)

    H = build_toy_hamiltonian(
        finite_model=finite_model,
        omega_0s=omega_0,
        couplings=[(g, idx_main), (g_probe, idx_probe)],
        pml_dampings=pml_dampings,
    )

    psi_0 = np.zeros(H.shape[0])
    psi_0[0] = 1

    new_psi_t = expm_multiply(-1j * H, psi_0, t_0, t_f, t_steps, endpoint=True)
    return new_psi_t


def plot_probe_bath_snapshot(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    new_psi_t: np.ndarray,
    r_0: list,
) -> None:
    """
    Plot a bath population snapshot for the two-emitter system.

    Parameters
    ----------
    pos_x : np.ndarray
        Unique x-coordinates of the crystal sites.
    pos_y : np.ndarray
        Unique y-coordinates of the crystal sites.
    new_psi_t : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for the two-emitter system.
    r_0 : list
        Position of the excited emitter in lattice units.
    """
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
        title=rf"$\mathbf{{r}}_0 = {r_0}$, probe at {NEW_R}",
        xlim=(-8, 8),
        ylim=(-8, 8),
    )
    fig.savefig(f"./figures/kimble_toy/bath_snapshot_at_{r_0}_with_probe.svg")
    plt.close(fig)


def plot_probe_time_evolution(
    ts: np.ndarray,
    new_psi_t: np.ndarray,
    r_0: list,
) -> None:
    """
    Plot emitter populations and concurrence for the two-emitter system.

    Parameters
    ----------
    ts : np.ndarray
        Time array [|J_1|⁻¹].
    new_psi_t : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for the two-emitter system.
    r_0 : list
        Position of the excited emitter in lattice units.
    """
    mod_Ce_excited_sq = np.abs(new_psi_t[:, 0]) ** 2
    mod_Ce_probe_sq = np.abs(new_psi_t[:, 1]) ** 2
    C_t = 2 * np.abs(mod_Ce_excited_sq * np.conj(mod_Ce_probe_sq))
    C_max = C_t.max()

    fig, ax = plt.subplots()
    ax.plot(ts, mod_Ce_excited_sq, c=MY_BLUE, label=rf"$|C_e(t)|^2$, r$_0={r_0}$")
    ax.plot(ts, mod_Ce_probe_sq, c=MY_GREEN, label=rf"$|C_p(t)|^2$, r$_p={NEW_R}$")

    ax2 = ax.twinx()
    ax2.plot(ts, C_t, c=MY_RED, ls="--", label=r"$C(t)$")

    ax.set(
        xlabel=r"$t \, [|J_1|^{-1}]$",
        ylabel=r"$|C_e(t)|^2$",
        yscale="log",
    )
    ax.legend(loc="upper right")
    ax2.set_ylabel(r"$C(t)$", color=MY_RED)
    ax2.tick_params(axis="y", colors=MY_RED)
    ax2.spines["right"].set_color(MY_RED)
    pad = C_max * 0.05
    ax2.set_ylim((-pad, C_max + pad))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.savefig(
        f"./figures/kimble_toy/emitter_timeevolution_concurrence_at_{r_0}_and_{NEW_R}.svg"
    )
    plt.close(fig)


# Run probe emitter computation
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

new_ts = np.linspace(T_0, NEW_T_F, T_STEPS, endpoint=True)

plot_probe_bath_snapshot(pos_x, pos_y, new_psi_t, R_0)
plot_probe_time_evolution(new_ts, new_psi_t, R_0)
