"""
Atom–light coupling in a 2D photonic crystal: reproducing Kimble et al. (2019).

Reproduces and extends the theoretical framework of [1] to study the coupling
between a quantum emitter and a 2D photonic crystal. The photonic crystal is
simulated using a tight-binding model, and the emitter is treated as a two-level
system. Specifically, this script:

  - Computes the coupling strength g_{n,k}(r_0) and its real-space counterpart
    g_{n,r}(r_0) via a discrete Fourier transform over k-points.
  - Evaluates the local density of states (LDOS) at the emitter position using
    Lorentzian broadening.
  - Solves the time evolution of both the emitter and the photonic bath for
    different emitter polarizations.
  - Introduces a second emitter to probe bath-mediated emission and quantifies
    the entanglement between the two emitters via the concurrence.

References
----------
.. [1] Coome, L. A., et al. & Kimble, H. J. "Two-dimensional photonic crystals
       for engineering atom-light interactions." Proc. Natl. Acad. Sci. U.S.A.
       116(26), 12743–12751 (2019). https://doi.org/10.1073/pnas.1822110116

Notes
-----
This script assumes that the electromagnetic fields are already computed at the
emitter positions. If not, run `./miscellaneous/mpb_field_computation.py` first.
"""

# Standard library
from scipy.sparse.linalg import expm_multiply  # time-dependent Schrödinger equation

# Third-party
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import meep as mp  # photonic crystal simulation
from meep import mpb  # MPB eigensolver interface
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
            "font.size": 11,
            "mathtext.fontset": "stix",
            # Axes
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 1,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            # Legend
            "legend.fontsize": 10,
            "legend.frameon": False,
            # Save
            "savefig.bbox": "tight",
        }
    )


configure_matplotlib()

# ------------------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------------------

# MPB simulation
LAT_PARAM: float = 30e-9  # lattice parameter [m]
N_BANDS: int = 2
BAND_IDX: int = 0  # band index (0-based)
N_K: int = 64  # k-points per direction in Monkhorst-Pack grid
EPSILON_INPUT_FILE: str = "data/kimble/epsilon.h5"
# NOTE: N_K is also used to identify the file where the fields are stored.

# Emitter parameters
THETA_1: float = 90 * np.pi / 180  # emitter 1 orientation [rad]
THETA_2: float = 135 * np.pi / 180  # emitter 2 orientation [rad]
DIP_STRENGTH: float = 1100  # dipole moment strength [Debye]
P_A: float = mf.unit_dip(a=LAT_PARAM)
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A
N_EMITTERS: int = 1  # emitters per simulation
R_0: list = [0.0, 0.0]  # emitter position(s) in lattice units
# NOTE: Two simulations are run in total, one per orientation.
# NOTE: A list of positions is also accepted by R_0.

# Derived emitter quantities
THETAS: list = [THETA_1, THETA_2]
DIPOLES: list = [
    DIP_STRENGTH_NORM * np.array([np.cos(theta), np.sin(theta), 0]) for theta in THETAS
]
DIP_COLORS: list = ["#1f77b4", "#ff7f0e"]  # one color per dipole orientation

# Tight-binding model
LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]  # orbital position in lattice units
N_CELLS: int = 61  # unit cells per direction
# Hopping coefficients up to 3rd nearest neighbours [units: (c/a)²]
J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

# Brillouin zone path for band structure
K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
K_LABELS: list = [r"$\Gamma$", r"X", r"M", r"$\Gamma$"]
N_K_PATH: int = 100  # number of k-points along the high-symmetry path

# Real-space grid
N_R: int = 13  # number of grid points per direction

# Time evolution [units: a/c]
T_0: int = 0  # start time
T_F: int = 500  # end time
T_STEPS: int = 100  # number of time steps
SNAP_IDX: int = 50  # snapshot index for field plots (0 to T_STEPS)

# Probe emitter
NEW_N_EMITTERS: int = 2
NEW_R: list = [-4.5, 4.5]
NEW_THETA: float = np.deg2rad(45)
NEW_DIPOLE: np.ndarray = 1e-1 * np.array([np.cos(NEW_THETA), np.sin(NEW_THETA), 0])
NEW_DIP_COLOR: str = "#2ca02c"
NEW_T_F: int = 200
NEW_SNAP_IDX: int = 45
# NOTE: NEW_SNAP_IDX is taken around the time the bath excitation hits
#       the probe emitter. If a different position is chosen, this value
#       may need to be adjusted.

# Plot colors
MY_RED: str = "#d62728"

# ------------------------------------------------------------------------------
# Load simulation metadata
# ------------------------------------------------------------------------------

EPSILON_DATA: dict = mf.read_epsilon_file(EPSILON_INPUT_FILE)
RESOLUTION: int = EPSILON_DATA["resolution"]
LATTICE_VECS: list = EPSILON_DATA["lattice_vectors"]
LATTICE_SIZE: list = EPSILON_DATA["lattice_size"]


def print_simulation_summary() -> None:
    """Print a summary of the key simulation parameters to stdout."""
    print(f"\nNumber of bands      = {N_BANDS}")
    print(f"Epsilon file used    = {EPSILON_INPUT_FILE}")
    print(f"Resolution           = {RESOLUTION}")
    print(f"Lattice size         = {LATTICE_SIZE}")
    print(f"Lattice vectors      = {LATTICE_VECS}")


print_simulation_summary()

# ------------------------------------------------------------------------------
# Dielectric function
# ------------------------------------------------------------------------------


def plot_dielectric_function(r_0: list) -> None:
    """
    Plot the dielectric function ε(r) of the photonic crystal with emitter dipoles.

    Parameters
    ----------
    r_0 : list
        Position of the emitter(s) in lattice units.
    """
    md = mpb.MPBData(rectify=False, periods=2)
    epsilon_converted = mf.convert_epsilon(EPSILON_INPUT_FILE, md)

    pos_x = epsilon_converted["positions-lattice"][:, 0]
    pos_y = epsilon_converted["positions-lattice"][:, 1]
    epsilon_vals = epsilon_converted["epsilon_values"]

    fig, ax = plt.subplots()
    sc = ax.scatter(
        pos_x,
        pos_y,
        c=epsilon_vals,
        marker="s",
        cmap="copper_r",
        s=mf.marker_size(len(pos_x)),
    )
    fig.colorbar(sc, ax=ax, label=r"$\epsilon(\mathbf{r})$")
    ax.set(xlabel=r"$x/a$", ylabel=r"$y/a$", aspect="equal")

    for i, theta in enumerate(THETAS, start=1):
        mf.draw_arrow(
            ax,
            r_0,
            theta,
            color=DIP_COLORS[i - 1],
            label=rf"$\theta_{i}$",
            length=0.7,
            label_offset=0.17,
            arrow_offset=0.07,
        )

    fig.savefig(f"./figures/kimble/dielectric_function_dipoles_at_{r_0}.svg")


plot_dielectric_function(R_0)

# ------------------------------------------------------------------------------
# Tight-binding and MPB band structure
# ------------------------------------------------------------------------------


def safe_sqrt(arr: np.ndarray) -> np.ndarray:
    """
    Compute the square root of an array, clipping negative values to zero.

    Negative values can arise from numerical noise in the tight-binding
    eigenvalues and are set to zero before taking the square root.

    Parameters
    ----------
    arr : np.ndarray
        Input array, possibly complex due to numerical noise.

    Returns
    -------
    np.ndarray
        Element-wise square root of the real part, with negatives clipped.
    """
    return np.sqrt(np.clip(np.real(arr), 0, None))


def build_tight_binding_model() -> tb.tb_model:
    """
    Build the infinite tight-binding model for the photonic crystal.

    Returns
    -------
    tb.tb_model
        Tight-binding model with on-site and hopping terms up to 3rd
        nearest neighbours.
    """
    model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
    model.set_onsite([J_0])

    # First nearest neighbours
    for vec in np.array([[1, 0], [0, 1]]):
        model.set_hop(J_1, 0, 0, vec)

    # Second nearest neighbours
    for vec in np.array([[1, 1], [1, -1]]):
        model.set_hop(J_2, 0, 0, vec)

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
        Detuning frequency omega_0 at the X point in units of c/a.

    Raises
    ------
    ValueError
        If the X point is not found in k_dist.
    """
    x_point_mask = np.isclose(k_dist, k_node[1])
    if not np.any(x_point_mask):
        raise ValueError("X point not found in k_dist — check K_PATH and N_K_PATH.")
    return safe_sqrt(tb_evals[BAND_IDX][x_point_mask]).item()


def compute_mpb_bands(k_vec: np.ndarray) -> np.ndarray:
    """
    Compute the TE photonic bands using MPB along a given k-path.

    Parameters
    ----------
    k_vec : np.ndarray, shape (Nk, 3)
        k-point coordinates along the high-symmetry path.

    Returns
    -------
    np.ndarray, shape (Nk, Nb)
        Mode frequencies in units of c/a.
    """
    geometry_lattice = mp.Lattice(
        size=LATTICE_SIZE,
        basis1=LATTICE_VECS[0],
        basis2=LATTICE_VECS[1],
    )
    ms = mpb.ModeSolver(
        num_bands=N_BANDS,
        epsilon_input_file=EPSILON_INPUT_FILE,
        geometry_lattice=geometry_lattice,
        k_points=mf.array_to_Vector3(k_vec),
    )
    ms.run_te()
    return ms.all_freqs


def plot_band_structure(
    k_dist: np.ndarray,
    k_node: np.ndarray,
    tb_evals: np.ndarray,
    mpb_freqs: np.ndarray,
    omega_0: float,
) -> None:
    """
    Plot and save the TB vs MPB band structure along the high-symmetry path.

    Parameters
    ----------
    k_dist : np.ndarray, shape (Nk,)
        Cumulative k-distances along the path.
    k_node : np.ndarray
        k-distances at each high-symmetry node.
    tb_evals : np.ndarray, shape (Nb, Nk)
        Tight-binding eigenvalues along the k-path.
    mpb_freqs : np.ndarray, shape (Nk, Nb)
        MPB mode frequencies along the k-path.
    omega_0 : float
        Detuning frequency to mark on the plot [c/a].
    """
    fig, ax = plt.subplots()

    for tb_eval in tb_evals:
        ax.plot(k_dist, safe_sqrt(tb_eval), color="r", linestyle="--")

    ax.plot(k_dist, mpb_freqs, color="k")
    ax.hlines(
        omega_0,
        xmin=k_dist[0],
        xmax=k_dist[-1],
        color="k",
        linestyle="--",
        alpha=0.5,
    )
    ax.set(
        xlim=(k_node[0], k_node[-1]),
        xticks=k_node,
        xticklabels=K_LABELS,
        ylabel=r"$\omega \, [c/a]$",
        title="TE modes",
    )
    ax.grid(axis="x")
    fig.savefig("./figures/kimble/MPBvTB_bandstructure.svg")


# Run band structure computation
infinite_model = build_tight_binding_model()
k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals = infinite_model.solve_all(k_vec)

omega_0 = compute_detuning_frequency(tb_evals, k_dist, k_node)
print(f"Detuning frequency at X point: {omega_0:.6f} (c/a)")

mpb_freqs = compute_mpb_bands(k_vec)
plot_band_structure(k_dist, k_node, tb_evals, mpb_freqs, omega_0)

# ------------------------------------------------------------------------------
# Real-space emitter-bath coupling g_r(r_0)
# ------------------------------------------------------------------------------


def load_field_data(n_k: int, r_0: list) -> dict:
    """
    Load precomputed MPB fields, frequencies, and k-points from disk.

    Parameters
    ----------
    n_k : int
        Number of k-points per direction, used to identify the data file.
    r_0 : list
        Emitter position in lattice units, used to identify the data file.

    Returns
    -------
    dict
        Dictionary containing:
        - ``freqs``: mode frequencies, shape (Nk, Nb).
        - ``Eks``: electric fields, shape (Nr, Nb, Nk, 3).
        - ``kpts``: k-points in Cartesian coordinates, shape (Nk, 3).

    Notes
    -----
    All quantities are in MPB internal units.
    """
    return mf.load(f"./data/kimble/fields/{n_k}_meshgrid_at_{r_0}.pkl")


def compute_coupling_grid(
    fr0_n_ks: np.ndarray,
    omega_ks_n: np.ndarray,
    ks: np.ndarray,
    n_r: int,
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    Compute the real-space coupling g_r(r_0) on a square lattice grid.

    Parameters
    ----------
    fr0_n_ks : np.ndarray, shape (Nb, Nk, 3)
        Electric field profile at the emitter position.
    omega_ks_n : np.ndarray, shape (Nk, Nb)
        Mode frequencies in units of c/a.
    ks : np.ndarray, shape (Nk, 3)
        k-point coordinates in Cartesian coordinates.
    n_r : int
        Number of grid points per direction.

    Returns
    -------
    gr0_rs_list : list of np.ndarray, each shape (n_r, n_r)
        Real-space coupling for each dipole orientation.
    xs : np.ndarray
        x-coordinates of the real-space grid.
    ys : np.ndarray
        y-coordinates of the real-space grid.
    """
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
    """
    Plot and save |g_r(r_0)| for each dipole orientation.

    Parameters
    ----------
    gr0_rs_list : list of np.ndarray, each shape (n_r, n_r)
        Real-space coupling for each dipole orientation.
    xs : np.ndarray
        x-coordinates of the real-space grid.
    ys : np.ndarray
        y-coordinates of the real-space grid.
    r_0 : list
        Emitter position in lattice units, used in title and filename.
    """
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
            ax, r_0, theta, color=DIP_COLORS[i - 1], length=4, arrow_offset=0.3
        )
        ax.set(
            title=rf"$\theta_{i} = {np.rad2deg(theta):.0f}$°",
            xlabel=r"$x/a$",
            ylabel=r"$y/a$",
            aspect="equal",
        )
        ax.title.set_color(DIP_COLORS[i - 1])
        fig.savefig(
            f"./figures/kimble/g_r({r_0})_for_theta_{np.rad2deg(theta):.0f}deg.svg"
        )


# Run real-space coupling computation
fields_data = load_field_data(N_K, R_0)

omega_ks_n = np.asarray(fields_data["freqs"])  # shape: (Nk, Nb)
ks = np.asarray(fields_data["kpts"])  # shape: (Nk, 3)
fr0_n_ks = np.asarray(fields_data["Eks"])[0]  # shape: (Nb, Nk, 3)

gr0_rs_list, xs_small, ys_small = compute_coupling_grid(fr0_n_ks, omega_ks_n, ks, N_R)
plot_real_space_coupling(gr0_rs_list, xs_small, ys_small, R_0)

# ------------------------------------------------------------------------------
# LDOS at the emitter position
# ------------------------------------------------------------------------------


def compute_and_plot_ldos(
    omega_ks_n: np.ndarray,
    fr0_n_ks: np.ndarray,
    omega_0: float,
    r_0: list,
) -> None:
    """
    Compute and plot the LDOS at the emitter position for each dipole orientation.

    Parameters
    ----------
    omega_ks_n : np.ndarray, shape (Nk, Nb)
        Mode frequencies in units of c/a.
    fr0_n_ks : np.ndarray, shape (Nb, Nk, 3)
        Electric field profile at the emitter position.
    omega_0 : float
        Detuning frequency to mark on the plot [c/a].
    r_0 : list
        Emitter position in lattice units, used in title and filename.
    """
    omega_ks = omega_ks_n[:, BAND_IDX]  # shape: (Nk,)
    omegas = np.linspace(omega_ks.min(), omega_ks.max(), N_K_PATH)
    fr0_ks = fr0_n_ks[BAND_IDX, :, :]  # shape: (Nk, 3)

    ldos_list = [
        mf.compute_ldos(
            omegas=omegas, omega_ks=omega_ks, fr0_ks=fr0_ks, dipole_moment=dipole
        )
        for dipole in DIPOLES
    ]

    for i, ldos in enumerate(ldos_list, start=1):
        fig, ax = plt.subplots()
        ax.scatter(omegas, ldos, c=DIP_COLORS[i - 1])
        ax.axvline(
            x=omega_0,
            color="k",
            linestyle="--",
            label=r"$\omega_0$",
            linewidth=1,
        )
        ax.legend()
        ax.set(
            xlabel=r"$\omega \, [c/a]$",
            ylabel=r"$\mathrm{LDOS} \, [c/a]$",
            title=rf"$R_0/a$={r_0}, $\theta_{i}={np.degrees(THETAS[i - 1]):.0f}$°",
        )
        fig.savefig(
            f"./figures/kimble/ldos_at_{r_0}_theta_{np.rad2deg(THETAS[i - 1]):.0f}.svg"
        )


compute_and_plot_ldos(omega_ks_n, fr0_n_ks, omega_0, R_0)

# ------------------------------------------------------------------------------
# Isofrequency contour in the Brillouin zone
# ------------------------------------------------------------------------------


def compute_and_plot_isofrequency(omega_0: float) -> None:
    """
    Compute and plot the isofrequency contour of the tight-binding band.

    Parameters
    ----------
    omega_0 : float
        Detuning frequency to highlight as an isofrequency contour [c/a].
    """
    k_x = np.linspace(-0.5, 0.5, N_K_PATH)
    k_y = np.linspace(-0.5, 0.5, N_K_PATH)
    k_x_grid, k_y_grid = np.meshgrid(k_x, k_y, indexing="ij")

    ks_grid_flat = np.column_stack((k_x_grid.ravel(), k_y_grid.ravel()))
    tb_evals_grid = np.array(
        [infinite_model.solve_one(k)[BAND_IDX] for k in ks_grid_flat]
    )
    omega_grid = safe_sqrt(tb_evals_grid).reshape(N_K_PATH, N_K_PATH)

    fig, ax = plt.subplots()
    c = ax.contourf(k_x_grid, k_y_grid, omega_grid, levels=12)
    cbar = fig.colorbar(c, ax=ax, label=r"$\omega_\mathbf{k} \, [c/a]$")
    cbar.formatter.set_powerlimits((0, 0))
    ax.contour(
        k_x_grid, k_y_grid, omega_grid, levels=[omega_0], colors="k", linewidths=2
    )
    ax.set(
        xlabel=r"$k_x \, [2\pi/a]$",
        ylabel=r"$k_y \, [2\pi/a]$",
        aspect="equal",
    )
    fig.savefig("./figures/kimble/tb_isofrequency_contour.svg")


compute_and_plot_isofrequency(omega_0)

# ------------------------------------------------------------------------------
# Finite tight-binding model and time evolution
# ------------------------------------------------------------------------------


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


def build_finite_grid(n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the real-space grid of the finite crystal.

    Parameters
    ----------
    n_cells : int
        Number of unit cells per direction.

    Returns
    -------
    xs : np.ndarray
        x-coordinates of the grid.
    ys : np.ndarray
        y-coordinates of the grid.
    rs : np.ndarray, shape (Nr, 2)
        Flattened grid points.
    """
    if n_cells % 2 == 0:
        xs = np.arange(-n_cells / 2, n_cells / 2, 1)
        ys = np.arange(-n_cells / 2, n_cells / 2 + 1, 1)
    else:
        xs = np.arange(-(n_cells // 2), n_cells // 2 + 1, 1)
        ys = np.arange(-(n_cells // 2), n_cells // 2 + 1, 1)
    rs = mf.meshgrid_points(xs, ys, indexing="xy")
    return xs, ys, rs


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Build the light-matter Hamiltonians and compute the time evolution.

    Parameters
    ----------
    finite_model : tb.tb_model
        Finite tight-binding model for the photonic crystal.
    omega_ks_n : np.ndarray, shape (Nk, Nb)
        Mode frequencies in units of c/a.
    fr0_n_ks : np.ndarray, shape (Nb, Nk, 3)
        Electric field profile at the emitter position.
    ks : np.ndarray, shape (Nk, 3)
        k-point coordinates in Cartesian coordinates.
    rs : np.ndarray, shape (Nr, 2)
        Real-space grid points.
    omega_0 : float
        Emitter detuning frequency [c/a].
    t_0 : float
        Start time [a/c].
    t_f : float
        End time [a/c].
    t_steps : int
        Number of time steps.

    Returns
    -------
    psi_t_1 : np.ndarray
        Time-evolved wave function for dipole 1, shape (t_steps, Ndof).
    psi_t_2 : np.ndarray
        Time-evolved wave function for dipole 2, shape (t_steps, Ndof).
    omega_sites_1 : np.ndarray
        Site energies from Hamiltonian 1.
    omega_sites_2 : np.ndarray
        Site energies from Hamiltonian 2.
    couplings : list of np.ndarray
        Real-space couplings for each dipole, shape (Nr,) each.

    Raises
    ------
    ValueError
        If the two coupling arrays have different lengths.
        If the two Hamiltonians have different shapes.
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

    if len(couplings[0]) != len(couplings[1]):
        raise ValueError(
            f"Coupling arrays have different lengths: "
            f"{len(couplings[0])} vs {len(couplings[1])}."
        )

    idxs = np.arange(len(couplings[0]))
    interaction_1 = [[list(couplings[0]), list(idxs)]]
    interaction_2 = [[list(couplings[1]), list(idxs)]]

    H_1, omega_sites_1 = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model, ω_0s=omega_0, gr0s_rs=interaction_1
    )
    H_2, omega_sites_2 = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model, ω_0s=omega_0, gr0s_rs=interaction_2
    )

    if H_1.shape != H_2.shape:
        raise ValueError(
            f"Hamiltonians have different shapes: {H_1.shape} vs {H_2.shape}."
        )

    psi_0 = np.zeros(H_1.shape[0])
    psi_0[0] = 1

    psi_t_1 = expm_multiply(-1j * H_1, psi_0, t_0, t_f, t_steps, endpoint=True)
    psi_t_2 = expm_multiply(-1j * H_2, psi_0, t_0, t_f, t_steps, endpoint=True)

    return psi_t_1, psi_t_2, omega_sites_1, omega_sites_2, couplings


def plot_emitter_time_evolution(
    ts: np.ndarray,
    psi_t_1: np.ndarray,
    psi_t_2: np.ndarray,
    r_0: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot and save the emitter population and entanglement entropy over time.

    Parameters
    ----------
    ts : np.ndarray
        Time array in units of Gamma_0^{-1}.
    psi_t_1 : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for dipole 1.
    psi_t_2 : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for dipole 2.
    r_0 : list
        Emitter position in lattice units, used in filename.

    Returns
    -------
    S_t_1 : np.ndarray
        Entanglement entropy for dipole 1.
    S_t_2 : np.ndarray
        Entanglement entropy for dipole 2.
    """
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
    ax.set(xlabel=r"$t \, [a/c]$", ylabel=r"$|C_e(t)|^2$", yscale="log")

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
        S_t_1,
        ls="--",
        color=DIP_COLORS[1],
        alpha=0.9,
        lw=2.0,
        label=r"$S(t)$, $\theta_2 = 135$°",
    )
    ax2.set(ylabel=r"$S(t)$")

    lines = [l1, l4]
    ax.legend(lines, [line.get_label() for line in lines])

    fig.savefig(f"./figures/kimble/emitter_timeevolution_entropy_at_{r_0}.svg")
    return S_t_1, S_t_2


def plot_bath_snapshot(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    psi_t_1: np.ndarray,
    psi_t_2: np.ndarray,
    snap_idx: int,
    n_emitters: int,
    r_0: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot and save bath population snapshots at a given time index.

    Parameters
    ----------
    pos_x : np.ndarray
        Unique x-coordinates of the crystal sites.
    pos_y : np.ndarray
        Unique y-coordinates of the crystal sites.
    psi_t_1 : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for dipole 1.
    psi_t_2 : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for dipole 2.
    snap_idx : int
        Time index for the snapshot.
    n_emitters : int
        Number of emitters (used to slice the photonic component).
    r_0 : list
        Emitter position in lattice units, used in title and filename.

    Returns
    -------
    z1 : np.ndarray
        Bath population grid for dipole 1.
    z2 : np.ndarray
        Bath population grid for dipole 2.
    pos_x : np.ndarray
        x-coordinates (passed through for reuse).
    pos_y : np.ndarray
        y-coordinates (passed through for reuse).
    """
    mod_Cr_sq_1 = np.abs(psi_t_1[snap_idx, n_emitters:]) ** 2
    mod_Cr_sq_2 = np.abs(psi_t_2[snap_idx, n_emitters:]) ** 2

    z1 = mod_Cr_sq_1.reshape(len(pos_x), len(pos_y))
    z2 = mod_Cr_sq_2.reshape(len(pos_x), len(pos_y))
    cbar_max = max(z1.max(), z2.max())

    for i, (theta, z) in enumerate(zip(THETAS, [z1, z2])):
        fig, ax = plt.subplots()
        sc = ax.pcolormesh(pos_x, pos_y, z, vmin=0, vmax=cbar_max)
        mf.draw_arrow(ax, r_0, theta, color=DIP_COLORS[i], length=16, arrow_offset=2)
        cbar = fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{R}(t)|^2$", pad=0.1)
        cbar.formatter.set_powerlimits((0, 0))
        ax.set(
            xlabel=r"$x/a$",
            ylabel=r"$y/a$",
            title=rf"$\mathbf{{r}}_0/a = {r_0}$",
            aspect="equal",
        )
        fig.savefig(
            f"./figures/kimble/bath_snapshot_at_{r_0}_theta_{np.rad2deg(theta):.0f}deg.svg"
        )

    return z1, z2, pos_x, pos_y


# Build finite model and its grid
finite_model = build_finite_model(infinite_model)
xs_finite, ys_finite, rs_finite = build_finite_grid(N_CELLS)

# Compute time evolution
ts = np.linspace(T_0, T_F, T_STEPS, endpoint=True)

psi_t_1, psi_t_2, omega_sites_1, omega_sites_2, couplings = compute_time_evolution(
    finite_model=finite_model,
    omega_ks_n=omega_ks_n,
    fr0_n_ks=fr0_n_ks,
    ks=ks,
    rs=rs_finite,
    omega_0=omega_0,
    t_0=T_0,
    t_f=T_F,
    t_steps=T_STEPS,
)

# Plot emitter time evolution
S_t_1, S_t_2 = plot_emitter_time_evolution(ts, psi_t_1, psi_t_2, R_0)

# Get crystal positions and plot bath snapshot
positions = mf.find_positions(finite_model._orb, finite_model._lat)
pos_x = np.unique(positions[:, 0]) - N_CELLS // 2
pos_y = np.unique(positions[:, 1]) - N_CELLS // 2

z1, z2, pos_x, pos_y = plot_bath_snapshot(
    pos_x, pos_y, psi_t_1, psi_t_2, SNAP_IDX, N_EMITTERS, R_0
)

# ------------------------------------------------------------------------------
# Quadrant entropy to verify directional emission channels
# ------------------------------------------------------------------------------

Cr_t_1 = psi_t_1[:, N_EMITTERS:]
Cr_t_2 = psi_t_2[:, N_EMITTERS:]

S_1, p_1 = mf.quadrant_entropy(rs_finite[:, 0], rs_finite[:, 1], Cr_t_1)
S_2, p_2 = mf.quadrant_entropy(rs_finite[:, 0], rs_finite[:, 1], Cr_t_2)

check_1 = np.allclose(S_1[1:], np.log(4), atol=1e-3)
check_2 = np.allclose(S_2[1:], np.log(2), atol=1e-3)

print(f"Is S_dir equal to log(4) for theta = {np.rad2deg(THETA_1):.0f}deg? {check_1}")
print(f"Is S_dir equal to log(2) for theta = {np.rad2deg(THETA_2):.0f}deg? {check_2}")

# ------------------------------------------------------------------------------
# Probe emitter: directional emission and concurrence
# ------------------------------------------------------------------------------


def compute_probe_emitter_evolution(
    finite_model: tb.tb_model,
    omega_ks_n: np.ndarray,
    fr0_n_ks: np.ndarray,
    ks: np.ndarray,
    rs: np.ndarray,
    couplings: list,
    omega_sites_1: np.ndarray,
    omega_sites_2: np.ndarray,
    omega_0: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the time evolution for the two-emitter system with a probe emitter.

    Places a probe emitter at NEW_R to verify bath-mediated directional emission.
    The probe is excited when the first emitter has orientation THETA_1 and not
    THETA_2, demonstrating polarization-controlled photon routing.

    Parameters
    ----------
    finite_model : tb.tb_model
        Finite tight-binding model for the photonic crystal.
    omega_ks_n : np.ndarray, shape (Nk, Nb)
        Mode frequencies in units of c/a.
    fr0_n_ks : np.ndarray, shape (Nb, Nk, 3)
        Electric field at the excited emitter position.
    ks : np.ndarray, shape (Nk, 3)
        k-point coordinates.
    rs : np.ndarray, shape (Nr, 2)
        Real-space grid points.
    couplings : list of np.ndarray
        Real-space couplings for the excited emitter.
    omega_sites_1 : np.ndarray
        Site energies from Hamiltonian 1 (used to find probe detuning).
    omega_sites_2 : np.ndarray
        Site energies from Hamiltonian 2 (used to verify consistency).
    omega_0 : float
        Excited emitter detuning frequency [c/a].

    Returns
    -------
    new_psi_t_1 : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for system 1 (excited + probe).
    new_psi_t_2 : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for system 2 (excited + probe).
    new_omega_0 : float
        Probe emitter detuning frequency [c/a].

    Raises
    ------
    ValueError
        If probe detuning frequencies from the two Hamiltonians do not match.
    """
    # Find the probe emitter detuning at the target lattice site
    target = [-5, 5]
    idx = np.where(np.all(np.isclose(rs, target), axis=1))[0][0]

    if not np.isclose(omega_sites_1[idx], omega_sites_2[idx]):
        raise ValueError(
            f"Probe detuning frequencies differ between Hamiltonians: "
            f"{omega_sites_1[idx]} vs {omega_sites_2[idx]}."
        )
    new_omega_0 = omega_sites_1[idx]

    # Load fields at the probe position and compute coupling
    probe_fields_data = load_field_data(N_K, NEW_R)
    new_omega_ks_n = np.asarray(probe_fields_data["freqs"])  # (Nk, Nb)
    new_fr0_n_ks = np.asarray(probe_fields_data["Eks"])[0]  # (Nb, Nk, 3)
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
        ω_0s=[omega_0, new_omega_0],
        gr0s_rs=new_interaction_1,
    )
    new_H_2, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        ω_0s=[omega_0, new_omega_0],
        gr0s_rs=new_interaction_2,
    )

    new_psi_0 = np.zeros(new_H_1.shape[0])
    new_psi_0[0] = 1

    new_ts = np.linspace(T_0, NEW_T_F, T_STEPS, endpoint=True)
    new_psi_t_1 = expm_multiply(
        -1j * new_H_1, new_psi_0, T_0, NEW_T_F, T_STEPS, endpoint=True
    )
    new_psi_t_2 = expm_multiply(
        -1j * new_H_2, new_psi_0, T_0, NEW_T_F, T_STEPS, endpoint=True
    )

    return new_psi_t_1, new_psi_t_2, new_omega_0


def plot_probe_bath_snapshot(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    new_psi_t_1: np.ndarray,
    new_psi_t_2: np.ndarray,
    r_0: list,
) -> None:
    """
    Plot bath population snapshots for the two-emitter system.

    Parameters
    ----------
    pos_x : np.ndarray
        Unique x-coordinates of the crystal sites.
    pos_y : np.ndarray
        Unique y-coordinates of the crystal sites.
    new_psi_t_1 : np.ndarray, shape (t_steps, Ndof)
        Wave function for system 1 (excited + probe).
    new_psi_t_2 : np.ndarray, shape (t_steps, Ndof)
        Wave function for system 2 (excited + probe).
    r_0 : list
        Position of the excited emitter in lattice units.
    """
    new_mod_Cr_sq_1 = np.abs(new_psi_t_1[NEW_SNAP_IDX, NEW_N_EMITTERS:]) ** 2
    new_mod_Cr_sq_2 = np.abs(new_psi_t_2[NEW_SNAP_IDX, NEW_N_EMITTERS:]) ** 2

    new_z1 = new_mod_Cr_sq_1.reshape(len(pos_x), len(pos_y))
    new_z2 = new_mod_Cr_sq_2.reshape(len(pos_x), len(pos_y))
    cbar_max = max(new_z1.max(), new_z2.max())

    for i, (theta, z) in enumerate(zip(THETAS, [new_z1, new_z2]), start=1):
        fig, ax = plt.subplots()
        sc = ax.pcolormesh(pos_x, pos_y, z, vmin=0, vmax=cbar_max)
        mf.draw_arrow(ax, NEW_R, NEW_THETA, NEW_DIP_COLOR, length=4, arrow_offset=0.2)
        mf.draw_arrow(ax, r_0, theta, DIP_COLORS[i - 1], length=4, arrow_offset=0.2)
        cbar = fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{r}(t)|^2$")
        cbar.formatter.set_powerlimits((0, 0))
        ax.set(
            xlabel=r"$x/a$",
            ylabel=r"$y/a$",
            aspect="equal",
            title=rf"$\theta_{i} = {np.rad2deg(theta):.0f}$°",
            xlim=(-7, 7),
            ylim=(-7, 7),
        )
        ax.title.set_color(DIP_COLORS[i - 1])
        fig.savefig(
            f"./figures/kimble/bath_snapshot_at_{r_0}_theta_{np.rad2deg(theta):.0f}deg_with_probe.svg"
        )


def plot_probe_time_evolution(
    ts: np.ndarray,
    new_psi_t_1: np.ndarray,
    new_psi_t_2: np.ndarray,
    r_0: list,
) -> None:
    """
    Plot emitter populations and concurrence for the two-emitter system.

    Parameters
    ----------
    ts : np.ndarray
        Time array in units of Gamma_0^{-1}.
    new_psi_t_1 : np.ndarray, shape (t_steps, Ndof)
        Wave function for system 1 (excited + probe).
    new_psi_t_2 : np.ndarray, shape (t_steps, Ndof)
        Wave function for system 2 (excited + probe).
    r_0 : list
        Position of the excited emitter in lattice units.
    """
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
            xlabel=r"$t [a/c]$",
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
            f"./figures/kimble/emitter_timeevolution_concurrence_at_{r_0}"
            f"_theta_{np.rad2deg(theta):.0f}deg_and_{NEW_R}.svg"
        )


# Run probe emitter computation
new_psi_t_1, new_psi_t_2, new_omega_0 = compute_probe_emitter_evolution(
    finite_model=finite_model,
    omega_ks_n=omega_ks_n,
    fr0_n_ks=fr0_n_ks,
    ks=ks,
    rs=rs_finite,
    couplings=couplings,
    omega_sites_1=omega_sites_1,
    omega_sites_2=omega_sites_2,
    omega_0=omega_0,
)

new_ts = np.linspace(T_0, NEW_T_F, T_STEPS, endpoint=True)

plot_probe_bath_snapshot(pos_x, pos_y, new_psi_t_1, new_psi_t_2, R_0)
plot_probe_time_evolution(new_ts, new_psi_t_1, new_psi_t_2, R_0)
