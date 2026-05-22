"""
Atom–light coupling in a 2D photonic crystal: reproducing Kimble et al. (2019).

Reproduces and extends the theoretical framework of [1] to study the coupling
between a quantum emitter and a 2D photonic crystal. The photonic crystal is
simulated using a tight-binding model, and the emitter is treated as a two-level
system. Specifically, this script:

  - Computes the coupling strength g_{n,k}(r_0) and its real-space counterpart
    g_{n,r}(r_0) via a discrete Fourier transform over k-points.
  - Evaluates the spontaneous-emission decay rate Γ(ω) at the emitter position
    via Fermi's golden rule with Lorentzian broadening.
  - Solves the time evolution of both the emitter and the photonic bath for
    different emitter polarizations.
  - Introduces a second emitter to probe bath-mediated emission and quantifies
    the entanglement between the two emitters via the concurrence.

References
----------
.. [1] Coome, L. A., et al. & Kimble, H. J. "Two-dimensional photonic crystals
       for engineering atom-light interactions." Proc. Natl. Acad. Sci. U.S.A.
       116(26), 12743-12751 (2019). https://doi.org/10.1073/pnas.1822110116

Notes
-----
This script assumes that the electromagnetic fields are already computed at the
emitter positions. If not, run `./miscellaneous/mpb_field_computation.py` first.
"""

# Standard library
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import meep as mp  # photonic crystal simulation

# Third-party
import numpy as np
import pythtb as tb  # tight-binding model
from meep import mpb  # MPB eigensolver interface

# Local
import emitter_crystal_coupling as mf
from emitter_crystal_coupling.constants import C_LIGHT

# ------------------------------------------------------------------------------
# Matplotlib configuration
# ------------------------------------------------------------------------------


def configure_matplotlib() -> None:
    """Configure Matplotlib rcParams for journal-quality figures.

    Sized for PRX half-column panels (~42.5 mm) assembled in Inkscape.
    """
    mm_to_inches = 1 / 25.4
    mpl.rcParams.update(
        {
            # Figure
            "figure.figsize": (42.5 * mm_to_inches, 42.5 * mm_to_inches),
            "figure.dpi": 300,
            "figure.constrained_layout.use": True,
            # Font
            "font.family": "serif",
            "font.size": 8,
            "mathtext.fontset": "stix",
            # Axes
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.5,
            # Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            # Lines
            "lines.linewidth": 0.8,
            # Legend
            "legend.fontsize": 7,
            "legend.frameon": False,
            # Save
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.format": "svg",
            "savefig.dpi": 300,
            "svg.fonttype": "none",
        }
    )


configure_matplotlib()

# Per-figure sizes for PRX half-column panels assembled in Inkscape.
_MM_TO_IN = 1 / 25.4
_HALF_COL = 42.5 * _MM_TO_IN
FIGSIZE_SQUARE_CBAR = (_HALF_COL, _HALF_COL * 0.95)  # square data axis + colorbar
FIGSIZE_WIDE = (_HALF_COL, _HALF_COL * 0.75)  # x-vs-y line plots
FIGSIZE_TIME_EVOL = (_HALF_COL, _HALF_COL * 0.6)  # time evolution / entropy line plots


# ------------------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------------------

# MPB simulation
LAT_PARAM: float = 300e-9  # lattice parameter [m]
N_BANDS: int = 2
BAND_IDX: int = 0  # band index (0-based)
N_K: int = 81  # k-points per direction in Monkhorst-Pack grid
RESOLUTION: int = 64  # MPB grid resolution (must match a precomputed epsilon file)
EPSILON_INPUT_FILE: str = f"./data/kimble/{RESOLUTION}-epsilon.h5"
# NOTE: N_K is also used to identify the file where the fields are stored.

# Emitter parameters
THETA_1: float = 90 * np.pi / 180  # emitter 1 orientation [rad]
THETA_2: float = 135 * np.pi / 180  # emitter 2 orientation [rad]
DIP_STRENGTH: float = 50  # default dipole moment strength [Debye]
DIP_STRENGTH_HIGH: float = (
    10000  # high dipole moment for strong-coupling comparison [Debye]
)
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A
DIP_STRENGTH_HIGH_NORM: float = DIP_STRENGTH_HIGH / P_A
N_EMITTERS: int = 1  # emitters per simulation
R_0: list[float] = [0, 0]  # emitter position(s) in lattice units
# NOTE: A list of positions is also accepted by R_0.

# Derived emitter quantities
THETAS: list[float] = [THETA_1, THETA_2]
DIPOLES: list[np.ndarray] = [
    DIP_STRENGTH_NORM * np.array([np.cos(theta), np.sin(theta), 0]) for theta in THETAS
]
# 90° dipole at the high strength, used by the single-emitter strength
# comparison alongside DIPOLES[0] (which is 90° at the default strength).
DIPOLE_90_HIGH: np.ndarray = DIP_STRENGTH_HIGH_NORM * np.array(
    [np.cos(THETA_1), np.sin(THETA_1), 0]
)
DIP_COLORS: list[str] = ["#e377c2", "#ff7f0e"]  # one color per dipole orientation

# Tight-binding model
LAT: list[list[int]] = [[1, 0], [0, 1]]
ORB: list[list[int]] = [[0, 0]]  # orbital position in lattice units
N_CELLS: int = 50  # unit cells per direction
N_PML: int = 10  # absorbing layer thickness [unit cells per edge]
GAMMA_MAX: float = 1e-2  # maximum damping rate at the hard wall [c/a]
# Hopping coefficients up to 2nd nearest neighbours [units: (c/a)²]
J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

# Brillouin zone path for band structure
K_PATH: list[list[float]] = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
K_LABELS: list[str] = [r"$\Gamma$", r"X", r"M", r"$\Gamma$"]
N_K_PATH: int = 100  # number of k-points along the high-symmetry path

# Real-space grid
N_R: int = 13  # number of grid points per direction

# Time evolution [units: 1/Γ₀]
T_0: float = 0  # start time
T_F: float = 3e-2  # end time
T_STEPS: int = 200  # number of time steps
SNAP_IDX: int = 5  # snapshot index for field plots (0 to T_STEPS)

# Probe emitter
NEW_N_EMITTERS: int = 2
NEW_R: list[float] = [-5, 5]
NEW_THETA: float = np.deg2rad(45)
NEW_DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array(
    [np.cos(NEW_THETA), np.sin(NEW_THETA), 0]
)
NEW_DIP_COLOR: str = "#2ca02c"
NEW_T_F: float = T_F  # [1/Γ₀]
NEW_SNAP_IDX: int = 10
# NOTE: NEW_SNAP_IDX is taken around the time the bath excitation hits
#       the probe emitter. If a different position is chosen, this value
#       may need to be adjusted.

# Plot colors
MY_RED: str = "#d62728"

# ------------------------------------------------------------------------------
# Load simulation metadata
# ------------------------------------------------------------------------------

EPSILON_DATA: dict = mf.read_epsilon_file(EPSILON_INPUT_FILE)
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


def plot_dielectric_function(r_0: list[float]) -> None:
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

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE_CBAR)
    sc = ax.scatter(
        pos_x,
        pos_y,
        c=epsilon_vals,
        marker="s",
        cmap="copper_r",
        s=mf.marker_size(len(pos_x)),
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, label=r"$\epsilon(\mathbf{r})$", shrink=0.7, aspect=30)
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
            lw=1,
            mutation_scale=5,
            fontsize=7,
        )

    fig.savefig(f"./figures/kimble/dielectric_function_dipoles_at_{r_0}.svg")
    plt.close(fig)


plot_dielectric_function(R_0)

# ------------------------------------------------------------------------------
# Tight-binding and MPB band structure
# ------------------------------------------------------------------------------


def build_tight_binding_model() -> tb.tb_model:
    """
    Build the infinite tight-binding model for the photonic crystal.

    Returns
    -------
    tb.tb_model
        Tight-binding model with on-site and hopping terms up to 2nd
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


def compute_omega_X(
    tb_evals: np.ndarray,
    k_dist: np.ndarray,
    k_node: np.ndarray,
) -> float:
    """
    Extract the band frequency at the X point of the Brillouin zone.

    The X point is a saddle point of the band with a high local density
    of states.  Used as a reference frequency on the band-structure plot;
    the emitter detuning ω₀ is taken from the peak of Γ(ω) (see
    `compute_and_plot_decay_rate`), which lies slightly off ω_X (in the
    current setup, ~0.012 c/a below).

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
        Frequency ω_X at the X point in units of c/a.

    Raises
    ------
    ValueError
        If the X point is not found in k_dist.
    """
    x_point_mask = np.isclose(k_dist, k_node[1])
    if not np.any(x_point_mask):
        raise ValueError("X point not found in k_dist — check K_PATH and N_K_PATH.")
    return mf.safe_sqrt(tb_evals[BAND_IDX][x_point_mask]).item()


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
        size=mp.Vector3(*LATTICE_SIZE),
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
    assert ms.all_freqs is not None
    return ms.all_freqs


def plot_band_structure(
    k_dist: np.ndarray,
    k_node: np.ndarray,
    tb_evals: np.ndarray,
    mpb_freqs: np.ndarray,
    omega_X: float,
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
    omega_X : float
        X-point frequency to mark on the plot [c/a].
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for tb_eval in tb_evals:
        ax.plot(k_dist, mf.safe_sqrt(tb_eval), color="r", linestyle="--")

    ax.plot(k_dist, mpb_freqs, color="k")
    ax.hlines(
        omega_X,
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
    plt.close(fig)


# Run band structure computation
infinite_model = build_tight_binding_model()
k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals: np.ndarray = np.asarray(infinite_model.solve_all(k_vec))

omega_X = compute_omega_X(tb_evals, k_dist, k_node)
print(f"Frequency at X point:       ω_X = {omega_X:.6f} (c/a)")

mpb_freqs = compute_mpb_bands(k_vec)
plot_band_structure(k_dist, k_node, tb_evals, mpb_freqs, omega_X)

# ------------------------------------------------------------------------------
# Real-space emitter-bath coupling g_r(r_0)
# ------------------------------------------------------------------------------


def load_field_data(n_k: int, r_0: list[float]) -> dict:
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
    return mf.load(
        f"./data/kimble/fields/res_{RESOLUTION}/nk_{n_k}/meshgrid_at_{r_0}.pkl"
    )


def compute_coupling_grid(
    fr0_n_ks: np.ndarray,
    omega_ks_n: np.ndarray,
    ks: np.ndarray,
    n_r: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
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
    gr0_rs_list: list[np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    r_0: list[float],
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
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE_CBAR)
        sc = ax.pcolormesh(
            xs,
            ys,
            np.abs(gr0_rs_list[i - 1]),
            norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max),
        )
        fig.colorbar(
            sc,
            ax=ax,
            label=r"$\mathcal{G}_\mathbf{R}(\mathbf{r}_0)$",
            shrink=0.6,
            aspect=20,
        )
        mf.draw_arrow(
            ax,
            r_0,
            theta,
            color=DIP_COLORS[i - 1],
            length=4,
            arrow_offset=0.4,
            lw=0.8,
            mutation_scale=5,
        )
        ax.set(
            xlabel=r"$x/a$",
            ylabel=r"$y/a$",
            aspect="equal",
        )
        fig.savefig(
            f"./figures/kimble/G_r({r_0})_for_theta_{np.rad2deg(theta):.0f}deg.svg"
        )
        plt.close(fig)


# Run real-space coupling computation
fields_data = load_field_data(N_K, R_0)

omega_ks_n = np.asarray(fields_data["freqs"])  # shape: (Nk, Nb)
ks = np.asarray(fields_data["kpts"])  # shape: (Nk, 3)
fr0_n_ks = np.asarray(fields_data["Eks"])[0]  # shape: (Nb, Nk, 3)

gr0_rs_list, xs_small, ys_small = compute_coupling_grid(fr0_n_ks, omega_ks_n, ks, N_R)
plot_real_space_coupling(gr0_rs_list, xs_small, ys_small, R_0)

# ------------------------------------------------------------------------------
# Coupling diagnostics: effective g(R=r_0, r_0) vs toy-model G/|J_1| ratio
# ------------------------------------------------------------------------------

# In kimble_toy.py: bath is Θ directly, so G/|J_1_toy| is dimensionless (= 0.1).
# Here: bath energies are ω_k = √ε_k [c/a], so the analogous dimensionless ratio
# is g_R(r_0) / √|J_1| — both numerator and denominator are in c/a.
_TOY_G_OVER_J1: float = 0.1

_sqrt_J1 = np.sqrt(abs(J_1))
print("\n--- Coupling diagnostics (R = r_0) ---")
print(f"  |J_1|             = {abs(J_1):.6e} (c/a)²")
print(f"  √|J_1|            = {_sqrt_J1:.6e} c/a  ← bath frequency scale")
print(f"  toy G/|J_1_toy|   = {_TOY_G_OVER_J1}")

for _i, (_theta, _g_rs) in enumerate(zip(THETAS, gr0_rs_list), start=1):
    _g_abs = np.abs(_g_rs)
    _g0 = _g_abs[N_R // 2, N_R // 2]
    _g_max = _g_abs.max()
    # Use the peak coupling as the scale: in the toy model G is both on-site
    # and the maximum, so |g|_max is the fair comparison when the real-space
    # coupling has its peak away from r_0.
    _ratio_J1 = _g_max / abs(J_1)  # units: a/c
    _ratio_sqrtJ1 = _g_max / _sqrt_J1  # dimensionless — comparable to toy
    _dip_needed = DIP_STRENGTH * _TOY_G_OVER_J1 / _ratio_sqrtJ1
    print(
        f"\n  θ_{_i} = {np.rad2deg(_theta):.0f}°  (DIP_STRENGTH = {DIP_STRENGTH} Debye)"
    )
    print(f"    |g(r_0, r_0)|        = {_g0:.6e} c/a  (on-site)")
    print(f"    |g|_max              = {_g_max:.6e} c/a  (peak, used for estimate)")
    print(f"    g_max / |J_1|        = {_ratio_J1:.6e} a/c")
    print(f"    g_max / √|J_1|       = {_ratio_sqrtJ1:.6e}  (dimensionless)")
    print(
        f"    DIP_STRENGTH needed  = {_dip_needed:.1f} Debye  (for g_max/√|J_1| = {_TOY_G_OVER_J1})"
    )

print("-" * 50)

# ------------------------------------------------------------------------------
# Decay rate Γ(ω) at the emitter position
# ------------------------------------------------------------------------------


def compute_and_plot_decay_rate(
    omega_ks_n: np.ndarray,
    fr0_n_ks: np.ndarray,
    omega_X: float,
    r_0: list[float],
) -> float:
    """
    Compute and plot Γ(ω) at the emitter position for both dipole orientations
    in a single figure, marking the X-point frequency ω_X and the actual peak
    ω_0 of Γ(ω).

    Parameters
    ----------
    omega_ks_n : np.ndarray, shape (Nk, Nb)
        Mode frequencies in units of c/a.
    fr0_n_ks : np.ndarray, shape (Nb, Nk, 3)
        Electric field profile at the emitter position.
    omega_X : float
        X-point frequency to mark on the plot [c/a].
    r_0 : list
        Emitter position in lattice units, used in title and filename.

    Returns
    -------
    omega_0 : float
        Emitter resonance frequency, defined as the argmax of Γ(ω) [c/a].
    """
    omega_ks = omega_ks_n[:, BAND_IDX]  # shape: (Nk,)
    omegas = np.linspace(omega_ks.min(), omega_ks.max(), N_K_PATH)
    fr0_ks = fr0_n_ks[BAND_IDX, :, :]  # shape: (Nk, 3)

    gamma_list = [
        mf.compute_decay_rate(
            omegas=omegas, omega_ks=omega_ks, fr0_ks=fr0_ks, dipole_moment=dipole
        )
        for dipole in DIPOLES
    ]

    # At a high-symmetry emitter position (e.g. r_0 = [0,0]) Γ(ω) is
    # orientation-independent.  The tolerance accommodates small numerical
    # gauge noise from the MPB fields; a much larger mismatch indicates the
    # position is no longer symmetric and the single ω_0 = argmax(Γ) extracted
    # below would be ambiguous.
    _rel_diff = np.max(np.abs(gamma_list[0] - gamma_list[1])) / max(
        np.max(np.abs(gamma_list[0])), 1e-30
    )
    print(f"  Γ(ω) max rel. diff. between orientations = {_rel_diff:.2e}")
    np.testing.assert_allclose(
        gamma_list[0],
        gamma_list[1],
        rtol=1e-2,
        atol=1e-12,
        err_msg=(
            f"Γ(ω) curves for the two orientations differ at r_0={r_0}; "
            "the single ω_0 extracted from argmax(Γ) is then ambiguous."
        ),
    )

    # argmax may shift between the two curves by ≤1 grid step from numerical
    # noise; the discrepancy is then sub-grid and irrelevant for the plot.
    omega_0 = float(omegas[np.argmax(gamma_list[0])])

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    # First orientation as a solid line, second as open circle markers on top.
    # Subsampling avoids crowding the line with markers.
    ax.plot(
        omegas,
        gamma_list[0],
        color=DIP_COLORS[0],
        linewidth=1,
        label=rf"${np.degrees(THETAS[0]):.0f}$°",
    )
    _step = max(1, len(omegas) // 25)
    ax.scatter(
        omegas[::_step],
        gamma_list[1][::_step],
        facecolors="none",
        edgecolors=DIP_COLORS[1],
        s=15,
        linewidths=0.8,
        label=rf"${np.degrees(THETAS[1]):.0f}$°",
    )
    ax.axvline(x=omega_X, color="k", linestyle=":", linewidth=1)
    ax.axvline(x=omega_0, color="k", linestyle="--", linewidth=1)
    ax.legend()
    ax.set(
        xlabel=r"$\omega \, [c/a]$",
        ylabel=r"$\Gamma(\omega) \, [c/a]$",
    )
    fig.savefig(f"./figures/kimble/decay_rate_at_{r_0}.svg")
    plt.close(fig)

    return omega_0


omega_0 = compute_and_plot_decay_rate(omega_ks_n, fr0_n_ks, omega_X, R_0)
print(f"Emitter resonance (Γ peak): ω_0 = {omega_0:.6f} (c/a)")
print(f"  Δω = ω_0 - ω_X            = {omega_0 - omega_X:+.6f} (c/a)")

# ------------------------------------------------------------------------------
# Isofrequency contour in the Brillouin zone
# ------------------------------------------------------------------------------


def compute_and_plot_isofrequency(model: tb.tb_model, omega_X: float) -> None:
    """
    Compute and plot the isofrequency contour of the tight-binding band.

    Parameters
    ----------
    model : tb.tb_model
        Infinite periodic tight-binding model.
    omega_X : float
        X-point frequency to highlight as an isofrequency contour [c/a].
    """
    k_x = np.linspace(-0.5, 0.5, N_K_PATH)
    k_y = np.linspace(-0.5, 0.5, N_K_PATH)
    k_x_grid, k_y_grid = np.meshgrid(k_x, k_y, indexing="ij")

    ks_grid_flat = np.column_stack((k_x_grid.ravel(), k_y_grid.ravel()))
    tb_evals_grid = np.array(
        [np.asarray(model.solve_one(k))[BAND_IDX] for k in ks_grid_flat]
    )
    omega_grid = mf.safe_sqrt(tb_evals_grid).reshape(N_K_PATH, N_K_PATH)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE_CBAR)
    c = ax.contourf(k_x_grid, k_y_grid, omega_grid, levels=12)
    cbar = fig.colorbar(
        c, ax=ax, label=r"$\omega_\mathbf{k} a/c$", shrink=0.45, aspect=20, pad=-0.3
    )
    assert isinstance(cbar.formatter, mpl_ticker.ScalarFormatter)
    cbar.formatter.set_powerlimits((0, 0))
    ax.contour(
        k_x_grid, k_y_grid, omega_grid, levels=[omega_X], colors="k", linewidths=0.8
    )
    ax.set(
        xlabel=r"$k_x a/2\pi$",
        ylabel=r"$k_y a/2\pi$",
        aspect="equal",
    )
    fig.savefig("./figures/kimble/tb_isofrequency_contour.svg")
    plt.close(fig)


compute_and_plot_isofrequency(infinite_model, omega_X)

# ------------------------------------------------------------------------------
# Finite tight-binding model and time evolution
# ------------------------------------------------------------------------------


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    """
    Compute per-orbital damping rates for the absorbing boundary layer.

    Uses a quadratic ramp from zero at the crystal-PML interface to
    ``gamma_max`` at the hard wall.  The profile is:

        γ_j = gamma_max * ((n_pml - dist_j) / n_pml)²   if dist_j < n_pml
        γ_j = 0                                           otherwise

    where ``dist_j`` is the distance (in unit cells) of orbital j from the
    nearest edge of the crystal.

    Parameters
    ----------
    rs : np.ndarray, shape (n_orbitals, 2)
        Real-space grid points in lattice units, centered at the origin.
        Must be the same array used for the emitter-bath coupling computation
        so that orbital indices match.
    n_cells : int
        Number of unit cells per direction.
    n_pml : int
        Thickness of the absorbing layer in unit cells.
    gamma_max : float
        Maximum damping rate at the outermost cell [c/a].

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


def compute_time_evolution(
    finite_model: tb.tb_model,
    omega_ks_n: np.ndarray,
    fr0_n_ks: np.ndarray,
    ks: np.ndarray,
    rs: np.ndarray,
    omega_0: float,
    dipoles: list[np.ndarray],
    t_0: float,
    t_f: float,
    t_steps: int,
    pml_dampings: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build the light-matter Hamiltonians and compute the time evolution for an
    arbitrary list of dipole moments. Prints ``cond(V)`` per dipole as a
    diagnostic on the diagonalisation noise floor.

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
    dipoles : list of np.ndarray
        Dipole moments (each shape (3,)) defining one propagation per entry.
    t_0 : float
        Start time [a/c].
    t_f : float
        End time [a/c].
    t_steps : int
        Number of time steps.
    pml_dampings : np.ndarray of shape (n_orbitals,) or None, optional
        Per-orbital damping rates for the absorbing boundary layer [c/a].
        Passed directly to ``make_light_matter_Hamiltonian``.

    Returns
    -------
    psi_t_list : list of np.ndarray
        Time-evolved wave function for each dipole, shape (t_steps, Ndof) each.
    couplings : list of np.ndarray
        Real-space couplings for each dipole, shape (Nr,) each.
    """
    gr0_n_rs_list = [
        mf.compute_real_space_coupling(
            fr0_n_ks=fr0_n_ks,
            dipole_moment=dipole,
            omega_ks_n=omega_ks_n,
            ks=ks,
            rs=rs,
        )
        for dipole in dipoles
    ]
    couplings = [gr0_n_rs[BAND_IDX, :] for gr0_n_rs in gr0_n_rs_list]

    times = np.linspace(t_0, t_f, t_steps)
    psi_t_list: list[np.ndarray] = []
    for i, coupling in enumerate(couplings):
        idxs = np.arange(len(coupling))
        H, _ = mf.make_light_matter_Hamiltonian(
            tb_model=finite_model,
            omega_0s=omega_0,
            gr0s_rs=[(list(coupling), list(idxs))],
            pml_dampings=pml_dampings,
        )
        psi_0 = np.zeros(H.shape[0], dtype=complex)
        psi_0[0] = 1

        psi_t, cond_V = mf.evolve_diagonalisation(H, psi_0, times, return_cond=True)
        print(f"    [dipole {i}] cond(V) = {cond_V:.3e}")
        psi_t_list.append(psi_t)

    return psi_t_list, couplings


def plot_emitter_time_evolution(
    ts_low: np.ndarray,
    ts_high: np.ndarray,
    psi_t_low: np.ndarray,
    psi_t_high: np.ndarray,
    r_0: list[float],
) -> None:
    """
    Plot and save the emitter population and entanglement entropy over time
    for two dipole strengths (both at θ = 90°).  Each curve is plotted in its
    own units of Γ₀(self)⁻¹.

    Parameters
    ----------
    ts_low : np.ndarray
        Time array for the low-strength run, in units of Γ₀(low)⁻¹.
    ts_high : np.ndarray
        Time array for the high-strength run, in units of Γ₀(high)⁻¹.
    psi_t_low : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for the low-strength dipole.
    psi_t_high : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for the high-strength dipole.
    r_0 : list
        Emitter position in lattice units, used in filename.
    """
    mod_Ce_sq_low = np.abs(psi_t_low[:, 0]) ** 2
    mod_Ce_sq_high = np.abs(psi_t_high[:, 0]) ** 2

    S_t_low = mf.binary_entropy(mod_Ce_sq_low)
    S_t_high = mf.binary_entropy(mod_Ce_sq_high)

    fig, ax = plt.subplots(figsize=FIGSIZE_TIME_EVOL)
    ax.plot(
        ts_low,
        mod_Ce_sq_low,
        color="k",
        linestyle="-",
        label=rf"${DIP_STRENGTH:.0f}$ D",
    )
    _e = int(np.floor(np.log10(DIP_STRENGTH_HIGH)))
    ax.plot(
        ts_high,
        mod_Ce_sq_high,
        color="k",
        linestyle="--",
        label=rf"${DIP_STRENGTH_HIGH / 10**_e:g}\times10^{{{_e}}}$ D",
    )
    _all_pops = np.concatenate([mod_Ce_sq_low, mod_Ce_sq_high])
    _pop_min = _all_pops[_all_pops > 0].min()
    ax.set(
        xlabel=r"$t \Gamma_0$",
        ylabel=r"$|C_e(t)|^2$",
        yscale="log",
        ylim=(_pop_min, 1.5),
    )

    ax2 = ax.twinx()
    ax2.plot(ts_low, S_t_low, ls="-", color=MY_RED)
    ax2.plot(ts_high, S_t_high, ls="--", color=MY_RED)
    ax2.set_ylabel(r"$S(t)$", color=MY_RED)
    ax2.tick_params(axis="y", colors=MY_RED)
    ax2.spines["right"].set_color(MY_RED)

    ax.legend(loc="upper right")

    fig.savefig(f"./figures/kimble/emitter_timeevolution_entropy_at_{r_0}.svg")
    plt.close(fig)


def plot_quadrant_entropy(
    ts: np.ndarray,
    S_1: np.ndarray,
    S_2: np.ndarray,
    r_0: list[float],
) -> None:
    """Plot and save the quadrant directional entropy vs time for both dipole orientations."""
    fig, ax = plt.subplots(figsize=FIGSIZE_TIME_EVOL)

    ax.plot(
        ts[1:],
        S_1[1:],
        color=DIP_COLORS[0],
        lw=3,
        label=rf"${np.rad2deg(THETA_1):.0f}$°",
    )
    ax.plot(
        ts[1:],
        S_2[1:],
        color=DIP_COLORS[1],
        lw=2,
        alpha=0.9,
        label=rf"${np.rad2deg(THETA_2):.0f}$°",
    )

    trans = ax.get_yaxis_transform()
    for val, label, ls in [
        (np.log(4), r"$\ln 4$", "--"),
        (np.log(3), r"$\ln 3$", "-."),
        (np.log(2), r"$\ln 2$", ":"),
    ]:
        ax.axhline(val, color="gray", lw=1, ls=ls, zorder=0)
        ax.text(
            0.98,
            val,
            f" {label}",
            color="gray",
            fontsize=9,
            va="bottom",
            ha="right",
            transform=trans,
        )

    ax.set(
        xlabel=r"$t\Gamma_0$",
        ylabel=r"$S_{\rm dir}(t)$",
        ylim=(0, np.log(4) * 1.2),
    )
    ax.legend(loc="lower right")

    fig.savefig(f"./figures/kimble/quadrant_entropy_at_{r_0}.svg")
    plt.close(fig)


def plot_bath_snapshot(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    psi_t_1: np.ndarray,
    psi_t_2: np.ndarray,
    snap_idx: int,
    n_emitters: int,
    r_0: list[float],
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
    exp = int(np.floor(np.log10(cbar_max))) if cbar_max > 0 else 0

    for i, (theta, z) in enumerate(zip(THETAS, [z1, z2])):
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE_CBAR)
        sc = ax.pcolormesh(pos_x, pos_y, z, vmin=0, vmax=cbar_max)
        mf.draw_arrow(
            ax,
            r_0,
            theta,
            color=DIP_COLORS[i],
            length=17.5,
            arrow_offset=1.75,
            lw=0.8,
            mutation_scale=5,
        )
        pml_boundary = N_CELLS // 2 - N_PML + 0.5
        ax.add_patch(
            mpl_patches.Rectangle(
                (-pml_boundary, -pml_boundary),
                2 * pml_boundary,
                2 * pml_boundary,
                fill=False,
                edgecolor="white",
                linewidth=0.6,
                linestyle="--",
            )
        )
        cbar = fig.colorbar(
            sc,
            ax=ax,
            label=rf"$|C_\mathbf{{R}}(t)|^2 \; [\times 10^{{{exp}}}]$",
            shrink=0.6,
            aspect=20,
        )
        cbar.formatter = mpl_ticker.FuncFormatter(lambda x, _: f"{x / 10**exp:.1f}")
        cbar.update_ticks()
        ax.set(
            xlabel=r"$x/a$",
            ylabel=r"$y/a$",
            aspect="equal",
        )
        fig.savefig(
            f"./figures/kimble/bath_snapshot_at_{r_0}_theta_{np.rad2deg(theta):.0f}deg.svg"
        )
        plt.close(fig)

    return z1, z2, pos_x, pos_y


# Build finite model and its grid
finite_model = build_finite_model(infinite_model)
xs_finite, ys_finite, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)

# Free-emission rates at the two dipole strengths. Each propagation will use
# its own Γ₀ so that t·Γ₀(self) puts every curve on a comparable axis.
gamma_0: float = float(
    mf.free_spontaneous_emission(
        omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
    )
)
gamma_0_high: float = float(
    mf.free_spontaneous_emission(
        omega_0=omega_0, dip_strength=DIP_STRENGTH_HIGH, a=LAT_PARAM
    )
)
# Convert input times from 1/Γ₀ to a/c for internal use.  We keep T_F as the
# input value (in 1/Γ₀ units) and derive per-strength t_f's in a/c.
_t_conv = LAT_PARAM / C_LIGHT * gamma_0
_t_conv_high = LAT_PARAM / C_LIGHT * gamma_0_high
t_0_ac = T_0 / _t_conv
t_f_ac = T_F / _t_conv
t_f_ac_high = T_F / _t_conv_high
new_t_f_ac = NEW_T_F / _t_conv  # probe sim is at DIP_STRENGTH (50 D)

ts = np.linspace(t_0_ac, t_f_ac, T_STEPS, endpoint=True)
ts_norm = _t_conv * ts
ts_high = np.linspace(t_0_ac, t_f_ac_high, T_STEPS, endpoint=True)
ts_norm_high = _t_conv_high * ts_high

# ------------------------------------------------------------------------------
# Timescale diagnostics: compare kimble.py vs toy-model equivalent
# ------------------------------------------------------------------------------
# Bridge to toy units: 1 toy time unit (|J_1_toy|^{-1}) ↔ 1/√|J_1_real| a/c,
# because the toy bath uses ε_k while the real bath uses ω_k = √ε_k.
_toy_T_F_ac = 200.0 / np.sqrt(abs(J_1))  # toy T_F=200 converted to a/c
_dist_probe = np.linalg.norm(np.array(NEW_R) - np.array(R_0))
_v_g_char = abs(J_1) / omega_0  # characteristic group velocity [c]: |J₁|/ω₀

print("\n--- Timescale diagnostics ---")
print(f"  Γ₀ (p = {DIP_STRENGTH:.0f} D)            = {gamma_0:.4e} s⁻¹")
print(f"  Γ₀ (p = {DIP_STRENGTH_HIGH:.0f} D)          = {gamma_0_high:.4e} s⁻¹")
print(f"  t_f (kimble.py, p=50 D)     = {t_f_ac:.1f} a/c  ({T_F:.2e} × Γ₀⁻¹)")
print(f"  t_f (toy, equiv. in a/c)    ≈ {_toy_T_F_ac:.1f} a/c  (= 200 / √|J₁|)")
print(f"  Ratio t_f_kimble / t_f_toy  ≈ {t_f_ac / _toy_T_F_ac:.1f}×")
print(f"  |r_probe − r₀|              = {_dist_probe:.2f} a")
print(f"  v_g ~ |J₁| / ω₀            = {_v_g_char:.4f} c")
print(f"  Estimated travel time       ≈ {_dist_probe / _v_g_char:.1f} a/c")
print("-" * 50)

# (a) Two orientations at the default 50 D — drives the bath snapshot,
#     quadrant entropy, and supplies couplings for the probe sim.
print(f"\n--- Time evolution at {DIP_STRENGTH:.0f} D ---")
psi_t_list_50D, couplings = compute_time_evolution(
    finite_model=finite_model,
    omega_ks_n=omega_ks_n,
    fr0_n_ks=fr0_n_ks,
    ks=ks,
    rs=rs_finite,
    omega_0=omega_0,
    dipoles=DIPOLES,
    t_0=t_0_ac,
    t_f=t_f_ac,
    t_steps=T_STEPS,
    pml_dampings=pml_dampings,
)
psi_t_1, psi_t_2 = psi_t_list_50D

# (b) 90° at the high strength — drives the strength-comparison plot
#     against psi_t_1 (90° @ 50 D).  Uses its own t_f in a/c units so both
#     curves span the same range when plotted vs t·Γ₀(self).
print(f"\n--- Time evolution at {DIP_STRENGTH_HIGH:.0f} D ---")
psi_t_list_high, _ = compute_time_evolution(
    finite_model=finite_model,
    omega_ks_n=omega_ks_n,
    fr0_n_ks=fr0_n_ks,
    ks=ks,
    rs=rs_finite,
    omega_0=omega_0,
    dipoles=[DIPOLE_90_HIGH],
    t_0=t_0_ac,
    t_f=t_f_ac_high,
    t_steps=T_STEPS,
    pml_dampings=pml_dampings,
)
psi_t_high = psi_t_list_high[0]

# Plot emitter time evolution: 90° at the two strengths, each in its own Γ₀ units
plot_emitter_time_evolution(ts_norm, ts_norm_high, psi_t_1, psi_t_high, R_0)

# Get crystal positions and plot bath snapshot
# NOTE: _orb and _lat are private PythTB attributes with no public equivalent.
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

plot_quadrant_entropy(ts_norm, S_1, S_2, R_0)

# ------------------------------------------------------------------------------
# Probe emitter: directional emission and concurrence
# ------------------------------------------------------------------------------


def compute_probe_emitter_evolution(
    finite_model: tb.tb_model,
    omega_ks_n: np.ndarray,
    fr0_n_ks: np.ndarray,
    ks: np.ndarray,
    rs: np.ndarray,
    couplings: list[np.ndarray],
    omega_0: float,
    pml_dampings: np.ndarray | None = None,
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
    omega_0 : float
        Excited emitter detuning frequency [c/a]. The probe uses the same value.

    Returns
    -------
    new_psi_t_1 : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for system 1 (excited + probe).
    new_psi_t_2 : np.ndarray, shape (t_steps, Ndof)
        Time-evolved wave function for system 2 (excited + probe).
    new_omega_0 : float
        Probe emitter detuning frequency [c/a].
    """
    new_omega_0 = omega_0

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
        (list(couplings[0]), list(idxs)),
        (list(new_coupling), list(idxs)),
    ]
    new_interaction_2 = [
        (list(couplings[1]), list(idxs)),
        (list(new_coupling), list(idxs)),
    ]

    new_H_1, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=[omega_0, new_omega_0],
        gr0s_rs=new_interaction_1,
        pml_dampings=pml_dampings,
    )
    new_H_2, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=[omega_0, new_omega_0],
        gr0s_rs=new_interaction_2,
        pml_dampings=pml_dampings,
    )

    new_psi_0 = np.zeros(new_H_1.shape[0], dtype=complex)
    new_psi_0[0] = 1

    new_times = np.linspace(t_0_ac, new_t_f_ac, T_STEPS)
    new_psi_t_1 = mf.evolve_diagonalisation(new_H_1, new_psi_0, new_times)
    new_psi_t_2 = mf.evolve_diagonalisation(new_H_2, new_psi_0, new_times)

    return new_psi_t_1, new_psi_t_2, new_omega_0


def plot_probe_bath_snapshot(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    new_psi_t_1: np.ndarray,
    new_psi_t_2: np.ndarray,
    r_0: list[float],
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
    exp = int(np.floor(np.log10(cbar_max))) if cbar_max > 0 else 0

    for i, (theta, z) in enumerate(zip(THETAS, [new_z1, new_z2]), start=1):
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE_CBAR)
        sc = ax.pcolormesh(pos_x, pos_y, z, vmin=0, vmax=cbar_max)
        mf.draw_arrow(
            ax,
            NEW_R,
            NEW_THETA,
            NEW_DIP_COLOR,
            length=5,
            arrow_offset=0.5,
            lw=0.8,
            mutation_scale=5,
        )
        mf.draw_arrow(
            ax,
            r_0,
            theta,
            DIP_COLORS[i - 1],
            length=5,
            arrow_offset=0.5,
            lw=0.8,
            mutation_scale=5,
        )
        pml_boundary = N_CELLS // 2 - N_PML + 0.5
        ax.add_patch(
            mpl_patches.Rectangle(
                (-pml_boundary, -pml_boundary),
                2 * pml_boundary,
                2 * pml_boundary,
                fill=False,
                edgecolor="white",
                linewidth=0.6,
                linestyle="--",
            )
        )
        cbar = fig.colorbar(
            sc,
            ax=ax,
            label=rf"$|C_\mathbf{{r}}(t)|^2 \; [\times 10^{{{exp}}}]$",
            shrink=0.6,
            aspect=20,
        )
        cbar.formatter = mpl_ticker.FuncFormatter(lambda x, _: f"{x / 10**exp:.1f}")
        cbar.update_ticks()
        ax.set(
            xlabel=r"$x/a$",
            ylabel=r"$y/a$",
            aspect="equal",
            xlim=(-8, 8),
            ylim=(-8, 8),
        )
        fig.savefig(
            f"./figures/kimble/bath_snapshot_at_{r_0}_theta_{np.rad2deg(theta):.0f}deg_with_probe.svg"
        )
        plt.close(fig)


def plot_probe_time_evolution(
    ts: np.ndarray,
    new_psi_t_1: np.ndarray,
    new_psi_t_2: np.ndarray,
    r_0: list[float],
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
        fig, ax = plt.subplots(figsize=FIGSIZE_TIME_EVOL)
        ax.plot(ts, mod_Ce_excited_sq[i], c=DIP_COLORS[i])
        ax.plot(ts, mod_Ce_probe_sq[i], c=NEW_DIP_COLOR)

        ax2 = ax.twinx()
        ax2.plot(ts, C_ts[i], c=MY_RED, ls="--")

        _all_pops = mod_Ce_excited_sq[i]
        _pop_min = _all_pops[_all_pops > 0].min()
        ax.set(
            xlabel=r"$t\Gamma_0$",
            ylabel=r"$|C_e(t)|^2$",
            yscale="log",
            ylim=(_pop_min, 1.5),
        )
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
        plt.close(fig)


# Run probe emitter computation
new_psi_t_1, new_psi_t_2, new_omega_0 = compute_probe_emitter_evolution(
    finite_model=finite_model,
    omega_ks_n=omega_ks_n,
    fr0_n_ks=fr0_n_ks,
    ks=ks,
    rs=rs_finite,
    couplings=couplings,
    omega_0=omega_0,
    pml_dampings=pml_dampings,
)

new_ts = np.linspace(t_0_ac, new_t_f_ac, T_STEPS, endpoint=True)
new_ts_norm = _t_conv * new_ts

plot_probe_bath_snapshot(pos_x, pos_y, new_psi_t_1, new_psi_t_2, R_0)
plot_probe_time_evolution(new_ts_norm, new_psi_t_1, new_psi_t_2, R_0)
