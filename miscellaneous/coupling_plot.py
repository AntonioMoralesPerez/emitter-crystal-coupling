"""
    Docstring for coupling_plot.py

This file is used for plotting the coupling in real space and in reciprocal
(k-) space for the already computed fields at such position. The main purpose
is to see where the maximum coupling strength is located in real space and over
the first Brillouin zone, as well as its overall shape.

NOTE: use the file `mpb_field_computation.py` for obtaining the fields before
executing this file.
"""

# some imports for plotting utilities
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np

import emitter_crystal_coupling as mf  # our package functions to work with
from emitter_crystal_coupling.constants import K_CONST

# %% Define the main constants and parameters for the current simulation
NK = 81  # number of points per direction for the Monkhorst-Pack grid
RESOLUTION = 64  # must match the precomputed field files

# Dipole orientations (matches the convention in kimble.py)
THETA_1 = np.deg2rad(90)
THETA_2 = np.deg2rad(135)
THETAS: list[float] = [THETA_1, THETA_2]

LAT_PARAM = 300e-9  # lattice parameter [m]
DIP_STRENGTH = 50  # [D]
P_DIMLESS = DIP_STRENGTH / mf.unit_dip(a=LAT_PARAM)

DIPOLES: list[np.ndarray] = [
    P_DIMLESS * np.array([np.cos(theta), np.sin(theta), 0]) for theta in THETAS
]

# Position of the quantum emitter in units of the lattice constant 'a'
R0 = [0, 0]

BAND_INDEX = 0  # index of the band to consider (0-based)

# ---------------------------------------------------------------------------- #
# Load the precomputed MPB fields
# ---------------------------------------------------------------------------- #
data = mf.load(f"../data/kimble/fields/res_{RESOLUTION}/nk_{NK}/meshgrid_at_{R0}.pkl")

omegas = 2 * np.pi * np.asarray(data["freqs"])  # Frequencies (Nk, Nb)
Eks = np.asarray(data["Eks"])  # Electric fields, shape: (Nr, Nb, Nk, 3)
kpts = np.asarray(data["kpts"])  # k-points in cartesian coordinates, shape: (Nk, 3)

# take the field at position r0 from the possible list. Here, only one point
# was computed
E_k_r0 = Eks[0]  # Shape: (Nb, Nk, 3)
n_kpoints = kpts.shape[0]

# ---------------------------------------------------------------------------- #
# Real-space grid for interpolation
# ---------------------------------------------------------------------------- #
num_r = 10
xjg = np.arange(
    -(num_r // 2) + int(np.floor(R0[0])), num_r // 2 + 1 + int(np.floor(R0[0])), 1
)
yjg = np.arange(
    -(num_r // 2) + int(np.floor(R0[1])), num_r // 2 + 1 + int(np.floor(R0[1])), 1
)
rjpts = mf.meshgrid_points(xjg, yjg, indexing="xy")

# ---------------------------------------------------------------------------- #
# Compute g_r(r0) and g_k(r0) for each dipole orientation
# ---------------------------------------------------------------------------- #
gr0_per_theta: list[np.ndarray] = []
gk0_per_theta: list[np.ndarray] = []

for dipole in DIPOLES:
    # Real-space coupling g_{n,r}(r0), shape (Nr,)
    gr0 = mf.compute_real_space_coupling(E_k_r0, dipole, omegas, kpts, rjpts)[
        BAND_INDEX, :
    ]
    gr0_per_theta.append(gr0)

    # Reciprocal-space coupling g_{n,k}(r0), shape (Nk,)
    # g_{n,k}(r_0) = -i * K_CONST * sqrt(ω_{n,k} / (2 N_k)) * (p · f_{n,k}(r_0))
    dipole_dot_field = np.einsum("i,ki->k", dipole, E_k_r0[BAND_INDEX])
    prefactor = -1j * K_CONST * np.sqrt(omegas[:, BAND_INDEX] / (2 * n_kpoints))
    gk0_per_theta.append(prefactor * dipole_dot_field)

# Shared colour scales across orientations (matches kimble.py convention)
cbar_max_r = max(np.abs(g).max() for g in gr0_per_theta)
cbar_max_k = max(np.abs(g).max() for g in gk0_per_theta)

# ---------------------------------------------------------------------------- #
# Plot |g_r(r0)| in real space for each orientation
# ---------------------------------------------------------------------------- #
arrow_length = 2

for theta, gr0rjs in zip(THETAS, gr0_per_theta):
    fig, ax = plt.subplots()
    sc = ax.scatter(
        rjpts[:, 0],
        rjpts[:, 1],
        c=np.abs(gr0rjs),
        vmin=0,
        vmax=cbar_max_r,
        marker="s",
        s=0.71 * mf.marker_size(len(rjpts)),
    )
    mf.draw_arrow(ax, R0, theta, length=arrow_length)
    ax.set(
        title=rf"$\theta = {np.rad2deg(theta):.0f}º$",
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
    )
    cbar = fig.colorbar(sc, ax=ax, label=r"$|g_\mathbf{r}(\mathbf{r}_0)|$")
    assert isinstance(cbar.formatter, mpl_ticker.ScalarFormatter)
    cbar.formatter.set_powerlimits((0, 0))

    fig.tight_layout()
    fig.savefig(
        f"../figures/miscellaneous/{R0}_coupling_angle_{np.rad2deg(theta):.0f}.png"
    )

# ---------------------------------------------------------------------------- #
# Plot |g_k(r0)| over the first Brillouin zone for each orientation
# ---------------------------------------------------------------------------- #
# kpts are stored in cartesian with the 2π factor absorbed (units of 1/a);
# divide by 2π so axes show k a / (2π), matching the isofrequency plot convention.
kx_plot = kpts[:, 0] / (2 * np.pi)
ky_plot = kpts[:, 1] / (2 * np.pi)

for theta, gk0 in zip(THETAS, gk0_per_theta):
    fig_k, ax_k = plt.subplots()
    sc_k = ax_k.scatter(
        kx_plot,
        ky_plot,
        c=np.abs(gk0),
        vmin=0,
        vmax=cbar_max_k,
        marker="s",
        s=0.71 * mf.marker_size(n_kpoints),
    )
    ax_k.set(
        title=rf"$\theta = {np.rad2deg(theta):.0f}º$",
        xlabel=r"$k_x\,a/2\pi$",
        ylabel=r"$k_y\,a/2\pi$",
        aspect="equal",
    )
    cbar_k = fig_k.colorbar(sc_k, ax=ax_k, label=r"$|g_\mathbf{k}(\mathbf{r}_0)|$")
    assert isinstance(cbar_k.formatter, mpl_ticker.ScalarFormatter)
    cbar_k.formatter.set_powerlimits((0, 0))

    fig_k.tight_layout()
    fig_k.savefig(
        f"../figures/miscellaneous/{R0}_coupling_kspace_angle_{np.rad2deg(theta):.0f}.png"
    )

plt.show()
