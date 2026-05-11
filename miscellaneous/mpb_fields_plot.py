"""
Visualize MPB electric fields at an emitter position in reciprocal and real space.

Three sections:
  1. Reciprocal space — amplitude and phase of Ex, Ey as scatter plots over k;
     total field amplitude; real/imaginary vector field quiver plots.
  2. Real space — discrete inverse Bloch transform via `compute_fr0rs` to obtain
     E_{n,R}(r_0); real/imaginary quiver plots over lattice translations.
  3. Unit-cell profile — field structure inside the first unit cell summed over k.

Run from the `miscellaneous/` directory. Requires precomputed MPB fields in
`data/kimble/fields/`. If absent, run `mpb_field_computation.py` first.
"""

# some imports for plotting utilities
import warnings

import matplotlib.pyplot as plt
import numpy as np

import emitter_crystal_coupling as mf  # our package functions to work with

# %% Define the main constants and parameters for the current simulation
NK = 81  # number of points per direction for the Monkhorst-Pack grid

# Position of the quantum emitter in units of the lattice constant 'a'
# list of positions are also allowed
R0 = [0, 0]

BAND_INDEX = 0  # index of the band to consider (0-based)

# ---------------------------------------------------------------------------- #
# Compute E_r(r0)
# ---------------------------------------------------------------------------- #

# Load the fields and k-points from the MPB calculation
# This file was generated previously
data = mf.load(f"../data/kimble/fields/{NK}_meshgrid_at_{R0}.pkl")

# Extract relevant quantities
Eks = np.asarray(data["Eks"])  # Electric fields, shape: (Nr, Nb, Nk, 3)
kpts = np.asarray(data["kpts"])  # k-points in cartesian coordinates, shape: (Nk, 3)

# take the field in the position r0 from the possible list
# NOTE: For the previously specified data, only one point was computed and 2-bands
E_ks_r0 = Eks[0, :, :, :]  # Shape: (Nb, Nk, 3)
E_ks_r0_band = E_ks_r0[BAND_INDEX, :, :]

# ------------------------------------------------------------------------------
# Compute the phase and amplitude of the `x` and `y` components of the electric
# field and plot it as a function of `k`. Also check that the `z`-component is
# almost zero (TE polarization).
# ------------------------------------------------------------------------------
amplitudes = np.abs(E_ks_r0_band)
phases = np.angle(E_ks_r0_band)

if not np.allclose(amplitudes[:, 2], 0):
    raise ValueError(
        "The z-component of the field is not zero. Not considered in this script."
    )

# remove last column
amplitudes = np.delete(amplitudes, -1, axis=1)
phases = np.delete(phases, -1, axis=1)

fig, axs = plt.subplots(2, 2)

x = kpts[:, 0]
y = kpts[:, 1]

for i, (amplitude, phase) in enumerate(zip(amplitudes.T, phases.T)):
    axv = axs[i, :]
    sc1 = axv[0].scatter(x, y, c=amplitude, marker="s", s=0.01 * mf.marker_size(len(x)))
    sc2 = axv[1].scatter(x, y, c=phase, marker="s", s=0.01 * mf.marker_size(len(x)))

    fig.colorbar(sc1, ax=axv[0])
    fig.colorbar(sc2, ax=axv[1])

    for ax in axv:
        ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", aspect="equal")

axs[0, 0].set_title(r"$|E_{x,\mathbf{k}}(\mathbf{r}_0)|$")
axs[0, 1].set_title(r"$\arg(E_{x,\mathbf{k}}(\mathbf{r}_0))$")
axs[1, 0].set_title(r"$|E_{y,\mathbf{k}}(\mathbf{r}_0)|$")
axs[1, 1].set_title(r"$\arg(E_{y,\mathbf{k}}(\mathbf{r}_0))$")

fig.tight_layout(h_pad=2)
plt.show()
fig.savefig(
    f"../figures/miscellaneous/{R0}_phase_amplitude_fields_reciprocal_space_.png"
)

# ------------------------------------------------------------------------------
# plot the total electric field amplitude instead of each components
# ------------------------------------------------------------------------------
total_amplitudes = np.linalg.norm(E_ks_r0_band, axis=1)

fig, ax = plt.subplots()

sc = ax.scatter(x, y, c=total_amplitudes, marker="s", s=mf.marker_size(len(x)))

fig.colorbar(sc, ax=ax)

ax.set(
    title=r"$|\mathbf{E}_\mathbf{k}(\mathbf{r}_0)|$",
    xlabel=r"$k_x$",
    ylabel=r"$k_y$",
    aspect="equal",
)

fig.tight_layout()
plt.show()
fig.savefig(f"../figures/miscellaneous/{R0}_total_amplitude_field_reciprocal_space.png")

# ------------------------------------------------------------------------------
# Plot the electric field profile at r₀ in `k` space as a vector field
# ------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2)

axs[0].quiver(x, y, np.real(E_ks_r0_band[:, 0]), np.real(E_ks_r0_band[:, 1]), color="r")
axs[1].quiver(x, y, np.imag(E_ks_r0_band[:, 0]), np.imag(E_ks_r0_band[:, 1]), color="b")

for ax in axs:
    ax.grid()
    ax.set(
        xlabel=r"$k_x$",
        ylabel=r"$k_y$",
        aspect="equal",
    )
axs[0].set_title(r"$\Re(\mathbf{E}_\mathbf{k}(\mathbf{r}_0))$")
axs[1].set_title(r"$\Im(\mathbf{E}_\mathbf{k}(\mathbf{r}_0))$")

fig.tight_layout()
plt.show()
fig.savefig(f"../figures/miscellaneous/{R0}_field_reciprocal_space.png")

# ---------------------------------------------------------------------------- #
# Define the real-space grid. Here you must pick a set of lattice translations
# ---------------------------------------------------------------------------- #
num_r = 7
xjg = np.arange(-(num_r // 2) + R0[0] // 1, num_r // 2 + 1 + R0[0] // 1, 1)
yjg = np.arange(-(num_r // 2) + R0[1] // 1, num_r // 2 + 1 + R0[1] // 1, 1)
rjpts = mf.meshgrid_points(xjg, yjg, indexing="xy")


def pad_to_3d(array: np.ndarray) -> np.ndarray:
    array = np.atleast_2d(array)
    if array.shape[1] < 3:
        pad = np.zeros((array.shape[0], 3 - array.shape[1]))
        array = np.hstack([array, pad])
    return array


rjpts = pad_to_3d(rjpts)

# ---------------------------------------------------------------------------- #
# Compute E_r(r0), shape (Nb, Nr, 3)
# ---------------------------------------------------------------------------- #
E_rjs_r0 = mf.compute_fr0rs(E_ks_r0, kpts, rjpts)

if not np.allclose(E_rjs_r0[:, :, 2], 0):
    raise ValueError(
        "The z-component of the field is not zero. Not considered in this script."
    )

E_rjs_r0_band = E_rjs_r0[BAND_INDEX]

# ------------------------------------------------------------------------------
# Symmetry check: along the y-axis (x = 0), E_y should vanish. The residual
# is a discretization-limited artefact (scales like 1/RESOLUTION²), so warn
# rather than abort.
# ------------------------------------------------------------------------------
on_y_axis = np.isclose(rjpts[:, 0], 0)
if on_y_axis.any():
    Ey_on_y_axis = E_rjs_r0_band[on_y_axis, 1]
    if not np.allclose(Ey_on_y_axis, 0):
        warnings.warn(
            f"E_y is not zero along the y-axis: "
            f"max |E_y| = {np.max(np.abs(Ey_on_y_axis)):.3e}.",
            stacklevel=2,
        )

# ------------------------------------------------------------------------------
# Plot E_R(r0) in real space as a vector field
# ------------------------------------------------------------------------------
# Shared scale so Re and Im subplots are visually comparable. With
# scale_units="xy", a max-magnitude arrow has length ≈ 1 lattice unit.
arrow_scale = max(
    np.abs(np.real(E_rjs_r0_band[:, :2])).max(),
    np.abs(np.imag(E_rjs_r0_band[:, :2])).max(),
)

fig, axs = plt.subplots(1, 2)

axs[0].quiver(
    rjpts[:, 0],
    rjpts[:, 1],
    np.real(E_rjs_r0_band[:, 0]),
    np.real(E_rjs_r0_band[:, 1]),
    color="r",
    angles="xy",
    scale_units="xy",
    scale=arrow_scale,
)
axs[1].quiver(
    rjpts[:, 0],
    rjpts[:, 1],
    np.imag(E_rjs_r0_band[:, 0]),
    np.imag(E_rjs_r0_band[:, 1]),
    color="r",
    angles="xy",
    scale_units="xy",
    scale=arrow_scale,
)

for ax in axs:
    ax.grid()
    ax.set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
    )
axs[0].set_title(r"$\Re(\mathbf{E}_\mathbf{R}(\mathbf{r}_0))$")
axs[1].set_title(r"$\Im(\mathbf{E}_\mathbf{R}(\mathbf{r}_0))$")

fig.tight_layout()
plt.show()
fig.savefig(f"../figures/miscellaneous/{R0}_fields_real_space.png")
