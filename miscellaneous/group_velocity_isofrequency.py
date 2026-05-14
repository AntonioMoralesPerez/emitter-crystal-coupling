"""
Group-velocity direction on the isofrequency contour omega_k = omega_0.

Bath dispersion (square TB lattice, NN + NNN, frequencies^2):
    eps_k = J_0 + 2 J_1 (cos kx + cos ky) + 4 J_2 cos kx cos ky
    omega_k = sqrt(eps_k)

The group velocity is v_g = grad_k omega_k = grad_k eps_k / (2 omega_k).

For an emitter resonant at the X-point (saddle point), this script extracts
all k-points on the isofrequency contour omega_k = omega_0, computes the
group-velocity direction at each, and plots its angle vs. the polar angle
of k. The diagonal y = x line marks isotropic propagation; deviations show
how strongly the emission direction differs from the k direction.
"""

import matplotlib.pyplot as plt
import numpy as np

# %% Parameters

J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

N_K: int = 601  # BZ grid resolution for contour extraction


def eps_k(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    return J_0 + 2 * J_1 * (np.cos(kx) + np.cos(ky)) + 4 * J_2 * np.cos(kx) * np.cos(ky)


def omega_k(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    return np.sqrt(eps_k(kx, ky))


def grad_omega(kx: np.ndarray, ky: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eps = eps_k(kx, ky)
    deps_dkx = -2 * J_1 * np.sin(kx) - 4 * J_2 * np.sin(kx) * np.cos(ky)
    deps_dky = -2 * J_1 * np.sin(ky) - 4 * J_2 * np.cos(kx) * np.sin(ky)
    omega = np.sqrt(eps)
    return deps_dkx / (2 * omega), deps_dky / (2 * omega)


# %% Resonance at the X-point

OMEGA_0: float = float(omega_k(np.array(0.0), np.array(np.pi)))
print(f"omega_0 = {OMEGA_0:.6f} c/a (at X-point)")

# %% Extract isofrequency contour

kx_grid = np.linspace(-np.pi, np.pi, N_K)
ky_grid = np.linspace(-np.pi, np.pi, N_K)
KX, KY = np.meshgrid(kx_grid, ky_grid, indexing="xy")
OMEGA = omega_k(KX, KY)

# Use a hidden contour to obtain (kx, ky) points on omega = omega_0.
_fig, _ax = plt.subplots()
cs = _ax.contour(KX, KY, OMEGA, levels=[OMEGA_0])
plt.close(_fig)

segments = cs.allsegs[0]
contour_pts = np.vstack(segments)
kx_c = contour_pts[:, 0]
ky_c = contour_pts[:, 1]

# Drop the BZ-boundary duplicates (|kx| = pi or |ky| = pi) for a cleaner plot.
eps_bz = 1e-3
mask = (np.abs(kx_c) < np.pi - eps_bz) & (np.abs(ky_c) < np.pi - eps_bz)
kx_c = kx_c[mask]
ky_c = ky_c[mask]

# %% Group velocity at contour points

vgx, vgy = grad_omega(kx_c, ky_c)

theta_k = np.arctan2(ky_c, kx_c)
theta_vg = np.arctan2(vgy, vgx)

# %% Plot: theta_v_g vs theta_k

_ticks = np.array([-1, -3/4, -1/2, -1/4, 0, 1/4, 1/2, 3/4, 1]) * np.pi
_tick_labels = [
    r"$-\pi$", r"$-3\pi/4$", r"$-\pi/2$", r"$-\pi/4$",
    "$0$",
    r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$",
]

fig, ax = plt.subplots(figsize=(6.2, 5.2))
sc = ax.scatter(
    theta_k, theta_vg, c=np.hypot(kx_c, ky_c), cmap="viridis", s=6
)
ax.plot([-np.pi, np.pi], [-np.pi, np.pi], "k:", lw=0.8, label=r"$\theta_{v_g} = \theta_k$")
ax.set(
    xlabel=r"$\theta_{\mathbf{k}}$ (angle of $\mathbf{k}$)",
    ylabel=r"$\theta_{\mathbf{v}_g}$ (angle of $\mathbf{v}_g$)",
    title=rf"Group velocity direction on $\omega_{{\mathbf{{k}}}} = \omega_0$ "
    rf"({OMEGA_0:.4f} $c/a$)",
    xticks=_ticks,
    yticks=_ticks,
)
ax.set_xticklabels(_tick_labels)
ax.set_yticklabels(_tick_labels)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(r"$|\mathbf{k}|\, a$")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(
    "../figures/miscellaneous/group_velocity_angle_isofrequency.png",
    bbox_inches="tight",
)
plt.show()

# %% Plot: contour and v_g arrows in the BZ

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.contour(KX, KY, OMEGA, levels=[OMEGA_0], colors="k", linewidths=1.3)
n_sub = max(1, len(kx_c) // 60)
ax2.quiver(
    kx_c[::n_sub],
    ky_c[::n_sub],
    vgx[::n_sub],
    vgy[::n_sub],
    color="C0",
    angles="xy",
    width=0.004,
)
ax2.set(
    xlabel=r"$k_x\, a$",
    ylabel=r"$k_y\, a$",
    title=r"Isofrequency contour at $\omega_0$ with $\mathbf{v}_g$ arrows",
    xlim=(-np.pi, np.pi),
    ylim=(-np.pi, np.pi),
    xticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    yticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    aspect="equal",
)
ax2.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
ax2.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(
    "../figures/miscellaneous/group_velocity_isofrequency_BZ.png",
    bbox_inches="tight",
)
plt.show()
