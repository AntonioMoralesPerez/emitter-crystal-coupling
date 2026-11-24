"""
Full-range probe orientation sweep at PROBE_R = [-5, 5].

Sweeps the probe orientation over the full physical range 0 - 180 deg
(dipoles are line-symmetric so this covers all distinct orientations).

Goal: locate both the maximum and the minimum, and check whether they
are exactly 90 deg apart. If they are, the response is a pure
|d_probe . E_eff(r_probe)|^2 polarisation rotation, which means the
~48 deg shift from the naive 45 deg expectation is genuine Bloch-mode
polarisation. If max and min deviate from 90 deg apart, additional
non-trivial structure is present.

NOTE: run mpb_field_computation.py with R0_LIST including [0, 0] and
      [-5, 5] and NK = 81 before executing.
"""

import matplotlib.pyplot as plt
import numpy as np
import pythtb as tb

import emitter_crystal_coupling as mf
from emitter_crystal_coupling.constants import C_LIGHT

# %% Parameters

LAT_PARAM: float = 300e-9
DIP_STRENGTH: float = 7000
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A

BAND_IDX: int = 0
N_K: int = 81
RESOLUTION: int = 64  # must match the precomputed field files

EXCITED_R: list = [0, 0]
EXCITED_THETA: float = 90 * np.pi / 180
EXCITED_DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array(
    [np.cos(EXCITED_THETA), np.sin(EXCITED_THETA), 0]
)

PROBE_R: list = [-5, 5]
PROBE_DIP_STRENGTH: float = DIP_STRENGTH
N_THETAS: int = 91  # 2 deg resolution
PROBE_THETAS: np.ndarray = np.linspace(0, 180, N_THETAS) * np.pi / 180

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 41
N_PML: int = 8
GAMMA_MAX: float = 2 * np.pi * 3e-2

J_0: float = 3.06180355
J_1: float = -0.78708042
J_2: float = 0.02130492

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
N_K_PATH: int = 100

T_F: float = (2 * np.pi) ** 2 * 1e-2
T_STEPS: int = 100


# %% Helper


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    half = n_cells // 2
    dist = np.minimum(half - np.abs(rs[:, 0]), half - np.abs(rs[:, 1]))
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


# %% TB model

infinite_model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
infinite_model.set_onsite([J_0])
for vec in [[1, 0], [0, 1]]:
    infinite_model.set_hop(J_1, 0, 0, vec)
for vec in [[1, 1], [1, -1]]:
    infinite_model.set_hop(J_2, 0, 0, vec)

k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals = infinite_model.solve_all(k_vec)
omega_0 = mf.safe_sqrt(tb_evals[BAND_IDX][np.isclose(k_dist, k_node[1])]).item()
print(f"omega_0 = {omega_0:.6f} c/a")

temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)
_, _, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)
indexes = np.arange(len(rs_finite))

# %% Time conversion

gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / C_LIGHT * gamma_0
t_f_internal = T_F / _t_conv

# %% Load fields

data_excited = mf.load(
    f"../data/kimble/fields/res_{RESOLUTION}/nk_{N_K}/meshgrid_at_{EXCITED_R}.pkl"
)
omega_ks_n = 2 * np.pi * np.asarray(data_excited["freqs"])
fr0_n_ks = np.asarray(data_excited["Eks"])[0]
ks = np.asarray(data_excited["kpts"])

excited_coupling = mf.compute_real_space_coupling(
    fr0_n_ks=fr0_n_ks,
    dipole_moment=EXCITED_DIPOLE,
    omega_ks_n=omega_ks_n,
    ks=ks,
    rs=rs_finite,
)[BAND_IDX, :]

data_probe = mf.load(
    f"../data/kimble/fields/res_{RESOLUTION}/nk_{N_K}/meshgrid_at_{PROBE_R}.pkl"
)
omega_ks_probe = 2 * np.pi * np.asarray(data_probe["freqs"])
fr0_probe = np.asarray(data_probe["Eks"])[0]
ks_probe = np.asarray(data_probe["kpts"])

# %% Sweep

max_probe_pops = np.zeros(N_THETAS)

print(f"\n--- full-range sweep at {PROBE_R}, omega_0 = {omega_0:.6f} c/a ---")
for i, probe_theta in enumerate(PROBE_THETAS):
    probe_dipole = (PROBE_DIP_STRENGTH / P_A) * np.array(
        [np.cos(probe_theta), np.sin(probe_theta), 0]
    )
    probe_coupling = mf.compute_real_space_coupling(
        fr0_n_ks=fr0_probe,
        dipole_moment=probe_dipole,
        omega_ks_n=omega_ks_probe,
        ks=ks_probe,
        rs=rs_finite,
    )[BAND_IDX, :]

    H, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=[omega_0, omega_0],
        gr0s_rs=[
            (list(excited_coupling), list(indexes)),
            (list(probe_coupling), list(indexes)),
        ],
        pml_dampings=pml_dampings,
    )

    psi0 = np.zeros(H.shape[0])
    psi0[0] = 1
    psit = mf.evolve_diagonalisation(H, psi0, np.linspace(0.0, t_f_internal, T_STEPS))
    max_probe_pops[i] = np.max(np.abs(psit[:, 1]) ** 2)

# %% Locate max / min and check 90 deg separation

max_idx = int(np.argmax(max_probe_pops))
min_idx = int(np.argmin(max_probe_pops))
theta_max = np.rad2deg(PROBE_THETAS[max_idx])
theta_min = np.rad2deg(PROBE_THETAS[min_idx])
sep = (theta_min - theta_max) % 180.0
sep = min(sep, 180.0 - sep)

print(
    f"\nMax at theta = {theta_max:.1f} deg  (max pop = {max_probe_pops[max_idx]:.4e})"
)
print(f"Min at theta = {theta_min:.1f} deg  (min pop = {max_probe_pops[min_idx]:.4e})")
print(f"|theta_max - theta_min| (mod 180) = {sep:.2f} deg  (expect 90 deg)")

# %% Plot

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.rad2deg(PROBE_THETAS), max_probe_pops, "o-", color="k", lw=1.2, ms=3)
ax.plot(
    theta_max,
    max_probe_pops[max_idx],
    "*",
    color="r",
    ms=11,
    label=rf"max at $\theta = {theta_max:.1f}$°",
)
ax.plot(
    theta_min,
    max_probe_pops[min_idx],
    "s",
    color="b",
    ms=7,
    label=rf"min at $\theta = {theta_min:.1f}$°",
)
ax.axvline(45, color="gray", ls=":", lw=1, label=r"naive 45°")
ax.axvline(135, color="gray", ls=":", lw=1)
ax.set(
    xlabel=r"$\theta_{\rm probe}$ [deg]",
    ylabel=r"$\max_t \, |C_{\rm probe}(t)|^2$",
    title=(
        rf"full-range sweep at {PROBE_R}, $\omega = \omega_0$ ({omega_0:.4f} $c/a$), "
        rf"$\theta_{{\rm exc}} = {np.rad2deg(EXCITED_THETA):.0f}$°"
        + "\n"
        + rf"$|\theta_{{\max}} - \theta_{{\min}}|$ (mod 180) = {sep:.2f}°"
    ),
)
ax.legend()
fig.tight_layout()
fig.savefig(
    f"../figures/miscellaneous/probe_orientation_sweep_full_range_at_{PROBE_R}.png",
    bbox_inches="tight",
)
plt.show()
