"""
Probe orientation sweep at increasing diagonal distances.

Sweeps the probe orientation around 45 deg at probe positions
[-15, 15], [-20, 20], [-25, 25], [-30, 30] (all on the same y = -x
diagonal ray from the source [0, 0], theta_exc = 90 deg).

Goal: distinguish two scenarios for the optimum-orientation shift away
from 45 deg:
  (A) Finite-distance correction. The shift relaxes towards 45 deg as
      |r_probe| grows: the field at the probe is asymptoting to a
      saddle-point Bloch mode whose polarisation differs from the
      naive geometric direction at short distances but rotates / decays
      towards a clean limit far from the source.
  (B) Bath-mode polarisation (multi-saddle interference). The shift is
      essentially a property of the photonic Bloch modes near X / X'
      and is largely distance-independent.

A clear monotonic trend with |r_probe| favours (A); a roughly
distance-independent angle favours (B).

NOTE: run mpb_field_computation.py with R0_LIST including [0, 0] and
      every probe position below, at NK = 81, before executing.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pythtb as tb
from scipy.sparse.linalg import expm_multiply

import emitter_crystal_coupling as mf
from emitter_crystal_coupling.constants import C_LIGHT

# %% Parameters

LAT_PARAM: float = 300e-9
DIP_STRENGTH: float = 7000
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A

BAND_IDX: int = 0
N_K: int = 81

EXCITED_R: list = [0, 0]
EXCITED_THETA: float = 90 * np.pi / 180
EXCITED_DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array(
    [np.cos(EXCITED_THETA), np.sin(EXCITED_THETA), 0]
)

PROBE_RS: list = [[-15, 15], [-20, 20], [-25, 25], [-30, 30]]
PROBE_DIP_STRENGTH: float = DIP_STRENGTH

PROBE_THETAS: np.ndarray = np.arange(43.0, 47.0 + 1e-9, 0.5) * np.pi / 180
N_THETAS: int = len(PROBE_THETAS)

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 101  # half-width 50: r_probe = [-30, 30] sits 12 cells past PML edge (~0.5 lambda)
N_PML: int = 8
GAMMA_MAX: float = 3e-2

J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
N_K_PATH: int = 100

T_F: float = 3e-2  # [1/Gamma_0] — generous margin for peak settling at r_probe up to [-30, 30]
T_STEPS: int = 120


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

# Sanity check: every probe (and the source) must sit strictly outside the PML.
_half = N_CELLS // 2
for _r in [EXCITED_R, *PROBE_RS]:
    _margin = _half - max(abs(_r[0]), abs(_r[1]))
    assert _margin > N_PML, (
        f"Position {_r} is inside (or too close to) the PML: "
        f"margin {_margin} <= N_PML {N_PML}. Increase N_CELLS."
    )

# %% Time conversion

gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / C_LIGHT * gamma_0  # [a/c] per [1/Gamma_0]
t_f_internal = T_F / _t_conv

# %% Excited-emitter coupling (computed once)

data_excited = mf.load(f"../data/kimble/fields/{N_K}_meshgrid_at_{EXCITED_R}.pkl")
omega_ks_n = np.asarray(data_excited["freqs"])
fr0_n_ks = np.asarray(data_excited["Eks"])[0]
ks = np.asarray(data_excited["kpts"])

excited_coupling = mf.compute_real_space_coupling(
    fr0_n_ks=fr0_n_ks,
    dipole_moment=EXCITED_DIPOLE,
    omega_ks_n=omega_ks_n,
    ks=ks,
    rs=rs_finite,
)[BAND_IDX, :]

# %% Sweep over distances and orientations

results: dict = {}

for probe_r in PROBE_RS:
    dist = float(np.linalg.norm(probe_r))

    print(f"\n--- probe at {probe_r} (|r| = {dist:.2f}), T_F = {T_F:.3e} 1/Gamma_0 ---")

    data_probe = mf.load(f"../data/kimble/fields/{N_K}_meshgrid_at_{probe_r}.pkl")
    omega_ks_probe = np.asarray(data_probe["freqs"])
    fr0_probe = np.asarray(data_probe["Eks"])[0]
    ks_probe = np.asarray(data_probe["kpts"])

    max_pops = np.zeros(N_THETAS)
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

        assert H.shape is not None
        psi0 = np.zeros(H.shape[0])
        psi0[0] = 1
        psit = expm_multiply(-1j * H, psi0, 0.0, t_f_internal, T_STEPS, endpoint=True)
        max_pops[i] = np.max(np.abs(psit[:, 1]) ** 2)

    max_idx = int(np.argmax(max_pops))
    theta_max = float(np.rad2deg(PROBE_THETAS[max_idx]))
    results[tuple(probe_r)] = {
        "dist": dist,
        "max_pops": max_pops,
        "theta_max": theta_max,
        "peak": float(max_pops[max_idx]),
    }
    print(f"  theta_max = {theta_max:.2f} deg, peak pop = {max_pops[max_idx]:.4e}")

# %% Plot: sweep curves

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = mpl.colormaps["viridis"](np.linspace(0.15, 0.85, len(PROBE_RS)))

for color, probe_r in zip(colors, PROBE_RS):
    res = results[tuple(probe_r)]
    norm_pops = res["max_pops"] / res["peak"]
    ax.plot(
        np.rad2deg(PROBE_THETAS),
        norm_pops,
        "o-",
        color=color,
        lw=1.4,
        ms=3.5,
        label=rf"{probe_r}, $\theta_{{\max}} = {res['theta_max']:.1f}$°",
    )

ax.axvline(45, color="gray", ls=":", lw=1, label="naive 45°")
ax.set(
    xlabel=r"$\theta_{\rm probe}$ [deg]",
    ylabel=r"$\max_t |C_{\rm probe}(t)|^2$ (normalised)",
    title=(
        rf"Probe orientation sweep along $y=-x$, "
        rf"$\theta_{{\rm exc}} = {np.rad2deg(EXCITED_THETA):.0f}$°"
    ),
)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(
    "../figures/miscellaneous/probe_orientation_sweep_distances.png",
    bbox_inches="tight",
)
plt.show()

# %% Plot: theta_max vs r_p / a

fig2, ax2 = plt.subplots(figsize=(5.5, 4))
r_ps = [results[tuple(r)]["dist"] for r in PROBE_RS]
theta_maxs = [results[tuple(r)]["theta_max"] for r in PROBE_RS]
ax2.plot(r_ps, theta_maxs, "o-", color="k", lw=1.4, ms=6)
ax2.axhline(45, color="gray", ls=":", lw=1, label="naive 45°")
ax2.set(
    xlabel=r"$|\mathbf{r}_{\rm probe}| / a$",
    ylabel=r"$\theta_{\max}$ [deg]",
    title=r"Optimum orientation vs. $r_p / a$ (along $y=-x$)",
)
ax2.legend()
fig2.tight_layout()
fig2.savefig(
    "../figures/miscellaneous/probe_orientation_theta_max_vs_distance.png",
    bbox_inches="tight",
)
plt.show()
