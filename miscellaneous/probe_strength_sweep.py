"""
Probe strength sweep: emitter and probe populations vs. probe dipole strength.

Fixes the excited emitter at R0=[0,0] (θ=90°) and the probe orientation at
PROBE_THETA=60°. Sweeps the probe dipole moment over PROBE_DIP_VALUES.

Parameters and conventions match kimble.py exactly.

NOTE: run mpb_field_computation.py at EXCITED_R and PROBE_R before executing.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pythtb as tb

import emitter_crystal_coupling as mf
from emitter_crystal_coupling.constants import C_LIGHT

# %% Parameters (matching kimble.py)

LAT_PARAM: float = 300e-9  # lattice parameter [m]
DIP_STRENGTH: float = 50  # excited emitter dipole [Debye]
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A

BAND_IDX: int = 0
N_K: int = 64

EXCITED_R: list = [0.0, 0.0]
EXCITED_THETA: float = 90 * np.pi / 180
EXCITED_DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array(
    [np.cos(EXCITED_THETA), np.sin(EXCITED_THETA), 0]
)

PROBE_R: list = [-5, 5]
PROBE_THETA: float = np.deg2rad(60)
PROBE_DIP_VALUES: list = [10, 25, 50, 75, 100, 150, 200]  # [Debye]

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 41
N_PML: int = 8
GAMMA_MAX: float = 3e-2

J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
N_K_PATH: int = 100

T_F: float = 2e-5  # end time [1/Γ₀]
T_STEPS: int = 50


# %% Helper


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    half = n_cells // 2
    dist = np.minimum(half - np.abs(rs[:, 0]), half - np.abs(rs[:, 1]))
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


# %% Tight-binding model and omega_0

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

# %% Time conversion (1/Γ₀ → a/c for internal use)

gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / C_LIGHT * gamma_0
t_f_internal = T_F / _t_conv
ts_norm = np.linspace(0.0, T_F, T_STEPS, endpoint=True)

# %% Load excited emitter fields and compute couplings

data_excited = mf.load(f"./data/kimble/fields/{N_K}_meshgrid_at_{EXCITED_R}.pkl")
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

probe_detuning = omega_0

# %% Load probe fields and compute direction-only base coupling
# Coupling scales linearly with dipole moment, so compute once with the unit
# direction vector and scale in the loop: coupling = (dip_val / P_A) * base.

data_probe = mf.load(f"./data/kimble/fields/{N_K}_meshgrid_at_{PROBE_R}.pkl")
omega_ks_probe = np.asarray(data_probe["freqs"])
fr0_probe = np.asarray(data_probe["Eks"])[0]
ks_probe = np.asarray(data_probe["kpts"])

probe_direction = np.array([np.cos(PROBE_THETA), np.sin(PROBE_THETA), 0])
base_probe_coupling = mf.compute_real_space_coupling(
    fr0_n_ks=fr0_probe,
    dipole_moment=probe_direction,
    omega_ks_n=omega_ks_probe,
    ks=ks_probe,
    rs=rs_finite,
)[BAND_IDX, :]

# %% Strength sweep
# results[i] = (Ce_sq, Cprobe_sq) arrays of shape (T_STEPS,)
# i = probe dipole strength index

results: list[Optional[tuple[np.ndarray, np.ndarray]]] = [None] * len(PROBE_DIP_VALUES)

for i, dip_val in enumerate(PROBE_DIP_VALUES):
    probe_coupling = (dip_val / P_A) * base_probe_coupling

    H, _ = mf.make_light_matter_Hamiltonian(
        tb_model=finite_model,
        omega_0s=[omega_0, probe_detuning],
        gr0s_rs=[
            (list(excited_coupling), list(indexes)),
            (list(probe_coupling), list(indexes)),
        ],
        pml_dampings=pml_dampings,
    )

    psi0 = np.zeros(H.shape[0])
    psi0[0] = 1
    psit = mf.evolve_diagonalisation(H, psi0, np.linspace(0.0, t_f_internal, T_STEPS))

    results[i] = (
        np.abs(psit[:, 0]) ** 2,  # |C_excited(t)|^2
        np.abs(psit[:, 1]) ** 2,  # |C_probe(t)|^2
    )

    res_i = results[i]
    assert res_i is not None
    print(f"dip={dip_val} D → max probe pop = {res_i[1].max():.4e}")

# %% Time-evolution plot (1×2): cols = Ce^2 / Cprobe^2

cmap = plt.get_cmap("plasma")
sweep_colors = [
    cmap(i / (len(PROBE_DIP_VALUES) - 1)) for i in range(len(PROBE_DIP_VALUES))
]

fig, (ax_e, ax_p) = plt.subplots(1, 2, sharex=True)

for i, (dip_val, color) in enumerate(zip(PROBE_DIP_VALUES, sweep_colors)):
    res_i = results[i]
    assert res_i is not None
    ce_sq, cp_sq = res_i
    label = rf"${dip_val}$ D"
    ax_e.plot(ts_norm, ce_sq, color=color, label=label)
    ax_p.plot(ts_norm, cp_sq, color=color, label=label)

for ax, ylabel in zip(
    [ax_e, ax_p],
    [r"$|C_e(t)|^2$", r"$|C_{\rm probe}(t)|^2$"],
):
    ax.set(
        xlabel=r"$t\,\Gamma_0$",
        ylabel=ylabel,
        yscale="log",
        title=rf"$\theta_{{\rm exc}} = {np.rad2deg(EXCITED_THETA):.0f}$°",
    )

ax_p.legend(fontsize=7)

fig.suptitle(
    rf"Probe strength sweep, $\theta_{{\rm probe}} = {np.rad2deg(PROBE_THETA):.0f}$°"
)
fig.tight_layout()
fig.savefig(
    f"./figures/miscellaneous/probe_strength_sweep_at_{PROBE_R}"
    f"_theta_{np.rad2deg(PROBE_THETA):.0f}deg.png"
)
plt.show()

# %% Summary plot: max probe population vs. probe dipole strength

max_pops = [r[1].max() for r in results if r is not None]

fig, ax = plt.subplots()

ax.plot(PROBE_DIP_VALUES, max_pops, "o-", color="k", lw=1.5, ms=5)
ax.axvline(
    DIP_STRENGTH, color="gray", ls="--", lw=1, label=rf"excited = {DIP_STRENGTH} D"
)
ax.set(
    xlabel=r"Probe dipole strength [Debye]",
    ylabel=r"$\max_t \, |C_{\rm probe}(t)|^2$",
    title=rf"$\theta_{{\rm exc}} = {np.rad2deg(EXCITED_THETA):.0f}$°, "
    rf"$\theta_{{\rm probe}} = {np.rad2deg(PROBE_THETA):.0f}$°",
)
ax.legend()

fig.tight_layout()
fig.savefig(
    f"./figures/miscellaneous/probe_strength_sweep_summary_at_{PROBE_R}"
    f"_theta_{np.rad2deg(PROBE_THETA):.0f}deg.png"
)
plt.show()
