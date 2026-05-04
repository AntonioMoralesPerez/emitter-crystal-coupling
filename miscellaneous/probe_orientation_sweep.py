"""
Probe orientation sweep: maximum probe population vs. orientation.

Fixes the excited emitter at R0=[0,0] (θ=90°) and sweeps the probe
orientation from 0 to 180°.

Parameters and conventions match kimble.py exactly: same dipole strength,
same PML settings, same time units (1/Γ₀), same TB model.

NOTE: run mpb_field_computation.py at EXCITED_R and PROBE_R before executing.
"""

import numpy as np
import matplotlib.pyplot as plt
import pythtb as tb
from scipy.sparse.linalg import expm_multiply

import emitter_crystal_coupling as mf

# %% Parameters (matching kimble.py)

LAT_PARAM: float = 300e-9  # lattice parameter [m]
DIP_STRENGTH: float = 7000  # dipole moment [Debye] — same for both emitters
P_A: float = mf.unit_dip(a=LAT_PARAM)
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A

BAND_IDX: int = 0
N_K: int = 101

EXCITED_R: list = [0, 0]
EXCITED_THETA: float = 90 * np.pi / 180
EXCITED_DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array(
    [np.cos(EXCITED_THETA), np.sin(EXCITED_THETA), 0]
)

PROBE_R: list = [-5, 5]
PROBE_DIP_STRENGTH: float = DIP_STRENGTH  # [Debye]
N_THETAS: int = 21
PROBE_THETAS: np.ndarray = np.linspace(40, 50, N_THETAS) * np.pi / 180

# Current kimble.py probe orientation — shown as reference on the plot
KIMBLE_PROBE_THETA: float = np.deg2rad(45)

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

T_F: float = 1e-2  # end time [1/Γ₀]
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
_t_conv = LAT_PARAM / mf.constants.C_LIGHT * gamma_0
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

# %% Load probe fields (orientation-independent, reused for every angle)

data_probe = mf.load(f"./data/kimble/fields/{N_K}_meshgrid_at_{PROBE_R}.pkl")
omega_ks_probe = np.asarray(data_probe["freqs"])
fr0_probe = np.asarray(data_probe["Eks"])[0]
ks_probe = np.asarray(data_probe["kpts"])

# %% Orientation sweep

max_probe_pops = np.zeros(N_THETAS)

print(f"\n--- probe frequency: omega_0 = {omega_0:.6f} c/a ---")
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
            [list(excited_coupling), list(indexes)],
            [list(probe_coupling), list(indexes)],
        ],
        pml_dampings=pml_dampings,
    )

    psi0 = np.zeros(H.shape[0])
    psi0[0] = 1
    psit = expm_multiply(-1j * H, psi0, 0.0, t_f_internal, T_STEPS, endpoint=True)
    max_probe_pops[i] = np.max(np.abs(psit[:, 1]) ** 2)

    print(
        f"θ_probe={np.rad2deg(probe_theta):.1f}° → "
        f"max probe pop = {max_probe_pops[i]:.4e}"
    )

# %% Plot

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(np.rad2deg(PROBE_THETAS), max_probe_pops, "o-", color="k", lw=1.5, ms=4)

max_idx = np.argmax(max_probe_pops)
ax.plot(
    np.rad2deg(PROBE_THETAS[max_idx]),
    max_probe_pops[max_idx],
    "*",
    color="r",
    ms=10,
    label=(
        rf"max $= {max_probe_pops[max_idx]:e}$"
        rf" at $\theta = {np.rad2deg(PROBE_THETAS[max_idx]):.1f}$°"
    ),
)
ax.axvline(
    np.rad2deg(KIMBLE_PROBE_THETA),
    color="gray",
    ls="--",
    lw=1,
    label=rf"kimble.py: $\theta = {np.rad2deg(KIMBLE_PROBE_THETA):.0f}$°",
)
ax.set(
    xlabel=r"$\theta_{\rm probe}$ [deg]",
    ylabel=r"$\max_t \, |C_{\rm probe}(t)|^2$",
    title=(
        rf"probe $\omega = \omega_0$ ({omega_0:.4f} $c/a$), "
        rf"$\theta_{{\rm exc}} = {np.rad2deg(EXCITED_THETA):.0f}$°"
    ),
)
ax.legend()

fig.tight_layout()
fig.savefig(
    f"./figures/miscellaneous/probe_orientation_sweep_at_{PROBE_R}.png",
    bbox_inches="tight",
)
plt.show()
