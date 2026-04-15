"""
Probe orientation sweep: maximum probe emitter population vs. orientation.

Fixes the excited emitter at R0=[0,0] with theta=90 deg (the orientation that
produces directional emission toward the probe site) and sweeps the probe
emitter orientation from 0 to 180 deg. For each angle, the two-emitter system
is time-evolved and the maximum probe population max_t |C_probe(t)|^2 is
recorded. The result is plotted as a function of the probe orientation to
verify that theta=45 deg is optimal.

NOTE: this script assumes that the MPB fields have already been computed at
both R0=[0,0] and new_R=[-4.5, 4.5]. If not, run `mpb_field_computation.py`.
"""

# %% Imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pythtb as tb
from scipy.sparse.linalg import expm_multiply

import src as mf

# %% Matplotlib configuration
mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.size": 4,
        "ytick.major.width": 1,
        "legend.fontsize": 10,
        "legend.frameon": False,
    }
)

# %% Parameters

NK = 64
N_CELLS = 31
BAND_INDEX = 0

LAT = [[1, 0], [0, 1]]
ORB = [[0, 0]]

# Tight-binding hopping coefficients [units: (c/a)^2]
TAU_0 = 0.07755639
TAU_1 = -0.01993698
TAU_2 = 0.00053966

DETUNING_FREQ = 0.274587  # emitter detuning at the X point [c/a]

# Excited emitter (fixed)
EXCITED_R = [0.0, 0.0]
EXCITED_THETA = 90 * np.pi / 180
P_EXCITED = 1e-2  # dipole strength [dimensionless]

# Probe emitter (orientation swept)
PROBE_R = [-4.5, 4.5]
P_PROBE = 1e-1  # dipole strength [dimensionless]
N_THETAS = 37  # number of probe orientations (0 to 180 deg)
PROBE_THETAS = np.linspace(0, 180, N_THETAS) * np.pi / 180

# Time evolution
T0 = 0
TF = 50
T_STEPS = 100

# Normalization factors
alpha_norm = TAU_0 / abs(TAU_1)
j1_norm = TAU_1 / abs(TAU_1)
j2_norm = TAU_2 / abs(TAU_1)
detuning_freq_norm = DETUNING_FREQ / np.sqrt(abs(TAU_1))

# %% Build tight-binding model

infinite_model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
infinite_model.set_onsite([alpha_norm])

for vec in [[1, 0], [0, 1]]:
    infinite_model.set_hop(j1_norm, 0, 0, vec)

for vec in [[1, 1], [1, -1]]:
    infinite_model.set_hop(j2_norm, 0, 0, vec)

temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)

xjg, yjg, rjpts = mf.build_finite_grid(N_CELLS)
indexes = np.arange(len(rjpts))

# %% Load field data and compute excited emitter coupling

data_excited = mf.load(f"./data/kimble/fields/{NK}_meshgrid_at_{EXCITED_R}.pkl")
omegas_excited = np.asarray(data_excited["freqs"])  # (Nk, Nb)
E_k_r0 = np.asarray(data_excited["Eks"])[0]  # (Nb, Nk, 3)
kpts_excited = np.asarray(data_excited["kpts"])  # (Nk, 3)

excited_dipole = P_EXCITED * np.array([np.cos(EXCITED_THETA), np.sin(EXCITED_THETA), 0])
gr0rjs_excited = mf.compute_real_space_coupling(
    E_k_r0, excited_dipole, omegas_excited, kpts_excited, rjpts
)
excited_coupling = gr0rjs_excited[BAND_INDEX, :] / np.sqrt(abs(TAU_1))

# %% Determine probe detuning from the single-emitter Hamiltonian

_, omegas_sites = mf.make_light_matter_Hamiltonian(
    finite_model,
    detuning_freq_norm,
    [[list(excited_coupling), list(indexes)]],
)

# Find the site closest to the probe position to use as its detuning frequency
target = [-5, 5]
probe_site_idx = np.where(np.all(np.isclose(rjpts, target), axis=1))[0][0]
probe_detuning = omegas_sites[probe_site_idx]

# %% Load probe field data (same for all orientations)

data_probe = mf.load(f"./data/kimble/fields/{NK}_meshgrid_at_{PROBE_R}.pkl")
omegas_probe = np.asarray(data_probe["freqs"])  # (Nk, Nb)
E_k_R = np.asarray(data_probe["Eks"])[0]  # (Nb, Nk, 3)
kpts_probe = np.asarray(data_probe["kpts"])  # (Nk, 3)

# %% Orientation sweep

max_probe_pops = np.zeros(N_THETAS)

for i, theta in enumerate(PROBE_THETAS):
    probe_dipole = P_PROBE * np.array([np.cos(theta), np.sin(theta), 0])

    gRrjs = mf.compute_real_space_coupling(
        E_k_R, probe_dipole, omegas_probe, kpts_probe, rjpts
    )
    probe_coupling = gRrjs[BAND_INDEX, :] / np.sqrt(abs(TAU_1))

    H, _ = mf.make_light_matter_Hamiltonian(
        finite_model,
        [detuning_freq_norm, probe_detuning],
        [
            [list(excited_coupling), list(indexes)],
            [list(probe_coupling), list(indexes)],
        ],
    )

    psi0 = np.zeros(H.shape[0])
    psi0[0] = 1  # excited emitter starts in |e>

    psit = expm_multiply(-1j * H, psi0, T0, TF, T_STEPS, endpoint=True)
    max_probe_pops[i] = np.max(np.abs(psit[:, 1]) ** 2)  # probe is emitter index 1

    print(
        f"theta = {np.rad2deg(theta):.1f} deg -> max probe pop = {max_probe_pops[i]:.4e}"
    )

# %% Plot

fig, ax = plt.subplots()

ax.plot(np.rad2deg(PROBE_THETAS), max_probe_pops, "o-", color="k", lw=1.5, ms=4)
ax.axvline(45, color="r", ls="--", lw=1, label=r"$\theta_{\rm probe} = 45°$")

max_idx = np.argmax(max_probe_pops)
ax.plot(
    np.rad2deg(PROBE_THETAS[max_idx]),
    max_probe_pops[max_idx],
    "*",
    color="r",
    ms=10,
    label=rf"max $= {max_probe_pops[max_idx]:.3f}$ at $\theta = {np.rad2deg(PROBE_THETAS[max_idx]):.1f}$°",
)

ax.set(
    xlabel=r"$\theta_{\rm probe}$ [deg]",
    ylabel=r"$\max_t \, |C_{\rm probe}(t)|^2$",
    xticks=np.arange(0, 181, 45),
)
ax.legend()

fig.tight_layout()
fig.savefig(f"./figures/miscellaneous/probe_orientation_sweep_at_{PROBE_R}.png")
plt.show()
