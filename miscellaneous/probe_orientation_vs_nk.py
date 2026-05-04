"""
Optimal probe orientation vs. Monkhorst-Pack grid size (N_K).

For each N_K in NK_VALUES, runs the probe orientation sweep and records
the angle that maximises the probe population.

NOTE: run mpb_field_computation.py for each desired N_K before executing.
"""

import matplotlib.pyplot as plt
import numpy as np
import pythtb as tb
from scipy.sparse.linalg import expm_multiply

import emitter_crystal_coupling as mf
from emitter_crystal_coupling.constants import C_LIGHT

# %% Parameters (matching kimble.py / probe_orientation_sweep.py)

LAT_PARAM: float = 300e-9
DIP_STRENGTH: float = 50
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A

BAND_IDX: int = 0

EXCITED_R: list = [0, 0]
EXCITED_THETA: float = 90 * np.pi / 180
EXCITED_DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array(
    [np.cos(EXCITED_THETA), np.sin(EXCITED_THETA), 0]
)

PROBE_R: list = [-5, 5]
PROBE_DIP_STRENGTH: float = DIP_STRENGTH

PROBE_THETAS: np.ndarray = np.linspace(40, 60, 13) * np.pi / 180

FIELDS_DIR: str = "./data/kimble/fields/"
NK_VALUES: list = [61, 71, 81, 91]

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

T_F: float = 2e-5
T_STEPS: int = 50


# %% Helper


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    half = n_cells // 2
    dist = np.minimum(half - np.abs(rs[:, 0]), half - np.abs(rs[:, 1]))
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


# %% Tight-binding model variants

TB_VARIANTS = {"1NN": 0.0, "1NN+2NN": J_2}

_, _, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)
indexes = np.arange(len(rs_finite))


def build_tb_variant(j2):
    model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
    model.set_onsite([J_0])
    for vec in [[1, 0], [0, 1]]:
        model.set_hop(J_1, 0, 0, vec)
    if j2 != 0.0:
        for vec in [[1, 1], [1, -1]]:
            model.set_hop(j2, 0, 0, vec)

    k_vec, k_dist, k_node = model.k_path(K_PATH, N_K_PATH, report=False)
    tb_evals = model.solve_all(k_vec)
    omega_0 = mf.safe_sqrt(tb_evals[BAND_IDX][np.isclose(k_dist, k_node[1])]).item()

    finite = model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False).cut_piece(
        N_CELLS, fin_dir=1, glue_edgs=False
    )
    gamma_0 = mf.free_spontaneous_emission(
        omega_0=omega_0, dip_strength=DIP_STRENGTH, a=LAT_PARAM
    )
    t_f_internal = T_F / (LAT_PARAM / C_LIGHT * gamma_0)

    return finite, omega_0, t_f_internal


# %% Main sweep over TB variants and N_K

# results[variant][nk] = (optimal_theta_deg, max_pop)
results = {label: {} for label in TB_VARIANTS}

for label, j2 in TB_VARIANTS.items():
    print(f"\n{'=' * 50}")
    print(f"TB variant: {label}")
    finite_model, omega_0, t_f_internal = build_tb_variant(j2)
    print(f"  omega_0 = {omega_0:.6f} c/a")

    for nk in NK_VALUES:
        print(f"\n  N_K = {nk} ({'even' if nk % 2 == 0 else 'odd'})")

        data_excited = mf.load(f"{FIELDS_DIR}{nk}_meshgrid_at_{EXCITED_R}.pkl")
        excited_coupling = mf.compute_real_space_coupling(
            fr0_n_ks=np.asarray(data_excited["Eks"])[0],
            dipole_moment=EXCITED_DIPOLE,
            omega_ks_n=np.asarray(data_excited["freqs"]),
            ks=np.asarray(data_excited["kpts"]),
            rs=rs_finite,
        )[BAND_IDX, :]

        data_probe = mf.load(f"{FIELDS_DIR}{nk}_meshgrid_at_{PROBE_R}.pkl")
        omega_ks_probe = np.asarray(data_probe["freqs"])
        fr0_probe = np.asarray(data_probe["Eks"])[0]
        ks_probe = np.asarray(data_probe["kpts"])

        pops = np.zeros(len(PROBE_THETAS))
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
            psit = expm_multiply(
                -1j * H, psi0, 0.0, t_f_internal, T_STEPS, endpoint=True
            )
            pops[i] = np.max(np.abs(psit[:, 1]) ** 2)

            print(f"    θ={np.rad2deg(probe_theta):.1f}° → max pop = {pops[i]:.4e}")

        best_idx = np.argmax(pops)
        results[label][nk] = (np.rad2deg(PROBE_THETAS[best_idx]), pops[best_idx])
        print(
            f"  → optimal θ = {results[label][nk][0]:.1f}°, "
            f"max pop = {results[label][nk][1]:.4e}"
        )

# %% Plot

nks = np.array(NK_VALUES)
odd_mask = nks % 2 == 1
even_mask = ~odd_mask

row_labels = [
    r"$\theta_{\rm opt}$ [deg]",
    r"$\max_t\,|C_{\rm probe}|^2$ at $\theta_{\rm opt}$",
]
col_labels = list(TB_VARIANTS.keys())

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey="row")

for col, tb_label in enumerate(col_labels):
    thetas = np.array([results[tb_label][nk][0] for nk in nks])
    pops_arr = np.array([results[tb_label][nk][1] for nk in nks])

    for row, vals in enumerate([thetas, pops_arr]):
        ax = axes[row, col]
        if odd_mask.any():
            ax.plot(
                nks[odd_mask], vals[odd_mask], "o", color="C0", label="odd $N_K$", ms=7
            )
        if even_mask.any():
            ax.plot(
                nks[even_mask],
                vals[even_mask],
                "s",
                color="C1",
                label="even $N_K$",
                ms=7,
            )
        if col == 0:
            ax.set_ylabel(row_labels[row])
        if row == 0:
            ax.set_title(tb_label)
        if row == 1:
            ax.set_xlabel("$N_K$")
        ax.legend()

fig.suptitle(
    rf"Optimal probe orientation vs. $N_K$ — $\theta_{{\rm exc}}={np.rad2deg(EXCITED_THETA):.0f}$°",
    y=1.01,
)

fig.tight_layout()
fig.savefig(
    f"./figures/miscellaneous/probe_orientation_vs_nk_at_{PROBE_R}.png",
    bbox_inches="tight",
)
plt.show()
