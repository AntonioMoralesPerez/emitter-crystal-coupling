"""
Probe orientation sweep using a direct TB Hamiltonian (no square root).

Same as probe_orientation_sweep.py but replaces H_b = sqrt(Theta) with
H_b = Theta, where Theta is the tight-binding matrix from PythTB. The
emitter frequency and coupling prefactor are adjusted self-consistently:

  - omega_0  = tb_eval at the X-point  (not sqrt of it)
  - coupling prefactor uses sqrt(omega_k^2 / 2Nk) = omega_k / sqrt(2Nk)

Time conversion still uses the physical frequency so axes are comparable
to probe_orientation_sweep.py.

NOTE: run mpb_field_computation.py at EXCITED_R and PROBE_R before executing.
"""

import matplotlib.pyplot as plt
import numpy as np
import pythtb as tb
from scipy.sparse import bmat, csr_matrix, diags

import emitter_crystal_coupling as mf
from emitter_crystal_coupling.constants import C_LIGHT

# %% Parameters (matching probe_orientation_sweep.py exactly)

LAT_PARAM: float = 300e-9  # [m]
DIP_STRENGTH: float = 7000  # [D]
P_A: float = float(mf.unit_dip(a=LAT_PARAM))
DIP_STRENGTH_NORM: float = DIP_STRENGTH / P_A

BAND_IDX: int = 0
N_K: int = 81

EXCITED_R: list = [0, 0]
EXCITED_THETA: float = 90 * np.pi / 180
EXCITED_DIPOLE: np.ndarray = DIP_STRENGTH_NORM * np.array(
    [np.cos(EXCITED_THETA), np.sin(EXCITED_THETA), 0]
)

PROBE_R: list = [-5, 5]
PROBE_DIP_STRENGTH: float = DIP_STRENGTH
N_THETAS: int = 21
PROBE_THETAS: np.ndarray = np.linspace(40, 60, N_THETAS) * np.pi / 180

KIMBLE_PROBE_THETA: float = np.deg2rad(45)

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 41
N_PML: int = 8
GAMMA_MAX: float = 3e-2  # [c/a]

# in units of [c/a]
J_0: float = 0.07755639
J_1: float = -0.01993698
J_2: float = 0.00053966

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
N_K_PATH: int = 100

T_F: float = 1e-2  # [Γ₀⁻¹]
T_STEPS: int = 100


# %% Helper: PML dampings


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    half = n_cells // 2
    dist = np.minimum(half - np.abs(rs[:, 0]), half - np.abs(rs[:, 1]))
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


# %% Helper: Hamiltonian with H_b = Theta (no sqrt)


def make_direct_tb_Hamiltonian(
    tb_model: tb.tb_model,
    omega_0s,
    gr0s_rs,
    pml_dampings: np.ndarray | None = None,
    atol: float = 1e-10,
):
    """Build H with H_b = Theta (raw TB matrix, no square-root transformation)."""
    n_orbitals = int(tb_model.get_num_orbitals())
    n_emitters = len(gr0s_rs)

    if np.isscalar(omega_0s):
        omega_0s = [omega_0s] * n_emitters

    # --- Bath block: H_b = Theta ---
    theta = np.zeros((n_orbitals, n_orbitals), dtype=complex)
    for i, val in enumerate(tb_model._site_energies):
        theta[i, i] += val
    for val, i, j in tb_model._hoppings:
        theta[i, j] += val
        theta[j, i] += np.conjugate(val)

    h_bath_csr = csr_matrix(theta)

    if pml_dampings is not None:
        pml_dampings = np.asarray(pml_dampings, dtype=float)
        pml_diag = diags(
            pml_dampings, offsets=0, shape=(n_orbitals, n_orbitals), format="csr"
        )
        h_bath_csr = h_bath_csr - 1j * pml_diag

    # --- Emitter block ---
    h_emitter = diags(
        np.asarray(omega_0s, dtype=complex),
        offsets=0,
        shape=(n_emitters, n_emitters),
        format="csr",
    )

    # --- Coupling block ---
    coupling_vals, i_coupling, j_coupling = [], [], []
    for emitter_idx, (g_vals, bath_idx) in enumerate(gr0s_rs):
        g_vals = np.asarray(g_vals, dtype=complex)
        bath_idx = np.asarray(bath_idx, dtype=int)
        i_coupling.append(np.full_like(bath_idx, emitter_idx))
        j_coupling.append(bath_idx)
        coupling_vals.append(g_vals)

    i_coupling = np.hstack(i_coupling)
    j_coupling = np.hstack(j_coupling)
    coupling_vals = np.hstack(coupling_vals)

    h_int = csr_matrix(
        (coupling_vals, (i_coupling, j_coupling)),
        shape=(n_emitters, n_orbitals),
        dtype=complex,
    )

    return bmat([[h_emitter, h_int], [h_int.getH(), h_bath_csr]], format="csr")


# %% TB model and omega_0

infinite_model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
infinite_model.set_onsite([J_0])
for vec in [[1, 0], [0, 1]]:
    infinite_model.set_hop(J_1, 0, 0, vec)
for vec in [[1, 1], [1, -1]]:
    infinite_model.set_hop(J_2, 0, 0, vec)

k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals = infinite_model.solve_all(k_vec)

# Physical frequency (for time conversion) and raw TB eigenvalue (for H)
omega_0_phys = mf.safe_sqrt(tb_evals[BAND_IDX][np.isclose(k_dist, k_node[1])]).item()
omega_0_sq = omega_0_phys**2  # eigenvalue used as emitter frequency in direct-TB model

print(f"omega_0 (phys) = {omega_0_phys:.6f} c/a")
print(f"omega_0 (sq)   = {omega_0_sq:.6f} (c/a)^2")

# %% Band structure plot

K_LABELS = [r"$\Gamma$", "X", "M", r"$\Gamma$"]

fig_bs, ax_bs = plt.subplots()
ax_bs.plot(k_dist, tb_evals[BAND_IDX], color="r", linestyle="--")
ax_bs.hlines(
    omega_0_sq,
    xmin=k_dist[0],
    xmax=k_dist[-1],
    color="k",
    linestyle="--",
    alpha=0.5,
    label=rf"$\omega_0^2 = {omega_0_sq:.4f}$ $(c/a)^2$",
)
ax_bs.set(
    xlim=(k_node[0], k_node[-1]),
    xticks=k_node,
    xticklabels=K_LABELS,
    ylabel=r"$\omega^2 \, [(c/a)^2]$",
    title="TB band structure (direct TB)",
)
ax_bs.grid(axis="x")
ax_bs.legend()
fig_bs.tight_layout()
fig_bs.savefig(
    "../figures/miscellaneous/tb_bandstructure_probe_orientation_sweep_direct_tb.png",
    bbox_inches="tight",
)
plt.show()

temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)
_, _, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)
indexes = np.arange(len(rs_finite))

# %% Time conversion (uses physical omega_0 so axes match probe_orientation_sweep.py)

gamma_0 = mf.free_spontaneous_emission(
    omega_0=omega_0_phys, dip_strength=DIP_STRENGTH, a=LAT_PARAM
)
_t_conv = LAT_PARAM / C_LIGHT * gamma_0
t_f_internal = T_F / _t_conv

# %% Load fields — use omega_ks^2 in coupling prefactor for self-consistency with H_b=Theta

data_excited = mf.load(f"../data/kimble/fields/{N_K}_meshgrid_at_{EXCITED_R}.pkl")
omega_ks_n_exc = (
    np.asarray(data_excited["freqs"]) ** 2
)  # omega^2 for direct-TB coupling
fr0_n_ks_exc = np.asarray(data_excited["Eks"])[0]
ks_exc = np.asarray(data_excited["kpts"])

excited_coupling = mf.compute_real_space_coupling(
    fr0_n_ks=fr0_n_ks_exc,
    dipole_moment=EXCITED_DIPOLE,
    omega_ks_n=omega_ks_n_exc,
    ks=ks_exc,
    rs=rs_finite,
)[BAND_IDX, :]

data_probe = mf.load(f"../data/kimble/fields/{N_K}_meshgrid_at_{PROBE_R}.pkl")
omega_ks_n_probe = np.asarray(data_probe["freqs"]) ** 2
fr0_probe = np.asarray(data_probe["Eks"])[0]
ks_probe = np.asarray(data_probe["kpts"])

# %% Orientation sweep

max_probe_pops = np.zeros(N_THETAS)

print(f"\n--- direct-TB sweep, omega_0_sq = {omega_0_sq:.6f} (c/a)^2 ---")
for i, probe_theta in enumerate(PROBE_THETAS):
    probe_dipole = (PROBE_DIP_STRENGTH / P_A) * np.array(
        [np.cos(probe_theta), np.sin(probe_theta), 0]
    )
    probe_coupling = mf.compute_real_space_coupling(
        fr0_n_ks=fr0_probe,
        dipole_moment=probe_dipole,
        omega_ks_n=omega_ks_n_probe,
        ks=ks_probe,
        rs=rs_finite,
    )[BAND_IDX, :]

    H = make_direct_tb_Hamiltonian(
        tb_model=finite_model,
        omega_0s=[omega_0_sq, omega_0_sq],
        gr0s_rs=[
            [list(excited_coupling), list(indexes)],
            [list(probe_coupling), list(indexes)],
        ],
        pml_dampings=pml_dampings,
    )

    assert H.shape is not None
    psi0 = np.zeros(H.shape[0])
    psi0[0] = 1
    psit = mf.evolve_diagonalisation(H, psi0, np.linspace(0.0, t_f_internal, T_STEPS))
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
        rf"direct TB ($H_b = \Theta$), probe $\omega_0^2 = {omega_0_sq:.4f}$ $(c/a)^2$, "
        rf"$\theta_{{\rm exc}} = {np.rad2deg(EXCITED_THETA):.0f}$°"
    ),
)
ax.legend()

fig.tight_layout()
fig.savefig(
    f"../figures/miscellaneous/probe_orientation_sweep_direct_tb_at_{PROBE_R}.png",
    bbox_inches="tight",
)
plt.show()
