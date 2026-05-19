"""
Toy-model coupling sweep: maximum concurrence vs. G/|J_1|.

Fixes the excited emitter at R_0=[0,0] and the probe at NEW_R=[-5,5]. Sweeps
the local coupling strength G (common to both emitters, in units of |J_1|)
and records max_t C(t).

Parameters and conventions match kimble_toy.py.
"""

from typing import Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pythtb as tb
from scipy.sparse import bmat, csr_matrix, diags

import emitter_crystal_coupling as mf

# %% Parameters (matching kimble_toy.py)

LAT: list = [[1, 0], [0, 1]]
ORB: list = [[0, 0]]
N_CELLS: int = 41
N_PML: int = 8
GAMMA_MAX: float = 2.0

J_1: float = -1.0
J_0: float = 0.0

K_PATH: list = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
N_K_PATH: int = 100

R_0: list = [0, 0]
NEW_R: list = [-5, 5]

G_VALUES: list = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.013, 0.016, 0.02]

T_0: float = 0.0
T_F: float = 5e5
T_STEPS: int = 200


# %% Helpers


def compute_pml_dampings(
    rs: np.ndarray, n_cells: int, n_pml: int, gamma_max: float
) -> np.ndarray:
    half = n_cells // 2
    dist = np.minimum(half - np.abs(rs[:, 0]), half - np.abs(rs[:, 1]))
    return np.where(dist < n_pml, gamma_max * ((n_pml - dist) / n_pml) ** 2, 0.0)


def find_site_index(rs: np.ndarray, r: list) -> int:
    dists = np.sum((rs - np.array(r, dtype=float)) ** 2, axis=1)
    return int(np.argmin(dists))


def build_toy_hamiltonian(
    finite_model: tb.tb_model,
    omega_0s,
    couplings: list,
    pml_dampings: np.ndarray | None = None,
) -> csr_matrix:
    n_orbitals = int(finite_model.get_num_orbitals())
    n_emitters = len(couplings)

    if np.isscalar(omega_0s):
        omega_0s = [omega_0s] * n_emitters

    theta = np.zeros((n_orbitals, n_orbitals), dtype=complex)
    for i, val in enumerate(finite_model._site_energies):
        theta[i, i] += val
    for val, i, j in finite_model._hoppings:
        theta[i, j] += val
        theta[j, i] += np.conjugate(val)

    h_bath = csr_matrix(theta)
    if pml_dampings is not None:
        pml_dampings = np.asarray(pml_dampings, dtype=float)
        h_bath = h_bath - 1j * diags(pml_dampings, format="csr")

    h_emitter = diags(
        np.asarray(omega_0s, dtype=complex),
        offsets=0,
        shape=(n_emitters, n_emitters),
        format="csr",
    )

    g_vals = [g for g, _ in couplings]
    i_rows = list(range(n_emitters))
    j_cols = [idx for _, idx in couplings]
    h_int = csr_matrix(
        (g_vals, (i_rows, j_cols)),
        shape=(n_emitters, n_orbitals),
        dtype=complex,
    )

    return cast(
        csr_matrix, bmat([[h_emitter, h_int], [h_int.getH(), h_bath]], format="csr")
    )


# %% Build tight-binding model and detuning

infinite_model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)
infinite_model.set_onsite([J_0])
for vec in np.array([[1, 0], [0, 1]]):
    infinite_model.set_hop(J_1, 0, 0, vec)

k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, N_K_PATH, report=False)
tb_evals = infinite_model.solve_all(k_vec)
omega_0 = float(tb_evals[0][np.isclose(k_dist, k_node[1])])
print(f"omega_0 = {omega_0:.6f} |J_1|")

temporal_model = infinite_model.cut_piece(N_CELLS, fin_dir=0, glue_edgs=False)
finite_model = temporal_model.cut_piece(N_CELLS, fin_dir=1, glue_edgs=False)

_, _, rs_finite = mf.build_finite_grid(N_CELLS)
pml_dampings = compute_pml_dampings(rs_finite, N_CELLS, N_PML, GAMMA_MAX)

idx_r0 = find_site_index(rs_finite, R_0)
idx_probe = find_site_index(rs_finite, NEW_R)
print(f"Excited site index: {idx_r0}  →  r = {rs_finite[idx_r0]}")
print(f"Probe site index:   {idx_probe}  →  r = {rs_finite[idx_probe]}")

ts = np.linspace(T_0, T_F, T_STEPS, endpoint=True)

# %% G sweep
# results[i] = (Ce_sq, Cp_sq, C_t) arrays of shape (T_STEPS,)

results: list[Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [None] * len(
    G_VALUES
)

for i, g in enumerate(G_VALUES):
    H = build_toy_hamiltonian(
        finite_model=finite_model,
        omega_0s=omega_0,
        couplings=[(g, idx_r0), (g, idx_probe)],
        pml_dampings=pml_dampings,
    )

    assert H.shape is not None
    psi0 = np.zeros(H.shape[0])
    psi0[0] = 1
    psit = mf.evolve_diagonalisation(H, psi0, ts)

    ce_sq = np.abs(psit[:, 0]) ** 2
    cp_sq = np.abs(psit[:, 1]) ** 2
    c_t = 2 * np.abs(ce_sq * np.conj(cp_sq))

    results[i] = (ce_sq, cp_sq, c_t)
    print(f"G/|J_1| = {g:.3g}  →  max C(t) = {c_t.max():.4e}")

# %% Time-evolution plot (1×3): Ce^2, Cp^2, C(t)

cmap = plt.get_cmap("plasma")
sweep_colors = [cmap(i / (len(G_VALUES) - 1)) for i in range(len(G_VALUES))]

fig, (ax_e, ax_p, ax_c) = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

for i, (g, color) in enumerate(zip(G_VALUES, sweep_colors)):
    res_i = results[i]
    assert res_i is not None
    ce_sq, cp_sq, c_t = res_i
    label = rf"${g:g}$"
    ax_e.plot(ts, ce_sq, color=color, label=label)
    ax_p.plot(ts, cp_sq, color=color, label=label)
    ax_c.plot(ts, c_t, color=color, label=label)

for ax, ylabel, yscale in zip(
    [ax_e, ax_p, ax_c],
    [r"$|C_e(t)|^2$", r"$|C_{\rm probe}(t)|^2$", r"$C(t)$"],
    ["log", "log", "linear"],
):
    ax.set(xlabel=r"$t\,[|J_1|^{-1}]$", ylabel=ylabel, yscale=yscale)

ax_c.legend(title=r"$G/|J_1|$", fontsize=7)

fig.suptitle(
    rf"Toy-model coupling sweep, $\mathbf{{r}}_0={R_0}$, $\mathbf{{r}}_p={NEW_R}$"
)
fig.tight_layout()
fig.savefig(f"./figures/miscellaneous/toy_coupling_sweep_at_{R_0}_and_{NEW_R}.png")
plt.show()

# %% Summary plot: max C(t) vs. G/|J_1|

max_c = [r[2].max() for r in results if r is not None]

best_idx = int(np.argmax(max_c))
best_g = G_VALUES[best_idx]
best_c = max_c[best_idx]

fig, ax = plt.subplots()

ax.plot(G_VALUES, max_c, "o-", color="k", lw=1.5, ms=5)
ax.plot(
    best_g,
    best_c,
    "*",
    color="red",
    ms=12,
    zorder=5,
    label=rf"max at $G/|J_1|={best_g:g}$, $C={best_c:.3g}$",
)
ax.legend()
ax.set(
    xlabel=r"$G/|J_1|$",
    ylabel=r"$\max_t \, C(t)$",
    xscale="log",
    title=rf"$\mathbf{{r}}_0={R_0}$, $\mathbf{{r}}_p={NEW_R}$",
)

fig.tight_layout()
fig.savefig(
    f"./figures/miscellaneous/toy_coupling_sweep_summary_at_{R_0}_and_{NEW_R}.png"
)
plt.show()
