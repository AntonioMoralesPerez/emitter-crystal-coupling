import numpy as np
from scipy.sparse import csr_matrix, diags, bmat
from scipy.linalg import eigh
from collections.abc import Sequence

import pythtb as tb


def make_light_matter_Hamiltonian(
    tb_model: tb.tb_model,
    omega_0s: float | Sequence[float] = (0.0,),
    gr0s_rs: Sequence[tuple] = (((0.5,), (0,)),),
    pml_dampings: np.ndarray | None = None,
    check_hermitian: bool = True,
    atol: float = 1e-10,
) -> tuple[csr_matrix, np.ndarray]:
    """
    Construct the total light-matter Hamiltonian for the system, in units of c/a.

    Parameters
    ----------
    tb_model : tb.tb_model
        Tight-binding model of the photonic crystal in units of (c/a)².
        Assumed to be a finite PythTB model with no periodic boundary conditions.
    omega_0s : float or Sequence of float
        Detuning frequencies of the emitters in units of c/a. If a single value
        is provided, it is broadcast to all emitters detected in `gr0s_rs`.
    gr0s_rs : Sequence of tuple
        Coupling data for each emitter in units of c/a, provided as
        ``(g_vals, bath_idx)`` pairs where ``g_vals[i]`` is the coupling
        strength to the orbital at ``bath_idx[i]``.
    pml_dampings : np.ndarray of shape (n_orbitals,) or None, optional
        Per-orbital damping rates γ_j ≥ 0 for the absorbing boundary layer
        [units: c/a]. When provided, adds ``-i * diag(γ)`` to the bath block,
        making H non-Hermitian. ``check_hermitian`` is automatically disabled.
        Default is None (no absorbing layer).
    check_hermitian : bool, optional
        If True, explicitly checks hermiticity of the final Hamiltonian.
        Automatically set to False when ``pml_dampings`` is provided.
        Default is True.
    atol : float, optional
        Absolute tolerance for numerical precision and hermiticity checks.
        Default is 1e-10.

    Returns
    -------
    H : scipy.sparse.csr_matrix
        Full light-matter Hamiltonian in units of c/a.
    omegas : np.ndarray
        Bath mode frequencies, defined as
        ``omegas = sqrt(clip(tb_evals, 0, None))``,
        where ``tb_evals`` are the eigenvalues of the TB Hamiltonian.

    Raises
    ------
    ValueError
        If ``omega_0s`` length does not match the number of emitters.
        If ``g_vals`` and ``bath_idx`` shapes do not match for any emitter.
        If ``bath_idx`` contains out-of-range indices for any emitter.
        If the final Hamiltonian has an unexpected shape.
        If the final Hamiltonian is not Hermitian (when ``check_hermitian=True``).

    Notes
    -----
    The Hamiltonian is structured as:

        H = [ H_e,     H_int  ]
            [ H_int†,  H_b    ]

    where:

    - ``H_e`` is a diagonal matrix of emitter detuning frequencies ``omega_0s``.
    - ``H_b = sqrt(Theta)``, with ``Theta`` the tight-binding Hamiltonian. The
      square root is taken in the eigenbasis: ``Theta = U diag(tb_evals) U†``,
      so ``H_b = U diag(sqrt(tb_evals)) U†``.
    - ``H_int`` is the coupling block connecting each emitter to the bath orbitals.
    """
    # --------------------------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------------------------
    n_orbitals = int(tb_model.get_num_orbitals())
    n_emitters = len(gr0s_rs)
    dim_h = n_orbitals + n_emitters

    # Broadcast emitter detunings if needed
    if np.isscalar(omega_0s):
        omega_0s = [omega_0s] * n_emitters
    elif len(omega_0s) == 1 and n_emitters > 1:
        omega_0s = list(omega_0s) * n_emitters

    if len(omega_0s) != n_emitters:
        raise ValueError(
            f"`omega_0s` length ({len(omega_0s)}) must match the number of "
            f"emitters ({n_emitters})."
        )

    # --------------------------------------------------------------------------
    # 1. Construct TB Hamiltonian Theta in real space (dense)
    # --------------------------------------------------------------------------
    # NOTE: assumes a finite model with no periodic boundary conditions.
    theta = np.zeros((n_orbitals, n_orbitals), dtype=complex)

    # NOTE: _site_energies and _hoppings are private PythTB attributes with no
    # public equivalent. Pin the PythTB version if this breaks across upgrades.
    for i, val in enumerate(tb_model._site_energies):
        theta[i, i] += val

    for val, i, j in tb_model._hoppings:
        theta[i, j] += val
        theta[j, i] += np.conjugate(val)

    # --------------------------------------------------------------------------
    # 2. Diagonalize Theta and form H_b = sqrt(Theta)
    # --------------------------------------------------------------------------
    tb_evals, u_mat = eigh(theta)
    omegas = np.sqrt(np.clip(tb_evals, 0.0, None))

    h_bath = u_mat @ np.diag(omegas) @ u_mat.conj().T
    h_bath[np.abs(h_bath) < atol] = 0.0
    h_bath_csr = csr_matrix(h_bath)

    # --------------------------------------------------------------------------
    # 2b. Absorbing boundary layer (PML)
    # --------------------------------------------------------------------------
    if pml_dampings is not None:
        pml_dampings = np.asarray(pml_dampings, dtype=float)
        if pml_dampings.shape != (n_orbitals,):
            raise ValueError(
                f"`pml_dampings` must have shape ({n_orbitals},), "
                f"got {pml_dampings.shape}."
            )
        pml_diag = diags(
            pml_dampings, offsets=0, shape=(n_orbitals, n_orbitals), format="csr"
        )
        h_bath_csr = h_bath_csr - 1j * pml_diag
        check_hermitian = False

    # --------------------------------------------------------------------------
    # 3. Emitter block H_e
    # --------------------------------------------------------------------------
    h_emitter = diags(
        np.asarray(omega_0s, dtype=complex),
        offsets=0,
        shape=(n_emitters, n_emitters),
        format="csr",
    )

    # --------------------------------------------------------------------------
    # 4. Coupling block H_int, shape (n_emitters, n_orbitals)
    # --------------------------------------------------------------------------
    coupling_vals = []
    i_coupling = []
    j_coupling = []

    for emitter_idx, (g_vals, bath_idx) in enumerate(gr0s_rs):
        g_vals = np.asarray(g_vals, dtype=complex)
        bath_idx = np.asarray(bath_idx, dtype=int)

        if g_vals.shape != bath_idx.shape:
            raise ValueError(
                f"Emitter {emitter_idx}: `g_vals` and `bath_idx` must have the same shape, "
                f"got {g_vals.shape} and {bath_idx.shape}."
            )
        if bath_idx.min(initial=0) < 0 or bath_idx.max(initial=-1) >= n_orbitals:
            raise ValueError(
                f"Emitter {emitter_idx}: `bath_idx` out of range [0, {n_orbitals - 1}]."
            )

        i_coupling.append(np.full_like(bath_idx, emitter_idx))
        j_coupling.append(bath_idx)
        coupling_vals.append(g_vals)

    if coupling_vals:
        i_coupling = np.hstack(i_coupling)
        j_coupling = np.hstack(j_coupling)
        coupling_vals = np.hstack(coupling_vals)
    else:
        i_coupling = np.array([], dtype=int)
        j_coupling = np.array([], dtype=int)
        coupling_vals = np.array([], dtype=complex)

    h_int = csr_matrix(
        (coupling_vals, (i_coupling, j_coupling)),
        shape=(n_emitters, n_orbitals),
        dtype=complex,
    )

    # --------------------------------------------------------------------------
    # 5. Full Hamiltonian
    # --------------------------------------------------------------------------
    H = bmat(
        [[h_emitter, h_int], [h_int.getH(), h_bath_csr]],
        format="csr",
    )

    # --------------------------------------------------------------------------
    # 6. Consistency checks
    # --------------------------------------------------------------------------
    if H.shape != (dim_h, dim_h):
        raise ValueError(
            f"Incorrect Hamiltonian shape {H.shape}, expected ({dim_h}, {dim_h})."
        )

    if check_hermitian:
        diff = H - H.getH()
        max_dev = np.max(np.abs(diff.data)) if diff.nnz else 0.0
        if max_dev > atol:
            raise ValueError(
                f"`H` is not Hermitian (max deviation {max_dev:.2e}, tolerance {atol:.2e})."
            )

    return H, omegas
