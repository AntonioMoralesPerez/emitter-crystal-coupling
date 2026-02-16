import numpy as np
from scipy.sparse import csr_matrix, diags, bmat
from scipy.linalg import eigh


def make_light_matter_Hamiltonian(
    model,
    emitter_detunings=(0.0,),
    couplings=(((0.5,), (0,)),),
    check_hermitian=True,
    atol=1e-10,
):
    """
    Construct the total light–matter Hamiltonian

        H = [[ H_emitters , V ],
             [ V^†         , H_bath ]]

    where H_bath = sqrt(Θ) with Θ the single-particle tight-binding Hamiltonian
    defined by ``model``.

    Parameters
    ----------
    model
        Tight-binding model providing ``get_num_orbitals()``, ``_site_energies``
        and ``_hoppings``.

    emitter_detunings
        Iterable of on-site energies for the emitters. If a single value is
        provided and multiple emitters are present, it is broadcast.

    couplings
        Sequence of tuples ``(g_vals, bath_indices)``, one per emitter, where
        ``g_vals[k]`` couples the emitter to bath orbital ``bath_indices[k]``.

    check_hermitian
        If True, explicitly checks Hermiticity of the final Hamiltonian.

    atol
        Absolute tolerance for numerical zeroing and Hermiticity checks.

    Returns
    -------
    H_total : scipy.sparse.csr_matrix
        Full light–matter Hamiltonian.

    V : numpy.ndarray
        Eigenvector matrix diagonalizing Θ (Θ = V diag(eigvals) V†).
    """

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    N_orbitals = int(model.get_num_orbitals())
    N_emitters = len(couplings)
    H_dimension = N_orbitals + N_emitters

    # Broadcast emitter detunings if needed
    if np.isscalar(emitter_detunings):
        emitter_detunings = [emitter_detunings] * N_emitters
    elif len(emitter_detunings) == 1 and N_emitters > 1:
        emitter_detunings = list(emitter_detunings) * N_emitters

    if len(emitter_detunings) != N_emitters:
        raise ValueError("Length of emitter_detunings must match number of emitters")

    # ------------------------------------------------------------------
    # 1. Construct Θ in real space (dense)
    # ------------------------------------------------------------------
    Theta = np.zeros((N_orbitals, N_orbitals), dtype=complex)

    for i, val in enumerate(model._site_energies):
        Theta[i, i] += val

    for val, i, j in model._hoppings:
        Theta[i, j] += val
        Theta[j, i] += np.conjugate(val)

    # ------------------------------------------------------------------
    # 2. Diagonalize Θ and form H_bath = sqrt(Θ)
    # ------------------------------------------------------------------
    eigvals, V = eigh(Theta)
    omegas = np.sqrt(np.clip(eigvals, 0.0, None))

    H_bath = V @ np.diag(omegas) @ V.conj().T
    H_bath[np.abs(H_bath) < atol] = 0.0
    H_bath_csc = csr_matrix(H_bath)

    # ------------------------------------------------------------------
    # 3. Emitters
    # ------------------------------------------------------------------
    H_emitters = diags(
        np.asarray(emitter_detunings, dtype=complex),
        offsets=0,
        shape=(N_emitters, N_emitters),
        format="csr",
    )

    # ------------------------------------------------------------------
    # 4. Coupling block V (emitters × bath)
    # ------------------------------------------------------------------
    coupling_vals = []
    i_coupling = []
    j_coupling = []

    for emitter_idx, (g_vals, bath_indices) in enumerate(couplings):
        g_vals = np.asarray(g_vals, dtype=complex)
        bath_indices = np.asarray(bath_indices, dtype=int)

        if g_vals.shape != bath_indices.shape:
            raise ValueError(
                f"Emitter {emitter_idx}: g_vals and bath_indices must have same shape"
            )

        if (
            bath_indices.min(initial=0) < 0
            or bath_indices.max(initial=-1) >= N_orbitals
        ):
            raise ValueError(
                f"Emitter {emitter_idx}: bath index out of range [0, {N_orbitals-1}]"
            )

        i_coupling.append(np.full_like(bath_indices, emitter_idx))
        j_coupling.append(bath_indices)
        coupling_vals.append(g_vals)

    if coupling_vals:
        i_coupling = np.hstack(i_coupling)
        j_coupling = np.hstack(j_coupling)
        coupling_vals = np.hstack(coupling_vals)
    else:
        i_coupling = np.array([], dtype=int)
        j_coupling = np.array([], dtype=int)
        coupling_vals = np.array([], dtype=complex)

    H_coupling = csr_matrix(
        (coupling_vals, (i_coupling, j_coupling)),
        shape=(N_emitters, N_orbitals),
        dtype=complex,
    )

    # ------------------------------------------------------------------
    # 5. Full Hamiltonian
    # ------------------------------------------------------------------
    H_total = bmat(
        [[H_emitters, H_coupling], [H_coupling.getH(), H_bath_csc]],
        format="csr",
    )

    # ------------------------------------------------------------------
    # 6. Consistency checks
    # ------------------------------------------------------------------
    if H_total.shape != (H_dimension, H_dimension):
        raise ValueError(
            f"Incorrect Hamiltonian shape {H_total.shape}, expected {(H_dimension, H_dimension)}"
        )

    if check_hermitian:
        diff = H_total - H_total.getH()  # type: ignore
        max_dev = np.max(np.abs(diff.data)) if diff.nnz else 0.0
        if max_dev > atol:
            raise ValueError(f"H_total is not Hermitian (max deviation {max_dev})")

    return H_total, omegas
