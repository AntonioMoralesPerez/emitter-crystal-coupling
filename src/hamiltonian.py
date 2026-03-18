import numpy as np
from scipy.sparse import csr_matrix, diags, bmat
from scipy.linalg import eigh

from typing import Union, Sequence
import warnings


def make_light_matter_Hamiltonian(
    tb_model,
    ω_0s: Union[float, Sequence[float]] = (0.0,),
    gr0s_rs: Sequence[tuple] = (((0.5,), (0,)),),
    check_hermitian: bool = True,
    atol: float = 1e-10,
):
    """
    Construct the total light-matter Hamiltonian for the system, in units of c/a.

    Parameters
    ----------
    tb_model : tb_model
        Tight-binding model of the photonic crystal in units of (c/a)²
        It is assume to be constructed using PythTB.
    ω_0s : float or Sequence of float
        Detuning frequencies of the emitters, in units of c/a. If a single value
        is provided, it is broadcast to all emitters detected in `gr0s_rs`.
    gr0s_rs : Sequence of tuple
        Coupling data for each emitter in units of c/a.
        It is provided as `(g_vals, bath_idx)` pairs where `g_vals[i]` is the
        coupling strength to the orbital at `bath_idx[i]`.
    check_hermitian : bool
        If True, explicitly checks hermiticity of the light-matter Hamiltonian.
        Default `True`.
    atol : float
        Absolute tolerance used for numerical precision and hermiticity checks.
        Default `1e-10`.

    Returns
    -------
    H : scipy.sparse.csr_matrix
        Full light-matter Hamiltonian in units of c/a.
    ωs : numpy.ndarray
        Bath mode frequencies, defined as `ωs = sqrt(clip(tb_evals, 0, None))`,
        where `tb_evals` are the eigenvalues of `Θ`.


    Notes
    -----
    The Hamiltonian is structured as:

    `H = [  [ H_e,      H_int   ],
            [ H_int†,   H_b     ]]`
    where:

    - `H_e` is a diagonal matrix of the emitter detuning frequencies `ω_0s`,
    in units of c/a.
    - `H_b = sqrt(Θ)`, with `Θ` the tight-binding Hamiltonian defined by `tb_model`.
    The square root is taken in the eigenbasis: `Θ = U diag(tb_evals) U†`, so
    `H_b = U diag(sqrt(tb_evals)) U†`.
    - `H_int` is the coupling block, constructed from `gr0s_rs`, that connects each
    emitter to the orbitals of the photonic crystal bath.
    """

    # --------------------------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------------------------
    N_o = int(tb_model.get_num_orbitals())
    N_e = len(gr0s_rs)
    dim_H = N_o + N_e

    # Broadcast emitter detunings if needed
    if np.isscalar(ω_0s):
        ω_0s = [ω_0s] * N_e
    elif len(ω_0s) == 1 and N_e > 1:
        ω_0s = list(ω_0s) * N_e

    if len(ω_0s) != N_e:
        raise ValueError("Length of emitter_detunings must match number of emitters")

    # --------------------------------------------------------------------------
    # 1. Construct Θ in real space (dense)
    # --------------------------------------------------------------------------
    Θ = np.zeros((N_o, N_o), dtype=complex)

    for i, val in enumerate(tb_model._site_energies):
        Θ[i, i] += val

    for val, i, j in tb_model._hoppings:
        Θ[i, j] += val
        Θ[j, i] += np.conjugate(val)
    # NOTE: here it is implicit that the TB model is finite, with no PBC in any
    #       direction

    # --------------------------------------------------------------------------
    # 2. Diagonalize Θ and form H_b = sqrt(Θ)
    # --------------------------------------------------------------------------
    tb_evals, U = eigh(Θ)
    ωs = np.sqrt(np.clip(tb_evals, 0.0, None))

    H_b = U @ np.diag(ωs) @ U.conj().T
    H_b[np.abs(H_b) < atol] = 0.0
    H_b_csr = csr_matrix(H_b)  # we make it a `csr_matrix` for faster operations

    # --------------------------------------------------------------------------
    # 3. Emitters
    # --------------------------------------------------------------------------
    H_e = diags(
        np.asarray(ω_0s, dtype=complex),
        offsets=0,
        shape=(N_e, N_e),
        format="csr",
    )

    # --------------------------------------------------------------------------
    # 4. Coupling block H_int (emitters × bath)
    # --------------------------------------------------------------------------
    coupling_vals = []
    i_coupling = []
    j_coupling = []

    for emitter_idx, (g_vals, bath_idx) in enumerate(gr0s_rs):
        g_vals = np.asarray(g_vals, dtype=complex)
        bath_idx = np.asarray(bath_idx, dtype=int)

        if g_vals.shape != bath_idx.shape:
            raise ValueError(
                f"Emitter {emitter_idx}: `g_vals` and `bath_idx` must have same shape"
            )

        if bath_idx.min(initial=0) < 0 or bath_idx.max(initial=-1) >= N_o:
            raise ValueError(
                f"Emitter {emitter_idx}: `bath_idx` out of range [0, {N_o-1}]"
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

    H_int = csr_matrix(
        (coupling_vals, (i_coupling, j_coupling)),
        shape=(N_e, N_o),
        dtype=complex,
    )

    # ------------------------------------------------------------------
    # 5. Full Hamiltonian
    # ------------------------------------------------------------------
    H = bmat(
        [[H_e, H_int], [H_int.getH(), H_b_csr]],
        format="csr",
    )

    # ------------------------------------------------------------------
    # 6. Consistency checks
    # ------------------------------------------------------------------
    if H.shape != (dim_H, dim_H):
        raise ValueError(
            f"Incorrect Hamiltonian shape {H.shape}, expected {(dim_H, dim_H)}"
        )

    if check_hermitian:
        diff = H - H.getH()
        max_dev = np.max(np.abs(diff.data)) if diff.nnz else 0.0
        if max_dev > atol:
            raise ValueError(f"`H` is not Hermitian (max deviation {max_dev})")

    return H, ωs
