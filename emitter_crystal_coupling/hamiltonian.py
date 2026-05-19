from collections.abc import Sequence
from typing import cast

import numpy as np
import pythtb as tb
from scipy.linalg import eigh


def make_light_matter_Hamiltonian(
    tb_model: tb.tb_model,
    omega_0s: float | Sequence[float] = (0.0,),
    gr0s_rs: Sequence[tuple] = (((0.5,), (0,)),),
    pml_dampings: np.ndarray | None = None,
    check_hermitian: bool = True,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
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
        Absolute tolerance for the hermiticity check.
        Default is 1e-10.

    Returns
    -------
    H : np.ndarray, shape (dim_h, dim_h)
        Full light-matter Hamiltonian in units of c/a (dense, complex).
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
      so ``H_b = (U * omegas) @ U.conj().T``.
    - ``H_int`` is the coupling block connecting each emitter to the bath orbitals.

    The result is dense because ``H_b`` is dense (eigenvectors of a 2-D TB
    Hamiltonian mix every site). Time-evolution paths that previously used
    ``scipy.sparse.linalg.expm_multiply`` still accept this dense form.
    """
    # --------------------------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------------------------
    n_orbitals = int(tb_model.get_num_orbitals())
    n_emitters = len(gr0s_rs)
    dim_h = n_orbitals + n_emitters

    # Broadcast emitter detunings if needed
    if np.isscalar(omega_0s):
        omega_0s_list: list[float] = [float(omega_0s)] * n_emitters  # type: ignore[arg-type]
    elif len(cast(Sequence[float], omega_0s)) == 1 and n_emitters > 1:
        omega_0s_list = list(cast(Sequence[float], omega_0s)) * n_emitters
    else:
        omega_0s_list = list(cast(Sequence[float], omega_0s))

    if len(omega_0s_list) != n_emitters:
        raise ValueError(
            f"`omega_0s` length ({len(omega_0s_list)}) must match the number of "
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
    # 2. Diagonalize Theta and form mode frequencies
    # --------------------------------------------------------------------------
    tb_evals, u_mat = eigh(theta)
    omegas = np.sqrt(np.clip(tb_evals, 0.0, None))

    # --------------------------------------------------------------------------
    # 3. Assemble full H directly into a dense array
    # --------------------------------------------------------------------------
    H = np.zeros((dim_h, dim_h), dtype=complex)

    # Emitter block
    H[np.arange(n_emitters), np.arange(n_emitters)] = np.asarray(
        omega_0s_list, dtype=complex
    )

    # Bath block: H_b = (U * omegas) @ U† (one matmul, no diag() allocation).
    H[n_emitters:, n_emitters:] = (u_mat * omegas) @ u_mat.conj().T

    # Absorbing boundary layer (PML)
    if pml_dampings is not None:
        pml_dampings = np.asarray(pml_dampings, dtype=float)
        if pml_dampings.shape != (n_orbitals,):
            raise ValueError(
                f"`pml_dampings` must have shape ({n_orbitals},), "
                f"got {pml_dampings.shape}."
            )
        bath_diag = np.arange(n_emitters, dim_h)
        H[bath_diag, bath_diag] -= 1j * pml_dampings
        check_hermitian = False

    # Coupling block H_int (and its conjugate transpose)
    for emitter_idx, (g_vals, bath_idx) in enumerate(gr0s_rs):
        g_vals = np.asarray(g_vals, dtype=complex)
        bath_idx = np.asarray(bath_idx, dtype=int)

        if g_vals.shape != bath_idx.shape:
            raise ValueError(
                f"Emitter {emitter_idx}: `g_vals` and `bath_idx` must have the same shape, "
                f"got {g_vals.shape} and {bath_idx.shape}."
            )
        if bath_idx.size and (
            bath_idx.min() < 0 or bath_idx.max() >= n_orbitals
        ):
            raise ValueError(
                f"Emitter {emitter_idx}: `bath_idx` out of range [0, {n_orbitals - 1}]."
            )

        col = n_emitters + bath_idx
        H[emitter_idx, col] = g_vals
        H[col, emitter_idx] = g_vals.conj()

    # --------------------------------------------------------------------------
    # 4. Consistency checks
    # --------------------------------------------------------------------------
    if check_hermitian:
        max_dev = float(np.max(np.abs(H - H.conj().T)))
        if max_dev > atol:
            raise ValueError(
                f"`H` is not Hermitian (max deviation {max_dev:.2e}, tolerance {atol:.2e})."
            )

    return H, omegas
