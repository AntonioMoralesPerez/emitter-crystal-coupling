"""
Time evolution via dense diagonalisation.

For non-Hermitian H (PML absorbing layer), the Krylov-based
``scipy.sparse.linalg.expm_multiply`` adapts substep count to the
spectral radius of (t_f - t_0) * H. Its cost therefore scales linearly
with t_f, which becomes prohibitive when t_f is set in 1/Γ₀ units and
the dipole strength is reduced to realistic values (Γ₀ shrinks → larger
internal t_f).

Diagonalising H once and evaluating ψ(t) = V (exp(-iλt) ⊙ c), with
c = V⁻¹ ψ₀, is O(n³) regardless of t_f and then O(n · T) for T time
points. The break-even point is around T ~ n / m where m is the Krylov
subspace size used by expm_multiply.
"""

from collections.abc import Sequence

import numpy as np
import scipy.linalg
from scipy import sparse


def evolve_diagonalisation(
    H: np.ndarray | sparse.spmatrix,
    psi0: np.ndarray,
    times: Sequence[float] | np.ndarray,
    return_cond: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """
    Time-evolve ψ₀ under H by full diagonalisation.

    Parameters
    ----------
    H : (n, n) array-like
        Hamiltonian in units of c/a. May be dense or sparse; sparse inputs
        are densified for the eigendecomposition. Need not be Hermitian
        (e.g. when a PML term is included).
    psi0 : (n,) array-like
        Initial state vector.
    times : (T,) array-like
        Times at which to evaluate ψ(t), in units consistent with H
        (i.e. a/c if H is in c/a).
    return_cond : bool, optional
        If True, also return ``cond(V)``, useful as a sanity check on
        the numerical noise floor (the error in ``ψ(t)`` from the solve
        scales with ``cond(V) · ε_mach``). Default is False.

    Returns
    -------
    psi_t : (T, n) np.ndarray
        ψ(t_i) for each t_i in ``times``.
    cond_V : float, optional
        Condition number of the right-eigenvector matrix. Only returned
        when ``return_cond=True``.

    Notes
    -----
    ``V`` from ``scipy.linalg.eig`` is **not** unitary, so the initial
    coefficients are obtained with ``np.linalg.solve(V, ψ₀)`` rather than
    ``V.conj().T @ ψ₀``.
    """
    H_dense = H.toarray() if sparse.issparse(H) else np.asarray(H)  # type: ignore
    times = np.asarray(times)

    lam, V = scipy.linalg.eig(H_dense)  # type: ignore
    c = np.linalg.solve(V, psi0)
    phase = np.exp(-1j * np.outer(lam, times))
    psi_t = (V @ (phase * c[:, None])).T

    if return_cond:
        return psi_t, float(np.linalg.cond(V))
    return psi_t
