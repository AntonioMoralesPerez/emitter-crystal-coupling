import numpy as np


def compute_ERr0(
    E_k_r0: np.ndarray,
    kpts: np.ndarray,
    Rpts: np.ndarray,
) -> np.ndarray:
    """
    Discrete inverse Bloch transform of MPB fields.

    Computes lattice-resolved field amplitudes (Wannier sampling):

        E_{n,R}(r0) = (1/Nk) * sum_k exp(-i k·R) E_{n,k}(r0)

    Parameters
    ----------
    E_k_r0 : complex ndarray (Nb, Nk, 3)
        Electric field evaluated at a fixed intracell position r0
        for each band and k-point.

    kpts : float ndarray (Nk, dim)
        k-points in *Cartesian reciprocal coordinates* (units: 1/length).
        IMPORTANT: If your k-points are fractional (MPB default),
        you MUST convert them using the reciprocal lattice matrix.

    Rpts : float ndarray (NR, dim)
        Bravais lattice vectors in Cartesian real-space coordinates
        (units: length). These represent which unit cells you want
        to sample the Wannier function in.

    Returns
    -------
    E_Rr0 : complex ndarray (Nb, NR, 3)
        Field amplitude of the Wannier-like function at position r0
        inside each lattice cell R.

    Notes
    -----
    - This does NOT reconstruct the full spatial field; it reconstructs
      the lattice dependence at a fixed intracell coordinate r0.
    """

    Nb, Nk, comp = E_k_r0.shape
    if comp != 3:
        raise ValueError("Last dimension of E_k_r0 must be 3 (vector components)")

    if kpts.shape[0] != Nk:
        raise ValueError("kpts must have Nk rows")

    if kpts.shape[1] != Rpts.shape[1]:
        raise ValueError("kpts and Rpts must have same spatial dimension")

    # phase factor: exp(-i k·R)
    phase = np.exp(-1j * (kpts @ Rpts.T))  # Nk x NR

    # inverse Bloch sum
    E_Rr0 = np.einsum("nkc,kr->nrc", E_k_r0, phase) / Nk

    return E_Rr0
