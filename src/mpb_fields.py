import numpy as np


def compute_fr0rs(
    fr0_n_ks: np.ndarray,
    k_pts: np.ndarray,
    r_pts: np.ndarray,
) -> np.ndarray:
    """
    Discrete inverse Bloch transform of MPB fields.

    Computes lattice-resolved field amplitudes (Wannier sampling):

        E_{n,R}(r0) = (1/Nk) * sum_k exp(-i k·R) E_{n,k}(r0)

    Parameters
    ----------
    fr0_n_ks : np.ndarray, shape (Nb, Nk, 3), complex
        Electric field profile evaluated at a fixed intracell position r0
        for each band and k-point.
    k_pts : np.ndarray, shape (Nk, dim), float
        k-points in Cartesian reciprocal coordinates (units: 1/length).

        .. note::
            If your k-points are fractional (MPB default), convert them
            using the reciprocal lattice matrix before passing them here.

    r_pts : np.ndarray, shape (NR, dim), float
        Bravais lattice vectors in Cartesian real-space coordinates
        (units: length), representing the unit cells to sample.

    Returns
    -------
    fr0_n_rs : np.ndarray, shape (Nb, NR, 3), complex
        Field amplitude of the Wannier-like function at position r0
        inside each lattice cell R.

    Raises
    ------
    ValueError
        If the last dimension of `fr0_n_ks` is not 3.
        If `k_pts` does not have Nk rows.
        If `k_pts` and `r_pts` do not share the same spatial dimension.

    Notes
    -----
    This does not reconstruct the full spatial field — it reconstructs
    the lattice dependence at a fixed intracell coordinate r0.
    """
    n_bands, n_kpoints, n_components = fr0_n_ks.shape

    if n_components != 3:
        raise ValueError(
            f"Last dimension of `fr0_n_ks` must be 3 (vector components), "
            f"got {n_components}."
        )
    if k_pts.shape[0] != n_kpoints:
        raise ValueError(
            f"`k_pts` must have {n_kpoints} rows to match Nk, " f"got {k_pts.shape[0]}."
        )
    if k_pts.shape[1] != r_pts.shape[1]:
        raise ValueError(
            f"`k_pts` and `r_pts` must share the same spatial dimension, "
            f"got {k_pts.shape[1]} and {r_pts.shape[1]}."
        )

    # Compute phase factors, shape: (Nk, NR)
    phase = np.exp(-1j * k_pts @ r_pts.T)

    # Inverse Bloch sum over k-points, shape: (Nb, NR, 3)
    fr0_n_rs = np.einsum("nkc,kr->nrc", fr0_n_ks, phase) / n_kpoints

    return fr0_n_rs
