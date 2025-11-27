import numpy as np
from typing import Union, Optional


def monkhorst_pack_sampling(
    N: Union[int, list, tuple],
    b1: np.ndarray,
    b2: np.ndarray,
    b3: Optional[np.ndarray] = None,
    coordinates: str = "lattice",
) -> np.ndarray:
    """
    Generate a Monkhorst-Pack k-point sampling of the Brillouin zone.

    Parameters
    ----------
    N : int or list/tuple of int
        Number of points per direction. If scalar, the same number is used
        for all dimensions.
    b1, b2 : array_like
        First two reciprocal lattice vectors (3-element arrays).
    b3 : array_like, optional
        Third reciprocal lattice vector for 3D sampling. Default is None (2D).
    coordinates : {'lattice', 'cartesian'}, optional
        Return coordinates in lattice units ('lattice') or Cartesian coordinates ('cartesian').
        Default is 'lattice'.

    Returns
    -------
    kpoints : ndarray, shape (N1*N2*(N3), 3)
        Array of k-points in the first Brillouin zone.

    Notes
    -----
    Monkhorst-Pack k-points are defined as:

        k_{n1,...,ndim} = sum_i bi * (2*ni - Ni - 1)/(2*Ni)

    References
    ----------
    H. J. Monkhorst and J. D. Pack, "Special points for Brillouin-zone integrations",
    Phys. Rev. B 13, 5188–5192 (1976).

    Examples
    --------
    >>> b1 = np.array([np.sqrt(3)/2, 0.5, 0])
    >>> b2 = np.array([np.sqrt(3)/2, -0.5, 0])
    >>> monkhorst_pack_sampling(2, b1, b2)
    array([[-0.25, -0.25, 0.  ],
           [-0.25,  0.25, 0.  ],
           [ 0.25, -0.25, 0.  ],
           [ 0.25,  0.25, 0.  ]])
    """
    # Convert scalar N to a list
    if np.isscalar(N):
        N = [N] * (3 if b3 is not None else 2)

    # Internal np.ndarray for vectorized computation
    N_arr = np.array(N, dtype=int)

    # Reciprocal vectors
    b1 = np.array(b1)
    b2 = np.array(b2)
    if b3 is not None:
        b3 = np.array(b3)

    # Monkhorst-Pack normalized grid function
    def mp_grid(Ni):
        return (2 * np.arange(1, Ni + 1) - Ni - 1) / (2 * Ni)

    # 2D case
    if b3 is None:
        n1, n2 = np.meshgrid(mp_grid(N_arr[0]), mp_grid(N_arr[1]), indexing="ij")
        kpoints_lattice = np.stack([n1, n2, np.zeros_like(n1)], axis=-1).reshape(-1, 3)
        if coordinates == "cartesian":
            kpoints = kpoints_lattice @ np.vstack([b1, b2, [0, 0, 0]])
        else:
            kpoints = kpoints_lattice
    # 3D case
    else:
        n1, n2, n3 = np.meshgrid(
            mp_grid(N_arr[0]), mp_grid(N_arr[1]), mp_grid(N_arr[2]), indexing="ij"
        )
        kpoints_lattice = np.stack([n1, n2, n3], axis=-1).reshape(-1, 3)
        if coordinates == "cartesian":
            kpoints = kpoints_lattice @ np.vstack([b1, b2, b3])
        else:
            kpoints = kpoints_lattice

    return kpoints
