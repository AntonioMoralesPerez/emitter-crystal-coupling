import numpy as np


def _mp_grid(n: int) -> np.ndarray:
    """
    Return the 1D Monkhorst-Pack grid for n points in [-0.5, 0.5).

    Parameters
    ----------
    n : int
        Number of grid points.

    Returns
    -------
    np.ndarray, shape (n,)
        Grid coordinates in fractional reciprocal lattice units.
    """
    return (2 * np.arange(1, n + 1) - n - 1) / (2 * n)


def monkhorst_pack_sampling(
    n_points: int | list | tuple,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray | None = None,
    coordinates: str = "lattice",
) -> np.ndarray:
    """
    Generate a Monkhorst-Pack k-point sampling of the Brillouin zone.

    Parameters
    ----------
    n_points : int or list or tuple of int
        Number of k-points per direction. If a scalar is provided, the same
        number is used for all dimensions.
    b1, b2 : np.ndarray, shape (3,)
        First two reciprocal lattice vectors.
    b3 : np.ndarray, shape (3,), optional
        Third reciprocal lattice vector for 3D sampling. Default is None (2D).
    coordinates : {'lattice', 'cartesian'}, optional
        Return k-points in fractional lattice units ('lattice') or Cartesian
        coordinates ('cartesian'). Default is 'lattice'.

    Returns
    -------
    kpoints : np.ndarray, shape (N1*N2[*N3], 3)
        Array of k-points in the first Brillouin zone.

    Raises
    ------
    ValueError
        If `coordinates` is not 'lattice' or 'cartesian'.

    Notes
    -----
    Monkhorst-Pack k-points are defined as:

        k_{n1,...,ndim} = sum_i b_i * (2*n_i - N_i - 1) / (2*N_i)

    References
    ----------
    .. [1] H. J. Monkhorst and J. D. Pack, "Special points for Brillouin-zone
           integrations", Phys. Rev. B 13, 5188–5192 (1976).

    Examples
    --------
    >>> b1 = np.array([1, 0, 0])
    >>> b2 = np.array([0, 1, 0])
    >>> monkhorst_pack_sampling(2, b1, b2)
    array([[-0.25, -0.25,  0.  ],
           [-0.25,  0.25,  0.  ],
           [ 0.25, -0.25,  0.  ],
           [ 0.25,  0.25,  0.  ]])
    """
    if coordinates not in ("lattice", "cartesian"):
        raise ValueError(
            f"`coordinates` must be 'lattice' or 'cartesian', got '{coordinates}'."
        )

    # Broadcast scalar n_points to all dimensions
    n_dims = 3 if b3 is not None else 2
    if np.isscalar(n_points):
        n_points = [n_points] * n_dims
    n_arr = np.array(n_points, dtype=int)

    b1 = np.array(b1)
    b2 = np.array(b2)

    if b3 is None:
        # 2D sampling
        n1, n2 = np.meshgrid(_mp_grid(n_arr[0]), _mp_grid(n_arr[1]), indexing="ij")
        kpoints_lattice = np.stack([n1, n2, np.zeros_like(n1)], axis=-1).reshape(-1, 3)

        if coordinates == "cartesian":
            # Pad basis matrix with a zero row for the unused 3rd dimension
            kpoints = kpoints_lattice @ np.vstack([b1, b2, [0, 0, 0]])
        else:
            kpoints = kpoints_lattice
    else:
        # 3D sampling
        b3 = np.array(b3)
        n1, n2, n3 = np.meshgrid(
            _mp_grid(n_arr[0]),
            _mp_grid(n_arr[1]),
            _mp_grid(n_arr[2]),
            indexing="ij",
        )
        kpoints_lattice = np.stack([n1, n2, n3], axis=-1).reshape(-1, 3)

        if coordinates == "cartesian":
            kpoints = kpoints_lattice @ np.vstack([b1, b2, b3])
        else:
            kpoints = kpoints_lattice

    return kpoints
