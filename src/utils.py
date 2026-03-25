import os
import pickle
import numpy as np
import h5py
import meep as mp
import meep.mpb as mpb

from .constants import E_CHARGE, C_LIGHT, H_BAR, EPSILON_0

from matplotlib.axes import Axes


def save(data_dict: dict[str, any], filename: str, dirname: str = "") -> None:
    """
    Save a dictionary of Python objects to a pickle file.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping variable names (str) to Python objects.
        Example: {"freqs": freqs, "Ex": Ex}.
    filename : str
        Name of the pickle file to save (e.g., 'data.pkl').
    dirname : str, optional
        Directory where the file should be saved. If empty, saves in the
        current working directory. If the directory does not exist, it is
        created automatically.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be written due to permission or filesystem errors.
    TypeError
        If `data_dict` cannot be pickled.

    Notes
    -----
    Uses Python's built-in `pickle` module for serialization. Be cautious
    when unpickling data from untrusted sources.

    Examples
    --------
    >>> data = {"var1": 5, "var2": np.array([0, 0, 0])}
    >>> save(data, "data.pkl", "./testfolder/testsubfolder")
    """
    try:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            filepath = os.path.join(dirname, filename)
        else:
            filepath = filename

        with open(filepath, "wb") as f:
            pickle.dump(data_dict, f)

    except (OSError, TypeError) as e:
        raise RuntimeError(f"Failed to save data to '{filename}': {e}") from e


def load(filename: str) -> dict[str, any]:
    """
    Load variables from a pickle file into a dictionary.

    Parameters
    ----------
    filename : str
        Path to the pickle file (e.g., './testfolder/testsubfolder/data.pkl').

    Returns
    -------
    dict
        Dictionary containing the stored variables, with the same keys as
        were passed to `save`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    OSError
        If the file cannot be opened due to permission or filesystem errors.
    pickle.UnpicklingError
        If the file is not a valid pickle file.

    Notes
    -----
    Unlike older versions that injected variables into the global namespace,
    this function returns a dictionary. Assign variables explicitly for clarity.

    Examples
    --------
    >>> data = load("./testfolder/testsubfolder/data.pkl")
    >>> freqs = data["freqs"]
    >>> Ex = data["Ex"]
    """
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: '{filename}'") from e
    except (OSError, pickle.UnpicklingError) as e:
        raise RuntimeError(f"Failed to load data from '{filename}': {e}") from e


def arrange_data(
    fields_in: np.ndarray,
    kpts_in: np.ndarray,
    r0pts_in: np.ndarray,
    freqs_in: np.ndarray,
    lattice_vecs: np.ndarray,
) -> dict[str, any]:
    """
    Arrange photonic crystal field data and points for real-space calculations.

    Converts k-points and r0-points to Cartesian coordinates, reshapes the
    field data by band and component, and applies Bloch phases.

    Parameters
    ----------
    fields_in : np.ndarray, shape (Nk*Nb*Nr, 3)
        Electric field data for all r0-points and k-points, concatenated
        along the first dimension. The last dimension indexes components
        (x, y, z).
    kpts_in : np.ndarray, shape (Nk, 3)
        k-points in lattice coordinates.
    r0pts_in : np.ndarray, shape (Nr, 3)
        Emitter positions in lattice coordinates.
    freqs_in : np.ndarray, shape (Nk, Nb)
        Frequencies of the photonic modes.
    lattice_vecs : np.ndarray, shape (3, 3)
        Lattice vectors of the crystal.

    Returns
    -------
    dict
        Dictionary containing:

        - ``'frequencies'``: mode frequencies, shape (Nk, Nb).
        - ``'uks'``: reshaped field components without Bloch phase,
          shape (Nr, Nb, Nk, 3).
        - ``'Eks'``: field components with Bloch phases applied,
          shape (Nr, Nb, Nk, 3).
        - ``'kpts_cartesian'``: k-points in Cartesian coordinates,
          shape (Nk, 3).
        - ``'r0pts_cartesian'``: r0 points in Cartesian coordinates,
          shape (Nr, 3).

    Raises
    ------
    ValueError
        If `fields_in` does not have shape (Nk*Nb*Nr, 3).

    Notes
    -----
    Bloch phases are applied using the convention ``exp(+i k·r)``.
    The function reshapes ``fields_in`` from (Nk*Nb*Nr, 3) to
    (Nk, Nb, Nr, 3) internally before reordering to (Nr, Nb, Nk, 3).
    """
    n_kpoints = kpts_in.shape[0]
    n_r = r0pts_in.shape[0]
    n_bands = freqs_in.shape[1]

    expected_size = n_kpoints * n_bands * n_r
    if fields_in.shape[0] != expected_size or fields_in.shape[1] != 3:
        raise ValueError(
            f"`fields_in` must have shape ({expected_size}, 3), "
            f"got {fields_in.shape}."
        )

    # Convert k-points and r0-points to Cartesian coordinates
    lattice_vecs_recipr = find_reciprocal_lattice_vectors(lattice_vecs, twopi=True)
    kpts_cartesian = lattice_to_cartesian(
        kpts_in, lattice_vecs_recipr
    )  # shape: (Nk, 3)
    r0pts_cartesian = lattice_to_cartesian(r0pts_in, lattice_vecs)  # shape: (Nr, 3)

    # Reshape fields to (Nk, Nb, Nr, 3)
    uks = fields_in.reshape((n_kpoints, n_bands, n_r, 3))

    # Compute Bloch phase exp(+i k·r), shape: (Nk, Nr)
    bloch_phase = np.exp(1j * np.dot(kpts_cartesian, r0pts_cartesian.T))

    # Apply Bloch phase to each band and component via broadcasting
    eks = uks * bloch_phase[:, None, :, None]  # shape: (Nk, Nb, Nr, 3)

    # Reorder axes to (Nr, Nb, Nk, 3) for downstream convenience
    uks = np.transpose(uks, (2, 1, 0, 3))
    eks = np.transpose(eks, (2, 1, 0, 3))

    return {
        "frequencies": freqs_in,
        "uks": uks,
        "Eks": eks,
        "kpts_cartesian": kpts_cartesian,
        "r0pts_cartesian": r0pts_cartesian,
    }


def read_epsilon_file(epsilon_input_file: str) -> dict[str, any]:
    """
    Read dielectric permittivity (epsilon) data and lattice vectors from an HDF5 file.

    Parameters
    ----------
    epsilon_input_file : str
        Path to the HDF5 file containing epsilon data. The file is expected
        to have datasets:

        - ``'data'``: 3D or lower-dimensional array with permittivity values.
        - ``'lattice vectors'``: 3x3 array of lattice vectors.

    Returns
    -------
    dict
        Dictionary containing:

        - ``'epsilon_data'``: ndarray with permittivity values.
        - ``'lattice_vectors'``: list of lattice vectors.
        - ``'resolution'``: list of integers specifying the number of points
          along each axis.
        - ``'lattice_size'``: list of integers, placeholder for lattice size
          (currently initialized to ones).

    Raises
    ------
    KeyError
        If the HDF5 file does not contain 'data' or 'lattice vectors' datasets.

    Notes
    -----
    Dimensions are padded to 3 if the data is lower-dimensional.
    ``'lattice_size'`` is currently a placeholder initialized to ones — adjust
    this if your lattice conventions require specific values.
    """
    with h5py.File(epsilon_input_file, "r") as f:
        if "data" not in f or "lattice vectors" not in f:
            raise KeyError(
                f"File '{epsilon_input_file}' must contain 'data' and "
                "'lattice vectors' datasets."
            )
        epsilon_data = f["data"][()]  # type: ignore
        lattice_vectors = f["lattice vectors"][()].tolist()  # type: ignore

    # Determine resolution (number of points along each axis), padded to 3D
    resolution = list(epsilon_data.shape)  # type: ignore
    while len(resolution) < 3:
        resolution.append(1)

    # Placeholder lattice size — TODO: adjust for specific lattice conventions
    lattice_size = [1] * len(epsilon_data.shape)  # type: ignore
    while len(lattice_size) < 3:
        lattice_size.append(0)

    return {
        "epsilon_data": epsilon_data,
        "lattice_vectors": lattice_vectors,
        "resolution": resolution,
        "lattice_size": lattice_size,
    }


def convert_epsilon(filename: str, md: mpb.MPBData) -> dict[str, any]:
    """
    Convert dielectric function data from an HDF5 file to a convenient format
    with positions in lattice and Cartesian coordinates.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing the dielectric function. Expected
        datasets:

        - ``'data'``: N-dimensional array with epsilon values.
        - ``'lattice vectors'``: 3x3 array of lattice vectors.

    md : mpb.MPBData
        MPBData instance used to transform the input data (e.g., to refine
        grid or convert to the desired format).

    Returns
    -------
    dict
        Dictionary containing:

        - ``'epsilon_values'``: ndarray, flattened dielectric function values.
        - ``'positions-cartesian'``: ndarray, shape (N_points, 3),
          positions in Cartesian coordinates.
        - ``'positions-lattice'``: ndarray, shape (N_points, 3),
          positions in lattice coordinates (normalized to lattice constants).
        - ``'lattice_vectors'``: list of lists, lattice vectors of the unit cell.

    Raises
    ------
    KeyError
        If the HDF5 file does not contain 'data' or 'lattice vectors' datasets.

    Notes
    -----
    The origin of the unit cell is placed at [0, 0, 0] in lattice coordinates.
    Positions are generated according to the grid size of the converted epsilon
    matrix. Assumes equal number of repeated periods along all directions as
    specified in ``md.multiply_size``.

    Examples
    --------
    >>> filename = 'epsilon.h5'
    >>> md = mpb.MPBData(rectify=False, periods=3)
    >>> data = convert_epsilon(filename, md)
    >>> epsilon_values = data['epsilon_values']
    >>> positions_car = data['positions-cartesian']
    >>> positions_lat = data['positions-lattice']
    """
    with h5py.File(filename, "r") as f:
        if "data" not in f or "lattice vectors" not in f:
            raise KeyError(
                f"File '{filename}' must contain 'data' and "
                "'lattice vectors' datasets."
            )
        epsilon_data = f["data"][()]  # type: ignore
        lattice_vectors = f["lattice vectors"][()].tolist()  # type: ignore

    # Number of periods along each axis (assumes equal for all directions)
    periods = md.multiply_size[0]

    # Convert to MPBArray and transform using MPBData
    epsilon_mpb_array = mpb.MPBArray(epsilon_data, lattice=lattice_vectors)
    epsilon_matrix = md.convert(epsilon_mpb_array)

    # Grid size of the converted data
    grid_size = np.array(epsilon_matrix.shape)

    # Generate pixel positions and shift origin to [0, 0, 0]
    axes = [np.arange(n) for n in grid_size]
    positions_pixels = meshgrid_points(*axes)  # shape: (N_points, ndim)

    center = grid_size / 2
    normalization_factor = grid_size / periods
    positions_lat = (positions_pixels - center) / normalization_factor

    # Convert lattice coordinates to Cartesian
    positions_car = lattice_to_cartesian(positions_lat, lattice_vectors)

    return {
        "epsilon_values": epsilon_matrix.flatten(),
        "positions-cartesian": positions_car,
        "positions-lattice": positions_lat,
        "lattice_vectors": lattice_vectors,
    }


# ------------------------------------------------------------------------------
# Reciprocal lattice and coordinate transforms
# ------------------------------------------------------------------------------


def find_reciprocal_lattice_vectors(
    lattice_vectors: np.ndarray | list[list[float]], twopi: bool = True
) -> np.ndarray:
    """
    Compute the reciprocal lattice vectors from a set of direct lattice vectors.

    The reciprocal lattice vectors b1, b2, b3 satisfy:

        a_i · b_j = 2π δ_ij   (if twopi=True)
        a_i · b_j = δ_ij      (if twopi=False)

    Parameters
    ----------
    lattice_vectors : np.ndarray or list of lists, shape (3, 3)
        Direct lattice vectors a1, a2, a3. Each row corresponds to a
        lattice vector.
    twopi : bool, optional
        If True (default), reciprocal vectors are multiplied by 2π.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Reciprocal lattice vectors b1, b2, b3, one per row.

    Raises
    ------
    ValueError
        If `lattice_vectors` does not have shape (3, 3).

    Notes
    -----
    Uses the standard solid-state definition:
    b_i = 2π * (a_j × a_k) / (a1 · (a2 × a3)), with cyclic permutation.
    See: Joannopoulos et al., "Photonic Crystals: Molding the Flow of Light",
    p. 235.

    Examples
    --------
    >>> lattice_vectors = [[1.0, 0.0, 0.0],
    ...                    [0.0, 1.0, 0.0],
    ...                    [0.0, 0.0, 1.0]]
    >>> b = find_reciprocal_lattice_vectors(lattice_vectors)
    >>> b[0]
    array([6.28318531, 0.        , 0.        ])
    """
    a = np.array(lattice_vectors, dtype=float)
    if a.shape != (3, 3):
        raise ValueError(f"`lattice_vectors` must have shape (3, 3), got {a.shape}.")

    b = np.linalg.inv(a).T
    if twopi:
        b *= 2 * np.pi

    return b


def lattice_to_cartesian(
    orbitals: np.ndarray | list[list[float]],
    lattice_vectors: np.ndarray | list[list[float]],
) -> np.ndarray:
    """
    Convert positions from lattice coordinates to Cartesian coordinates.

    Parameters
    ----------
    orbitals : np.ndarray or list of lists, shape (N_points, ndim)
        Positions in lattice coordinates. Each row is a point in the unit
        cell. ndim can be 1, 2, or 3.
    lattice_vectors : np.ndarray or list of lists, shape (3, 3)
        Lattice vectors of the crystal. Each row corresponds to a lattice
        vector (a1, a2, a3). If the lattice dimension is < 3, pad missing
        vectors as standard: 2D → a3 = [0, 0, 1], 1D → a2 = [0, 1, 0],
        a3 = [0, 0, 1].

    Returns
    -------
    np.ndarray, shape (N_points, 3)
        Positions in Cartesian coordinates.

    Notes
    -----
    Converts each position ``r`` in lattice coordinates to Cartesian using:

        r_cartesian = r1*a1 + r2*a2 + r3*a3

    Examples
    --------
    >>> orbitals = [[0, 0], [0, 1], [1, 0], [1, 1]]
    >>> lattice_vectors = [[np.sqrt(3)/2,  0.5, 0],
    ...                    [np.sqrt(3)/2, -0.5, 0],
    ...                    [0, 0, 1]]
    >>> lattice_to_cartesian(orbitals, lattice_vectors)
    array([[0.        ,  0.        , 0.],
           [0.8660254 , -0.5       , 0.],
           [0.8660254 ,  0.5       , 0.],
           [1.73205081,  0.        , 0.]])
    """
    orbitals = np.array(orbitals, dtype=float)
    lattice_vectors = np.array(lattice_vectors, dtype=float)

    _, ndim = orbitals.shape

    # Pad orbitals to 3D if necessary
    if ndim < 3:
        orbitals = np.pad(orbitals, ((0, 0), (0, 3 - ndim)), mode="constant")

    return orbitals @ lattice_vectors


def array_to_Vector3(
    arr: np.ndarray | list[list[float]] | list[float],
) -> list[mp.Vector3]:
    """
    Convert an array or list of vectors to a list of Meep Vector3 objects.

    Pads vectors with zeros if they have fewer than 3 components.

    Parameters
    ----------
    arr : np.ndarray or list
        1D array/list (single vector) or 2D array/list (multiple vectors)
        with 1–3 components per vector.

    Returns
    -------
    list of mp.Vector3
        List of Meep Vector3 objects corresponding to the input vectors.

    Examples
    --------
    >>> array_to_Vector3([1, 2])
    [Vector3(1, 2, 0)]
    >>> array_to_Vector3([[1, 0], [0, 1]])
    [Vector3(1, 0, 0), Vector3(0, 1, 0)]
    """
    arr = np.atleast_2d(np.array(arr, dtype=float))
    padded = np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])), mode="constant")
    return [mp.Vector3(*vec) for vec in padded]


def meshgrid_points(*xi: np.ndarray | list[float], indexing: str = "ij") -> np.ndarray:
    """
    Generate a 2D array of points from N 1D coordinate arrays using a meshgrid.

    Each row of the output corresponds to a point in the N-dimensional grid.

    Parameters
    ----------
    *xi : array_like
        1D arrays representing the coordinates along each dimension.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or Cartesian ('xy') indexing of the output.

    Returns
    -------
    np.ndarray, shape (N_points, N)
        Array where each row is a point in the N-dimensional grid.

    Notes
    -----
    Equivalent to flattening the output of `numpy.meshgrid` into a list
    of coordinates for each point.

    See Also
    --------
    numpy.meshgrid

    Examples
    --------
    >>> x = np.linspace(0, 2, 3)
    >>> y = np.linspace(0, 2, 3)
    >>> meshgrid_points(x, y)
    array([[0., 0.],
           [0., 1.],
           [0., 2.],
           [1., 0.],
           [1., 1.],
           [1., 2.],
           [2., 0.],
           [2., 1.],
           [2., 2.]])
    """
    grids = np.meshgrid(*xi, indexing=indexing)  # type: ignore
    return np.vstack([g.ravel() for g in grids]).T


def find_positions(
    orbitals: np.ndarray | list[list[float]],
    lattice_vectors: np.ndarray | list[list[float]],
) -> np.ndarray:
    """
    Convert orbital positions from lattice coordinates to Cartesian coordinates.

    Parameters
    ----------
    orbitals : np.ndarray or list of lists, shape (N_points, dim_lattice)
        Positions of orbitals in lattice coordinates. Each row is a point.
    lattice_vectors : np.ndarray or list of lists, shape (dim_lattice, dim_cart)
        Lattice vectors defining the unit cell. Each row is a lattice vector.

    Returns
    -------
    np.ndarray, shape (N_points, dim_cart)
        Positions in Cartesian coordinates.

    Raises
    ------
    ValueError
        If the number of orbital coordinates does not match the number of
        lattice vectors.

    Notes
    -----
    Cartesian positions are computed as a linear combination of lattice vectors:

        r_cartesian = sum_i r_i * a_i

    where r_i are the lattice coordinates and a_i are the lattice vectors.

    Examples
    --------
    >>> orbitals = [[0, 0], [1, 0], [0, 1]]
    >>> lattice_vectors = [[1, 0], [0, 1]]
    >>> find_positions(orbitals, lattice_vectors)
    array([[0., 0.],
           [1., 0.],
           [0., 1.]])
    """
    orbitals = np.array(orbitals, dtype=float)
    lattice_vectors = np.array(lattice_vectors, dtype=float)

    if orbitals.shape[1] != lattice_vectors.shape[0]:
        raise ValueError(
            f"Number of orbital coordinates ({orbitals.shape[1]}) must match "
            f"number of lattice vectors ({lattice_vectors.shape[0]})."
        )

    return orbitals @ lattice_vectors


# ------------------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------------------


def marker_size(n: float) -> float:
    """
    Compute a marker size for scatter plots that scales inversely with n.

    The size decreases as the number of points increases, approximately as:

        size ~ 1e5 / (log(sqrt(n)) * n)

    Parameters
    ----------
    n : float
        Number of points. Must be greater than 1.

    Returns
    -------
    float
        Computed marker size suitable for use with Matplotlib scatter plots.

    Raises
    ------
    ValueError
        If n <= 1, since log(sqrt(n)) would be zero or negative.

    Examples
    --------
    >>> marker_size(10)
    14477.026
    >>> marker_size(1000)
    1157.947
    """
    if n <= 1:
        raise ValueError("`n` must be greater than 1 to compute marker size.")
    return 1e5 / (np.log(np.sqrt(n)) * n)


def draw_arrow(
    ax: Axes,
    r0: list[float],
    angle: float,
    color: str = "r",
    label: str | None = None,
    length: float = 0.2,
    label_offset: float = 0.1,
    arrow_offset: float = 0,
    arrowstyle: str = "->",
    lw: float = 2.2,
    mutation_scale: float = 14,
    fontsize: int = 11,
) -> None:
    """
    Draw a centered arrow with an optional label on a Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw on.
    r0 : list of float
        Center position [x0, y0] of the arrow in data coordinates.
    angle : float
        Angle of the arrow in radians, measured from the positive x-axis.
    color : str, optional
        Arrow and label color. Default is 'r'.
    label : str or None, optional
        Text label to display near the arrow tip. Default is None.
    length : float, optional
        Total length of the arrow. Default is 0.2.
    label_offset : float, optional
        Distance from the center to the label, along the arrow direction.
        Default is 0.1.
    arrow_offset : float, optional
        Additional offset for the arrow tip beyond half the length.
        Default is 0.
    arrowstyle : str, optional
        Matplotlib arrowstyle string. Default is '->'.
    lw : float, optional
        Line width of the arrow. Default is 2.2.
    mutation_scale : float, optional
        Scale factor for the arrowhead. Default is 14.
    fontsize : int, optional
        Font size for the label. Default is 11.
    """
    x0, y0 = r0
    half_l = length / 2

    start_pos = (
        x0 - half_l * np.cos(angle),
        y0 - half_l * np.sin(angle),
    )
    tip_pos = (
        x0 + (half_l + arrow_offset) * np.cos(angle),
        y0 + (half_l + arrow_offset) * np.sin(angle),
    )
    label_pos = (
        x0 + (half_l + label_offset) * np.cos(angle),
        y0 + (half_l + label_offset) * np.sin(angle),
    )

    ax.annotate(
        "",
        xy=tip_pos,
        xytext=start_pos,
        arrowprops=dict(
            arrowstyle=arrowstyle,
            lw=lw,
            color=color,
            mutation_scale=mutation_scale,
        ),
    )

    ax.text(
        *label_pos,
        label,
        color=color,
        fontsize=fontsize,
        ha="center",
        va="center",
        weight="bold",
    )


# ------------------------------------------------------------------------------
# Physical unit conversions
# ------------------------------------------------------------------------------


def unit_dip(a: float) -> float:
    """
    Compute the dipolar unit for a given lattice parameter.

    Defined as p_a = e * a, returned in Debyes.

    Parameters
    ----------
    a : float
        Lattice parameter of the crystal in meters.

    Returns
    -------
    float
        Dipolar unit p_a corresponding to the lattice parameter, in Debyes.
    """
    return cm2debye(E_CHARGE * a)


def debye2cm(dip_strength_in: float | np.ndarray) -> float | np.ndarray:
    """
    Convert dipole moment strength from Debyes to SI units of C·m.

    Parameters
    ----------
    dip_strength_in : float or np.ndarray
        Dipole moment strength in Debyes.

    Returns
    -------
    float or np.ndarray
        Dipole moment strength in C·m.

    Notes
    -----
    1 D ≈ 3.33564×10⁻³⁰ C·m.
    """
    return 3.33564e-30 * dip_strength_in


def cm2debye(dip_strength_in: float | np.ndarray) -> float | np.ndarray:
    """
    Convert dipole moment strength from SI units of C·m to Debyes.

    Parameters
    ----------
    dip_strength_in : float or np.ndarray
        Dipole moment strength in C·m.

    Returns
    -------
    float or np.ndarray
        Dipole moment strength in Debyes.

    Notes
    -----
    1 D ≈ 3.33564×10⁻³⁰ C·m.
    """
    return dip_strength_in / 3.33564e-30


def mpb2hz(omega_in: float | np.ndarray, a: float) -> float | np.ndarray:
    """
    Convert MPB frequencies (normalized to c/a) to SI units of Hz.

    Parameters
    ----------
    omega_in : float or np.ndarray
        Frequencies from MPB in normalized units of c/a.
    a : float
        Lattice parameter of the crystal in meters.

    Returns
    -------
    float or np.ndarray
        Frequencies in SI units of Hz.
    """
    return omega_in * C_LIGHT / a


def free_spontaneous_emission(omega_0: float, dip_strength: float, a: float) -> float:
    """
    Compute the free-space spontaneous emission rate Gamma_0 of an emitter.

    Parameters
    ----------
    omega_0 : float
        Resonant frequency of the emitter in units of c/a.
    dip_strength : float
        Dipole moment strength of the emitter in Debyes.
    a : float
        Lattice parameter of the photonic crystal in meters.

    Returns
    -------
    float
        Free-space spontaneous emission rate Gamma_0 in rad/s.

    Notes
    -----
    The free-space spontaneous emission rate is given by:

        Gamma_0 = omega_0^3 * p^2 / (3 * pi * epsilon_0 * hbar * c^3)

    where omega_0 and p are converted to SI units internally.
    """
    omega_0_si = mpb2hz(omega_0, a)
    dip_strength_si = debye2cm(dip_strength)

    return (
        omega_0_si**3
        * dip_strength_si**2
        / (3 * np.pi * EPSILON_0 * H_BAR * C_LIGHT**3)
    )
