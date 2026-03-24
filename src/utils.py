import os
import pickle
import numpy as np
import h5py
import meep as mp
import meep.mpb as mpb

from .constants import E_CHARGE, C_LIGHT, H_BAR, EPSILON_0

from typing import Any, Dict, List, Union
from matplotlib.axes import Axes


def save(data_dict: Dict[str, Any], filename: str, dirname: str = "") -> None:
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
        current working directory. If the directory does not exist, it is created.

    Returns
    -------
    None

    Notes
    -----
    Uses Python's built-in `pickle` module for serialization. Be cautious when
    unpickling data from untrusted sources.

    Raises
    ------
    OSError
        If the file cannot be written due to permission or filesystem errors.
    TypeError
        If the data_dict cannot be pickled.

    Examples
    --------
    >>> data = {"var1": 5, "var2": np.array([0, 0, 0])}
    >>> save(data, "data.pkl", "./testfolder/testsubfolder")
    """
    try:
        # Create directory if it does not exist
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            filepath = os.path.join(dirname, filename)
        else:
            filepath = filename

        # Save dictionary using pickle
        with open(filepath, "wb") as f:
            pickle.dump(data_dict, f)

        # Optional: print confirmation
        # print(f"Data successfully saved to {filepath}")

    except (OSError, TypeError) as e:
        raise RuntimeError(f"Failed to save data to '{filename}': {e}") from e


def load(filename: str) -> Dict[str, Any]:
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

    Notes
    -----
    Unlike older versions that injected variables into the global namespace,
    this function returns a dictionary. Assign variables explicitly for clarity.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    OSError
        If the file cannot be opened due to permission or filesystem errors.
    pickle.UnpicklingError
        If the file is not a valid pickle file.

    Examples
    --------
    >>> data = load("./testfolder/testsubfolder/data.pkl")
    >>> freqs = data["freqs"]
    >>> Ex = data["Ex"]
    """
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)

        # Optional: print confirmation
        # print(f"Data successfully loaded from {filename}")
        return data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: '{filename}'") from e
    except (OSError, pickle.UnpicklingError) as e:
        raise RuntimeError(f"Failed to load data from '{filename}': {e}") from e


# ...


def arrange_data(
    fields_in: np.ndarray,
    kpts_in: np.ndarray,
    r0pts_in: np.ndarray,
    freqs_in: np.ndarray,
    lattice_vecs: np.ndarray,
) -> Dict[str, Any]:
    """
    Arrange photonic crystal field data and points for real-space calculations.

    Converts k-points and r0-points to Cartesian coordinates, reshapes the
    field data by band and component, and applies Bloch phases.

    Parameters
    ----------
    fields_in : ndarray, shape (Nk*Nb*Nr, 3)
        Electric field data for all r0-points and k-points, concatenated
        along the first dimension. The last dimension indexes components
        (x, y, z).
    kpts_in : ndarray, shape (Nk, 3)
        k-points in lattice coordinates.
    r0pts_in : ndarray, shape (Nr, 3)
        Emitter positions in lattice coordinates.
    freqs_in : ndarray, shape (Nk, Nb)
        Frequencies of the photonic modes.
    lattice_vecs : ndarray, shape (3, 3)
        Lattice vectors of the crystal.

    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'frequencies': Mode frequencies (Nk x Nb).
        - 'uks': Reshaped field components without Bloch phase (Nr × Nb × Nk × 3).
        - 'Eks': Field components multiplied by Bloch phases (Nr × Nb × Nk × 3).
        - 'kpts_cartesian': k-points in Cartesian coordinates (Nk × 3).
        - 'r0pts_cartesian': r0 points in Cartesian coordinates (Nr × 3).

    Notes
    -----
    * Bloch phases are applied using the convention `exp(+i k·r)`.
    * The function reshapes fields_in from (Nk*Nb*Nr, 3) → (Nk, Nb, Nr, 3) internally.
    """

    # --- Input validation ---
    Nk = kpts_in.shape[0]
    Nr = r0pts_in.shape[0]
    Nb = freqs_in.shape[1]

    expected_size = Nk * Nb * Nr
    if fields_in.shape[0] != expected_size or fields_in.shape[1] != 3:
        raise ValueError(
            f"fields_in must have shape ({expected_size}, 3), " f"got {fields_in.shape}"
        )

    # --- Convert to Cartesian coordinates ---
    lattice_vecs_recipr = find_reciprocal_lattice_vectors(lattice_vecs, twopi=True)
    kpts_cartesian = lattice_to_cartesian(kpts_in, lattice_vecs_recipr)  # Nk x 3
    r0pts_cartesian = lattice_to_cartesian(r0pts_in, lattice_vecs)  # Nr x 3

    # --- Reshape fields: Nk x Nb x Nr x 3 ---
    uks = fields_in.reshape((Nk, Nb, Nr, 3))

    # --- Compute Bloch phase: Nk x Nr ---
    bloch_phase = np.exp(1j * np.dot(kpts_cartesian, r0pts_cartesian.T))

    # --- Apply Bloch phase to each band and component ---
    Eks = uks * bloch_phase[:, None, :, None]  # broadcasting over Nb and 3 components

    # --- Reorder axes to Nr x Nb x Nk x 3 for convenience ---
    uks = np.transpose(uks, (2, 1, 0, 3))
    Eks = np.transpose(Eks, (2, 1, 0, 3))

    return {
        "frequencies": freqs_in,
        "uks": uks,
        "Eks": Eks,
        "kpts_cartesian": kpts_cartesian,
        "r0pts_cartesian": r0pts_cartesian,
    }


def read_epsilon_file(epsilon_input_file: str) -> Dict[str, Any]:
    """
    Read dielectric permittivity (epsilon) data and lattice vectors from an HDF5 file.

    Parameters
    ----------
    epsilon_input_file : str
        Path to the HDF5 file containing epsilon data. The file is expected to
        have datasets:
        - 'data': 3D or lower-dimensional array with permittivity values.
        - 'lattice vectors': 3x3 array of lattice vectors.

    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'epsilon_data': ndarray with permittivity values.
        - 'lattice_vectors': list of lattice vectors.
        - 'resolution': list of integers specifying the number of points along each axis.
        - 'lattice_size': list of integers, placeholder for lattice size (currently filled with ones).

    Notes
    -----
    * The function pads dimensions to 3 if the data is lower-dimensional.
    * 'lattice_size' is currently a placeholder initialized to ones; you may
      need to adjust this depending on your lattice conventions.
    """
    # --- Read epsilon data and lattice vectors from HDF5 file ---
    with h5py.File(epsilon_input_file, "r") as f:
        if "data" not in f or "lattice vectors" not in f:
            raise KeyError(
                f"File {epsilon_input_file} must contain 'data' and 'lattice vectors' datasets."
            )

        epsilon_data = f["data"][()]  # type: ignore
        lattice_vectors = f["lattice vectors"][()].tolist()  # type: ignore

    # --- Determine resolution (number of points along each axis) ---
    resolution = list(epsilon_data.shape)  # type: ignore
    while len(resolution) < 3:
        resolution.append(1)

    # --- Determine lattice size (placeholder: ones) ---
    # TODO: Adjust this if you have specific lattice size conventions
    lattice_size = [1] * len(epsilon_data.shape)  # type: ignore
    while len(lattice_size) < 3:
        lattice_size.append(0)  # default 0 for missing axes

    # --- Prepare output dictionary ---
    data_dict = {
        "epsilon_data": epsilon_data,
        "lattice_vectors": lattice_vectors,
        "resolution": resolution,
        "lattice_size": lattice_size,
    }

    return data_dict


def convert_epsilon(filename: str, md) -> Dict[str, Any]:
    """
    Convert dielectric function data from an HDF5 file to a convenient format
    with positions in lattice and Cartesian coordinates.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing the dielectric function. Expected datasets:
        - 'data': N-dimensional array with epsilon values.
        - 'lattice vectors': 3x3 array of lattice vectors.
    md : mpb.MPBData
        MPBData instance used to transform the input data (e.g., to refine grid or convert to desired format).

    Returns
    -------
    data_dict : dict
        Dictionary containing:
        - 'epsilon_values' : ndarray
            Flattened dielectric function values on the corresponding positions.
        - 'positions-cartesian' : ndarray, shape (N_points, 3)
            Positions in the unit cell in Cartesian coordinates.
        - 'positions-lattice' : ndarray, shape (N_points, 3)
            Positions in the unit cell in lattice coordinates (normalized to lattice constants).
        - 'lattice_vectors' : list of lists
            Lattice vectors of the unit cell.

    Notes
    -----
    * The origin of the unit cell is placed at [0,0,0] in lattice coordinates.
    * Positions are generated according to the grid size of the converted epsilon matrix.
    * Assumes equal number of repeated periods along all directions as specified in md.multiply_size.

    Examples
    --------
    >>> filename = 'epsilon.h5'
    >>> md = mpb.MPBData(rectify=False, periods=3)
    >>> data = convert_epsilon(filename, md)
    >>> epsilon_values = data['epsilon_values']
    >>> positions_car = data['positions-cartesian']
    >>> positions_lat = data['positions-lattice']
    """
    # --- Read epsilon data and lattice vectors from HDF5 ---
    with h5py.File(filename, "r") as f:
        if "data" not in f or "lattice vectors" not in f:
            raise KeyError(
                f"File {filename} must contain 'data' and 'lattice vectors' datasets."
            )
        epsilon_data = f["data"][()]  # type: ignore
        lattice_vectors = f["lattice vectors"][()].tolist()  # type: ignore

    # --- Number of periods along each axis (assumes equal for all directions) ---
    periods = md.multiply_size[0]

    # --- Convert to MPBArray and transform using MPBData ---
    epsilon_MPBArray = mpb.MPBArray(epsilon_data, lattice=lattice_vectors)
    epsilon_matrix = md.convert(epsilon_MPBArray)

    # --- Grid size of converted data ---
    grid_size = np.array(epsilon_matrix.shape)

    # --- Generate pixel positions in the unit cell ---
    axes = [np.arange(n) for n in grid_size]
    positions_pixels = meshgrid_points(*axes)  # shape: (N_points, ndim)

    # --- Shift origin to [0,0,0] and normalize by periods/grid_size ---
    center = grid_size / 2
    spatial_shift = -center  # shape: (ndim,)
    normalization_factor = grid_size / periods
    positions_lat = (positions_pixels + spatial_shift) / normalization_factor

    # --- Convert lattice coordinates to Cartesian ---
    positions_car = lattice_to_cartesian(positions_lat, lattice_vectors)

    # --- Flatten epsilon values ---
    epsilon_values = epsilon_matrix.flatten()

    # --- Prepare output dictionary ---
    data_dict = {
        "epsilon_values": epsilon_values,
        "positions-cartesian": positions_car,
        "positions-lattice": positions_lat,
        "lattice_vectors": lattice_vectors,
    }

    return data_dict


### Other auxiliary functions
def find_reciprocal_lattice_vectors(
    lattice_vectors: Union[np.ndarray, List[List[float]]], twopi: bool = True
) -> np.ndarray:
    """
    Compute the reciprocal lattice vectors corresponding to a set of direct lattice vectors.

    The reciprocal lattice vectors b1, b2, b3 satisfy:
        a_i · b_j = 2π δ_ij  (if twopi=True)
        or
        a_i · b_j = δ_ij     (if twopi=False)

    Parameters
    ----------
    lattice_vectors : array_like, shape (3, 3)
        Direct lattice vectors a1, a2, a3. Each row corresponds to a lattice vector.
    twopi : bool, optional
        If True (default), reciprocal vectors are multiplied by 2π. If False,
        returns the reciprocal vectors without the 2π factor.

    Returns
    -------
    reciprocal_vectors : list of lists, shape (3, 3)
        Reciprocal lattice vectors b1, b2, b3 as a list of 3 lists.

    Notes
    -----
    This uses the standard definition from solid state physics / photonic crystals:
    b_i = 2π * (a_j × a_k) / (a1 · (a2 × a3)), with cyclic permutation of i,j,k.
    See: Joannopoulos et al., "Photonic Crystals: Molding the Flow of Light", p. 235.

    Examples
    --------
    >>> lattice_vectors = [[1.0, 0.0, 0.0],
    ...                    [0.0, 1.0, 0.0],
    ...                    [0.0, 0.0, 1.0]]
    >>> b1, b2, b3 = find_reciprocal_lattice_vectors(lattice_vectors)
    >>> b1
    [6.283185307179586, 0.0, 0.0]
    """
    # Convert input to ndarray
    a = np.array(lattice_vectors, dtype=float)
    if a.shape != (3, 3):
        raise ValueError(f"lattice_vectors must be of shape (3,3), got {a.shape}")

    # Reciprocal lattice vectors: b_i = inv(a).T * (optionally 2π)
    b = np.linalg.inv(a).T
    if twopi:
        b *= 2 * np.pi

    return b


def lattice_to_cartesian(
    orbitals: Union[np.ndarray, List[List[float]]],
    lattice_vectors: Union[np.ndarray, List[List[float]]],
) -> np.ndarray:
    """
    Convert positions from lattice coordinates to Cartesian coordinates.

    Parameters
    ----------
    orbitals : array_like, shape (N_points, ndim)
        Positions in lattice coordinates. Each row is a point in the unit cell.
        ndim can be 1, 2, or 3.
    lattice_vectors : array_like, shape (3, 3)
        Lattice vectors of the crystal. Each row corresponds to a lattice vector (a1, a2, a3).
        If the lattice dimension is < 3, pad missing vectors as standard:
            2D: a3 = [0,0,1], 1D: a2 = [0,1,0], a3 = [0,0,1]

    Returns
    -------
    positions_cartesian : ndarray, shape (N_points, 3)
        Positions in Cartesian coordinates.

    Notes
    -----
    Converts each position `r` in lattice coordinates to Cartesian coordinates using:
        r_cartesian = r1*a1 + r2*a2 + r3*a3

    Examples
    --------
    >>> orbitals = [[0,0], [0,1], [1,0], [1,1]]
    >>> lattice_vectors = [[np.sqrt(3)/2,0.5,0],
    ...                    [np.sqrt(3)/2,-0.5,0],
    ...                    [0,0,1]]
    >>> lattice_to_cartesian(orbitals, lattice_vectors)
    array([[0.        , 0.        , 0.        ],
           [0.8660254 , -0.5       , 0.        ],
           [0.8660254 ,  0.5       , 0.        ],
           [1.73205081, 0.        , 0.        ]])
    """
    # Convert inputs to numpy arrays
    orbitals = np.array(orbitals, dtype=float)
    lattice_vectors = np.array(lattice_vectors, dtype=float)

    _, ndim = orbitals.shape

    # Pad orbitals to 3D if necessary
    if ndim < 3:
        orbitals = np.pad(orbitals, ((0, 0), (0, 3 - ndim)), mode="constant")

    # Compute Cartesian positions (vectorized)
    positions_cartesian = orbitals @ lattice_vectors

    return positions_cartesian


def array_to_Vector3(
    arr: Union[np.ndarray, List[List[float]], List[float]],
) -> List[mp.Vector3]:
    """
    Convert an array or list of lattice vectors to a list of Meep Vector3 objects.

    Pads vectors with zeros if they have less than 3 components.

    Parameters
    ----------
    arr : array_like
        1D array/list (single vector) or 2D array/list (multiple vectors)
        with 1-3 components per vector.

    Returns
    -------
    vectors : list of mp.Vector3
        List of Meep Vector3 objects corresponding to input vectors.

    Examples
    --------
    >>> array_to_Vector3([1, 2])
    [Vector3(1, 2, 0)]
    >>> array_to_Vector3([[1,0], [0,1]])
    [Vector3(1,0,0), Vector3(0,1,0)]
    """
    arr = np.atleast_2d(np.array(arr, dtype=float))
    # Pad all vectors to 3 components in one line
    padded = np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])), mode="constant")
    # Convert to list of Vector3
    return [mp.Vector3(*vec) for vec in padded]


def meshgrid_points(
    *xi: Union[np.ndarray, List[float]], indexing: str = "ij"
) -> np.ndarray:
    """
    Generate a 2D array of points from N 1D coordinate arrays using a meshgrid.

    Each row of the output corresponds to a point in the N-dimensional grid.

    Parameters
    ----------
    *xi : array_like
        1-D arrays representing the coordinates along each dimension.
    indexing : {'xy', 'ij'}, optional
        Matrix ('ij', default) or Cartesian ('xy') indexing of the output.

    Returns
    -------
    points : ndarray, shape (N_points, N)
        Array where each row is a point in the N-dimensional grid.

    Notes
    -----
    This function is equivalent to flattening the output of `numpy.meshgrid`
    into a list of coordinates for each point.

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
    # Create N-dimensional meshgrid
    grids = np.meshgrid(*xi, indexing=indexing)  # type: ignore

    # Flatten each grid and stack as columns, then transpose to get points
    points = np.vstack([g.ravel() for g in grids]).T

    return points


def find_positions(
    orbitals: Union[np.ndarray, List[List[float]]],
    lattice_vectors: Union[np.ndarray, List[List[float]]],
) -> np.ndarray:
    """
    Convert positions from lattice coordinates to Cartesian coordinates.

    Parameters
    ----------
    orbitals : array_like, shape (N_points, dim_lattice)
        Positions of orbitals in lattice coordinates. Each row is a point.
    lattice_vectors : array_like, shape (dim_lattice, dim_cart)
        Lattice vectors defining the unit cell. Each row corresponds to a lattice vector.

    Returns
    -------
    positions : ndarray, shape (N_points, dim_cart)
        Positions in Cartesian coordinates.

    Notes
    -----
    The Cartesian positions are computed as a linear combination of lattice vectors:
        r_cartesian = sum_i (r_i * a_i)
    where r_i are the lattice coordinates and a_i are the lattice vectors.

    Examples
    --------
    >>> orbitals = [[0,0], [1,0], [0,1]]
    >>> lattice_vectors = [[1,0], [0,1]]
    >>> find_positions(orbitals, lattice_vectors)
    array([[0., 0.],
           [1., 0.],
           [0., 1.]])
    """
    orbitals = np.array(orbitals, dtype=float)
    lattice_vectors = np.array(lattice_vectors, dtype=float)

    # Validate dimensions
    if orbitals.shape[1] != lattice_vectors.shape[0]:
        raise ValueError(
            f"Number of orbital coordinates ({orbitals.shape[1]}) "
            f"must match number of lattice vectors ({lattice_vectors.shape[0]})"
        )

    # Compute Cartesian positions (vectorized)
    positions = orbitals @ lattice_vectors

    return positions


def marker_size(n: float) -> float:
    """
    Compute a marker size for plotting that scales with the number of points.

    The size decreases as the number of points increases, approximately as:
        size ~ 1e5 / (log(sqrt(n)) * n)

    Parameters
    ----------
    n : float
        Number of points. Must be greater than 1.

    Returns
    -------
    size : float
        Computed marker size suitable for plotting.

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
        raise ValueError("n must be greater than 1 to compute marker size.")

    return 1e5 / (np.log(np.sqrt(n)) * n)


def draw_arrow(
    ax: Axes,
    R0: list[float],
    angle: float,
    color: str = "r",
    label: str = None,
    length: float = 0.2,
    label_offset: float = 0.1,
    arrow_offset: float = 0,
    arrowstyle: str = "->",
    lw: float = 2.2,
    mutation_scale: float = 14,
    fontsize: int = 11,
) -> None:
    """
    Draw a clean arrow centered at R0 with a label on a Matplotlib axis.
    """
    x0, y0 = R0
    half_L = length / 2

    # Calculate the centered start and tip positions
    start_pos = (
        x0 - half_L * np.cos(angle),
        y0 - half_L * np.sin(angle),
    )
    tip_pos = (
        x0 + (half_L + arrow_offset) * np.cos(angle),
        y0 + (half_L + arrow_offset) * np.sin(angle),
    )

    # Label position remains relative to the tip
    label_pos = (
        x0 + (half_L + label_offset) * np.cos(angle),
        y0 + (half_L + label_offset) * np.sin(angle),
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


def unit_dip(a: float) -> float:
    """
    Compute the dipolar unit for the corresponding lattice parameter `a`. It is
    defined as `p_a = e*a`. The result is returned in Debyes.

    Parameters
    ----------
    a : float
        Lattice parameter of the crystal in meters.

    Returns
    -------
    p_a : float
        Dipolar unit corresponding to the lattice parameter in Debyes.
    """
    p_a = cm2debye(E_CHARGE * a)
    return p_a


def debye2cm(dip_strength_in: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Transform dipole moments strength from Debyes to SI units of C·m.

    Parameter
    ---------
    dip_strength_in : float, np.ndarray
        Dipole moment strength in Debyes.

    Returns
    -------
    dip_strength_out : float, np.ndarray
        Dipole moment strength in C·m.

    Notes
    -----
    1 D ≈ 3.33564e-30 Cm
    """
    dip_strength_out = 3.33564e-30 * dip_strength_in
    return dip_strength_out


def cm2debye(dip_strength_in: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Transform dipole moments strength from SI units of C·m to Debyes.

    Parameter
    ---------
    dip_strength_in : float, np.ndarray
        Dipole moment strength in C·m.

    Returns
    -------
    dip_strength_out : float, np.ndarray
        Dipole moment strength in Debyes.

    Notes
    -----
    1 D ≈ 3.33564e-30 Cm
    """
    dip_strength_out = dip_strength_in / 3.33564e-30
    return dip_strength_out


def mpb2hz(ω_in: Union[float, np.ndarray], a: float) -> np.ndarray:
    """
    Translate MPB frequencies (normalized to c/a) into SI units of Hz.

    Parameters
    ----------
    ω_in : float, np.ndarray
        Frequencies obtained from MPB (normalized by c/a).

    a : float
        Lattice parameter of the crystal in meters.

    Returns
    -------
    ω_out: np.ndarray
        Frequencies in SI units of Hz.
    """
    ω_out = ω_in * C_LIGHT / a
    return ω_out


def free_spontaneous_emission(ω_0: float, dip_strength: float, a: float) -> float:
    """
    Compute the (average) free-space spontaneous emission rate Γ₀ of the emitter(s)
    in rad/s.

    Parameters
    ----------
    ω_0 : float
        (Average) Resonant frequency of the emitter(s) in c/a.

    dip_strength : float
        (Average) Dipole moment strength of the emitter(s) in Debeys.

    a : float
        Lattice parameter of the photonic crystal in meters.

    Returns
    -------
    Γ_0 : float
        (Average) free-space spontaneous emission rate of the emitter(s) in rad/s.

    Notes
    -----
    We use the following formula for obtaining the free-space spontaneous emission
    rate:
        Γ₀ = ω₀³ p² / (3 π ε₀ hbar c³),
    """
    ω_0_si = mpb2hz(ω_0, a)
    dip_strength_si = debye2cm(dip_strength)

    Γ_0 = ω_0_si**3 * dip_strength_si**2 / (3 * np.pi * EPSILON_0 * H_BAR * C_LIGHT**3)
    return Γ_0
