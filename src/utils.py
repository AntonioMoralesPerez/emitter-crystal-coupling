import os
import pickle
import numpy as np
import h5py
import meep as mp
import meep.mpb as mpb


def save(data_dict, filename, dirname=""):
    """
    Save a dictionary of variables to a pickle file.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping variable names (str) to values (any Python object).
        Example: {"freqs": freqs, "Ex": Ex}
    filename : str
        Name of the pickle file to save (e.g. 'data.pkl').
    dirname : str, optional
        Directory where the file should be saved. If empty, saves in the
        current directory. If the directory does not exist, it is created.

    Returns
    -------
    None

    Notes
    -----
    Uses `pickle` for serialization. Be cautious when unpickling data from
    untrusted sources.

    Examples
    --------
    >>> data = {"var1": 5, "var2": np.array([0,0,0])}
    >>> save(data, "data.pkl", "./testfolder/testsubfolder")
    """
    if dirname:
        os.makedirs(dirname, exist_ok=True)
        filename = os.path.join(dirname, filename)

    with open(filename, "wb") as f:
        pickle.dump(data_dict, f)


def load(filename):
    """
    Load variables from a pickle file into a dictionary.

    Parameters
    ----------
    filename : str
        Path to the pickle file (e.g. './testfolder/testsubfolder/data.pkl').

    Returns
    -------
    dict
        Dictionary containing the stored variables, with the same keys as
        passed to `save`.

    Notes
    -----
    Unlike older versions that set globals directly, this function returns
    a dictionary. You should assign variables explicitly for clarity.

    Examples
    --------
    >>> data = load("./testfolder/testsubfolder/data.pkl")
    >>> freqs = data["freqs"]
    >>> Ex = data["Ex"]
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


# ...


def arrange_data(fields_in, kpts_in, r0pts_in, freqs_in, lattice_vecs):
    """
    TODO: Document function
    """

    # Get the list of k-points in cartesian coordiantes
    # ..
    lattice_vecs_recipr = find_reciprocal_lattice_vectors(lattice_vecs, twopi=True)
    # ..
    kpts_cartesian = lattice_to_cartesian(kpts_in, lattice_vecs_recipr)

    # Get the list of r0-points in cartesian coordiantes
    r0pts_cartesian = lattice_to_cartesian(r0pts_in, lattice_vecs)

    #
    number_of_r0pts = len(r0pts_in)
    number_of_kpts = len(kpts_in)

    # Separate by components and do a first reshape of the output
    ux_aux = fields_in[:, 0].reshape((number_of_r0pts, -1, number_of_kpts), order="F")
    uy_aux = fields_in[:, 1].reshape((number_of_r0pts, -1, number_of_kpts), order="F")
    uz_aux = fields_in[:, 2].reshape((number_of_r0pts, -1, number_of_kpts), order="F")
    # .. after this reshaping we obtain the following:
    #   ui_aux[:,0,:] component i=x,y,z of the 1-st band
    #   ui_aux[:,1,:] component i=x,y,z of the 2-nd band
    #        ...
    #   ui_aux[:,n,:] component i=x,y,z of the n-th band

    # Get the number of bands
    number_of_bands = np.shape(ux_aux)[1]

    # Do second reshape to prepare data for calculation of the g(r0,r)
    ux_list = []
    uy_list = []
    uz_list = []
    for i in range(number_of_bands):
        ux_list.append(np.transpose(ux_aux[:, i, :]))
        uy_list.append(np.transpose(uy_aux[:, i, :]))
        uz_list.append(np.transpose(uz_aux[:, i, :]))
    # ..
    ux = np.array(ux_list)
    uy = np.array(uy_list)
    uz = np.array(uz_list)

    # Multiply by the bloch phase (we chose the positive sign)
    # bloch_phase = np.exp(-1j * (kpts_cartesian @ np.transpose(r0pts_cartesian)))
    bloch_phase = np.exp(+1j * (kpts_cartesian @ np.transpose(r0pts_cartesian)))

    Ex = ux * bloch_phase
    Ey = uy * bloch_phase
    Ez = uz * bloch_phase

    # Prepare the output as a dictionary:
    d = dict()
    d["frequencies"] = freqs_in
    d["ux"] = ux
    d["uy"] = uy
    d["uz"] = uz
    # ..
    d["Ex"] = Ex
    d["Ey"] = Ey
    d["Ez"] = Ez
    # ..
    d["kpts-cartesian"] = kpts_cartesian
    d["r0pts-cartesian"] = r0pts_cartesian

    return d


def read_epsilon_file(epsilon_input_file):
    """ """

    # Read the epsilon data and lattice vectors form input file
    with h5py.File(epsilon_input_file, "r") as f:
        epsilon_data = f["data"][()]
        lattice_vectors = f["lattice vectors"][()].tolist()

    # Determine the resolution by looking at the shape of epsilon.h5 "data"
    resolution = list(np.shape(epsilon_data))
    # ..
    while len(resolution) < 3:
        resolution += [1]

    # Determine lattice size by looking at the shape of epsilon.h5 "data"
    # TODO: review and document this! I am assuming things here...
    lattice_size = list(np.ones(len(np.shape(epsilon_data))))
    # ..
    while len(lattice_size) < 3:
        lattice_size += [0]

    # Prepare the output as a dictionary:
    d = dict()
    d["epsilon data"] = epsilon_data
    d["lattice vectors"] = lattice_vectors
    d["resolution"] = resolution
    d["lattice size"] = lattice_size

    return d  # epsilon_data, lattice_vectors, resolution


def convert_epsilon(filename, md):
    """
    Return converted epsilon data in a convenient format

    Parameters
    ----------
    filename : str
        Name of the h5 file contaning the dielectric funcyion
    md : mpb.MPBData() instance
        mpb.MPBData instance to transform the input data

    Returns
    -------
        d : dictionary

        d['epsilon values'] = epsilon_values
        d['positions-cartesian'] = positions_car
        d['positions-lattice'] = positions_lat
        d['lattice vectors'] = lattice_vectors

        epsilon_values : array
            Dielectric function values on the corresponding positions
        positions_car : array
            Positions in the unit cell in cartesian coordinates
        positions_lat : array
            Positions in the unit cell in lattice coordinates

    Notes
    -----
    See Also
    --------
    Examples
    --------
    >>> # The name of the h5 file
    >>> filename = 'epsilon.h5'
    >>> # Create a MPBData instace to transform the dielectric function
    >>> md = mpb.MPBData(rectify=False, periods=3)
    >>> epsilon_values, positions_car, positions_lat = convert_epsilon(filename, md)

    """
    # Read data from epsilon file
    with h5py.File(filename, "r") as f:
        epsilon_data = f["data"][()]
        lattice_vectors = f["lattice vectors"][()].tolist()
    # Get `periods` from the input MPBData instance
    # NOTE Here, I am assuming equal number of periods to be repeated along all directions
    periods = md.multiply_size[0]
    # ...
    # Create the MPBArray to feed mpb.MPBData.covert() function
    epsilon_MPBArray = mpb.MPBArray(epsilon_data, lattice=lattice_vectors)
    # Convert according to the MPBData instance defined above (output is a numpy.ndarray)
    epsilon_matrix = md.convert(epsilon_MPBArray)
    # ...
    # Obtain the grid size of the converted data = periods x resolution
    grid_size = list(np.shape(epsilon_matrix))
    # Place the origin of the unit cell in [0,0,0] and normalized to the lattice constant
    # .. define the spatial shift
    center = np.array(grid_size) / 2
    spatial_shift = -np.tile(center, (np.prod(grid_size), 1))
    # .. define the normalization factor
    normalization_factor = np.array(grid_size) / periods
    # TODO: Check if the choice above is consistent with all geometries
    # ...
    # Get the positions in pixel coordinates
    iteration_range = np.arange(len(grid_size))
    rg = list(
        map(lambda i: np.arange(grid_size[i]), iteration_range)
    )  # equivalent to [xg_1, xg_2, ...] with xg_i = np.arange(grid_size[i])
    positions_pixels = meshgrid_points(
        *rg
    )  # indexing = 'xy'?          # equivalent to mf.meshgrid_points(xg_1, xg_2, ...)
    # Get the positions in lattice coordinates
    positions_lat = (positions_pixels + spatial_shift) / normalization_factor
    # Get the positions in cartesian coordinates
    positions_car = lattice_to_cartesian(positions_lat, lattice_vectors)
    # Get the epsilon values
    epsilon_values = epsilon_matrix.flatten()

    # Prepare the output as a dictionary:
    d = dict()
    d["epsilon values"] = epsilon_values
    d["positions-cartesian"] = positions_car
    d["positions-lattice"] = positions_lat
    d["lattice vectors"] = lattice_vectors

    return d


### Other auxiliary functions
def find_reciprocal_lattice_vectors(lattice_vectors, twopi=True):
    """
        Return the reciprocal lattice vectors b1, b2, b3 corresponding to the lattice
        vectors a1, a2, a3.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
        See Joannopoulos, Photonic Crystals: Molding the flow of light p.235

    See Also
    --------

    Examples
    --------
    >>>
    >>> lattice_vectors = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    >>> b1, b2, b3 = find_reciprocal_lattice_vectors(lattice_vectors)
    """
    output = np.transpose(np.linalg.inv(lattice_vectors))

    if twopi:
        output = 2 * np.pi * output

    return output.tolist()


def lattice_to_cartesian(orbitals, lattice_vectors):
    """
    Return the positions in cartesian coordinates given the positions in the
    lattice coordinates and the lattice vectors

    Parameters
    ----------
    orbitals : array
        Array with the positions in lattice coordinates
    lattice_vectors : array
        3x3 array with each row containing the i-th lattice vector (ai)
            lattice_vectors = [a1, a2, a3]
        If the dimension of the lattice is 2: a3 -> [0,0,1]
        If the dimension of the lattice is 1: a2 -> [0,1,0] and a3 --> [0,0,1]

    Returns
    -------
    positions : array
        Array with the positions in the cartesian coordinates

    Notes
    -----

    See Also
    --------

    Examples
    ----------
    >>> orbitals = [[0,0], [0,1], [1,0], [1,1]]
    >>> lattice_vectors = [[np.sqrt(3)/2,0.5,0], [np.sqrt(3)/2,-0.5,0], [0,0,1]]
    >>> positions = lattice_to_cartesian(orbitals, lattice_vectors)
    array([[ 0.        ,  0.        ,  0.        ],
           [ 0.8660254 , -0.5       ,  0.        ],
           [ 0.8660254 ,  0.5       ,  0.        ],
           [ 1.73205081,  0.        ,  0.        ]])
    """

    # Transform to numpy.array (in case it is not)
    orbitals = np.array(orbitals)

    # get number of orbitals
    N_orbitals = np.shape(orbitals)[0]

    # if 2D fill with zeros
    while np.shape(orbitals)[1] < 3:
        orbitals = np.c_[orbitals, np.zeros(N_orbitals)]

    # Find the positions (in cartesian coordinates)
    output = np.zeros(np.shape(orbitals))
    for i in range(3):
        output = output + orbitals[:, i].reshape(N_orbitals, 1) * lattice_vectors[i]

    return output


def array_to_Vector3(arr):
    """
    Take a list or a numpy.array and returns a meep.Vector3 object out of it.
    If the input array has less than 3 columns, it fills with zeros

    Parameters
    ----------
    arr : array_like
        numpy array or list with 1, 2, or 3 columns and arbitrary (>=1) rows

    Returns
    -------

    Notes
    -----

    See Also
    --------

    Examples
    ----------

    """
    dims = np.shape(arr)

    if len(dims) == 1:

        output = mp.Vector3(*arr)

    else:

        output = list(map(lambda i: mp.Vector3(*arr[i, :]), np.arange(dims[0])))

    return output


def meshgrid_points(*xi, indexing="ij"):
    """
    Returns a list of points created from a meshgrid.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of output.

    Returns
    -------

    Notes
    -----

    See Also
    --------

    https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

    https://stackoverflow.com/questions/12864445/how-to-convert-the-output-of-meshgrid-to-the-corresponding-array-of-points


    Examples
    ----------
    >>> x = np.linspace(0, 2, 3)
    >>> y = np.linspace(0, 2, 3)

    >>> meshgrid_points(x, y)
    [[0. 0.]
    [0. 1.]
    [0. 2.]
    [1. 0.]
    [1. 1.]
    [1. 2.]
    [2. 0.]
    [2. 1.]
    [2. 2.]]
    """

    # Define a grid using the meshgrid function
    grid = np.meshgrid(*xi, indexing=indexing)

    # Transform the meshgrid output into a list of points
    output = np.vstack(list(map(np.ravel, grid))).T

    return output


def find_positions(orbitals, lattice_vectors):
    # get lattice's dimension
    lattice_dimension = np.shape(lattice_vectors)[0]

    # get number of orbitals
    N_orbitals = np.shape(orbitals)[0]

    # Find the positions (in cartesian coordinates)
    positions = np.zeros(np.shape(orbitals))
    for i in range(lattice_dimension):
        positions = (
            positions + orbitals[:, i].reshape(N_orbitals, 1) * lattice_vectors[i]
        )

    ### Output ...
    return positions


marker_size = lambda n: 2e5 / (np.log(np.sqrt(n)) * n)
