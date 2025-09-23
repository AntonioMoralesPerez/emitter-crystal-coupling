import os, glob
import numpy as np

import meep as mp
from meep import mpb

from scipy.sparse import csc_matrix

# from scipy.sparse.linalg import expm, expm_multiply

import h5py
import pickle

# ...


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


def compute_gr0r(E, dipole_moment, detuning_freq, kpts, rpts):
    """
    TODO: Document the function
    """

    # Check that the inputs have the proper dimensions
    # .. for the kpts
    kpts = np.array(kpts)
    while np.shape(kpts)[1] < 3:
        kpts = np.c_[kpts, np.zeros(np.shape(kpts)[0])]

    # .. for the rpts
    rpts = np.array(rpts)
    while np.shape(rpts)[1] < 3:
        rpts = np.c_[rpts, np.zeros(np.shape(rpts)[0])]

    # check that the r-space discretization is the same as the k-space one
    # if np.shape(kpts)[0] != np.shape(rpts)[0]:
    #     raise ValueError(
    #         "The number of k-points and r-points must be the same. "
    #         f"Got {np.shape(kpts)[0]} k-points and {np.shape(rpts)[0]} r-points."
    #     )

    # Perform the Fourier transform
    # .. round fields to desired precision
    decimals = 16  # int(-np.log10(tolerance))
    Ex_rounded = np.round(E[0], decimals)
    Ey_rounded = np.round(E[1], decimals)
    Ez_rounded = np.round(E[2], decimals)

    # .. compute the dipole times the field
    dipole_times_field = (
        Ex_rounded * dipole_moment[0]
        + Ey_rounded * dipole_moment[1]
        + Ez_rounded * dipole_moment[2]
    )

    # .. define the normalization factor
    N = np.shape(kpts)[0]  # number of k-points
    prefactor = -1j * np.sqrt(detuning_freq / (2 * N))

    # .. define the exponential factor
    exponential_factor = np.exp(-1j * (kpts @ np.transpose(rpts)))

    # Output
    gr0r = prefactor * np.transpose(dipole_times_field) @ exponential_factor

    return gr0r


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


def convert_fields(filename, md, bloch_phase=True):
    """
    Return the converted fields data in a convenient format

    Parameters
    ----------
    filename : str
        Name of the h5 file contaning the fields
    md : mpb.MPBData() instance
        mpb.MPBData instance to transform the input data

    Returns
    -------
        d : dictionary

        d['field values'] = field_values
        d['positions-cartesian'] = positions_car
        d['positions-lattice'] = positions_lat
        d['lattice vectors'] = lattice_vectors
        d['k-point'] = k_point

        field_values : list [field_x_values : array, field_y_values : array, field_z_values : array]
            Field values on the corresponding positions
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
    >>> filename = 'e.k11.b01.h5'
    >>> # Create a MPBData instace to transform the fields
    >>> md = mpb.MPBData(rectify=False, periods=3)
    >>> field_values, positions_car, positions_lat = convert_field(filename, md)

    """
    # Read data from fields file
    with h5py.File(filename, "r") as f:
        field_x_real_data = f["x.r"][()]
        field_x_imag_data = f["x.i"][()]
        field_y_real_data = f["y.r"][()]
        field_y_imag_data = f["y.i"][()]
        field_z_real_data = f["z.r"][()]
        field_z_imag_data = f["z.i"][()]
        lattice_vectors = f["lattice vectors"][()].tolist()
        k_point = f["Bloch wavevector"][()]
    # .. get resolution from input data
    input_resolution = list(np.shape(field_x_real_data))
    # Get `periods` from the input MPBData instance
    # NOTE Here, I am assuming equal number of periods to be repeated along all directions
    periods = md.multiply_size[0]
    # ...
    # Manipulate the output to feed the mpb.MPBData.convert() function
    field_x_data_aux = (field_x_real_data + 1j * field_x_imag_data).ravel()
    field_y_data_aux = (field_y_real_data + 1j * field_y_imag_data).ravel()
    field_z_data_aux = (field_z_real_data + 1j * field_z_imag_data).ravel()
    # 1) Iterleave field components
    # (see https://stackoverflow.com/questions/7946798/interleave-multiple-lists-of-the-same-length-in-python)
    lists = [field_x_data_aux, field_y_data_aux, field_z_data_aux]
    field_data_aux = [val for tup in zip(*lists) for val in tup]
    # 2) Reshape the data to pass to MPBData
    # .. the dims to reshape are like [resolution_in_x, resolution_in_y, resolution_in_z, 3]
    while len(input_resolution) < 3:
        input_resolution += [
            1
        ]  # add [1] if the system is 2D: e.g., input_resolution = [32,32,1]
    dims = np.append(
        input_resolution, [3]
    ).tolist()  # add a [3] to account for the three field components
    # .. reshaping:
    field_data = np.reshape(field_data_aux, dims)
    # ...
    # Create the MPBArray
    field_MPBArray = mpb.MPBArray(
        field_data,
        lattice=lattice_vectors,
        kpoint=mp.Vector3(*k_point),
        bloch_phase=bloch_phase,
    )
    # Convert according to the MPBData instance defined above (output is a numpy.ndarray)
    # This is equivalent to the 'data-new' with no rectification
    field_matrix = md.convert(field_MPBArray)
    # ...
    # Obtain the grid size of the converted data
    grid_size = list(np.shape(field_matrix)[0:-1])
    # Define spatial shift and normalization factor to place the origin of the unit cell in [0,0,0] and normalized to the lattice constant
    # TODO: Check if the choice below consistent with all geometries
    center = np.array(grid_size) / 2
    spatial_shift = -np.tile(center, (np.prod(grid_size), 1))
    normalization_factor = np.array(grid_size) / periods
    # ...
    # Get the positions in pixel coordinates
    iteration_range = np.arange(len(grid_size))
    rg = list(
        map(lambda i: np.arange(grid_size[i]), iteration_range)
    )  # equivalent to [xg_1, xg_2, ...] with xg_i = np.arange(grid_size[i])
    positions_pixels = meshgrid_points(
        *rg
    )  # indexing = 'xy'?          # equivalet to mf.meshgrid_points(xg_1, xg_2, ...)
    # Get the positions in lattice coordinates
    positions_lat = (positions_pixels + spatial_shift) / normalization_factor
    # Get the positions in cartesian coordinates
    positions_car = lattice_to_cartesian(positions_lat, lattice_vectors)
    # Get the field values
    field_x_values = field_matrix[:, :, 0].flatten()
    field_y_values = field_matrix[:, :, 1].flatten()
    field_z_values = field_matrix[:, :, 2].flatten()
    field_values = [field_x_values, field_y_values, field_z_values]

    # Prepare the output as a dictionary:
    d = dict()
    d["field values"] = field_values
    d["positions-cartesian"] = positions_car
    d["positions-lattice"] = positions_lat
    d["lattice vectors"] = lattice_vectors
    d["k-point"] = k_point

    return d  # field_values, positions_car, positions_lat


# def delete_fields(bands=[]):
#     """
#         Delete fields in current folder

#     Parameters
#     ----------
#     bands : list
#         list of bands whose associated fields we want to keep

#     Returns
#     -------

#     Notes
#     -----

#     See Also
#     --------

#     Examples
#     ----------

#     """
#     # list filenames of all computed fields
#     h5fields = glob.glob("*.k*.b*.h5")

#     # get filenames of computed fields that we want to keep
#     h5fields_to_keep = []

#     if len(bands) > 0:
#         for i in bands:
#             if i < 10:
#                 h5fields_to_keep.extend(glob.glob(f"*b0{i}*.h5"))
#             elif i >= 10 and i < 100:
#                 h5fields_to_keep.extend(glob.glob(f"*b{i}*.h5"))
#             else:
#                 print(
#                     "How are the bands labeled when there are more that 100? Who knows :)"
#                 )

#     # get filenames of computed fields that we want to eliminate
#     h5fields_to_delete = list(set(h5fields) - set(h5fields_to_keep))

#     # eliminate the unwanted data
#     for name in h5fields_to_delete:
#         os.remove(name)


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
        Array with the positions in lattice coordiantes
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


def MonkhorstPack(N, b1, b2, b3=None, coordinates="lattice"):
    """
        Return a Monkhorst-Pack sampling of the first Brillouin zone

    Parameters
    ----------
    N : int or list
        Number of points assumed in each direction.
        If `N` is a scalar, we assume equal number of points in each direction.
    b1, b2, b3 : array_like
        Reciprocal lattice vectors.  The third one is optional.
    coordinates: {'lattice', 'cartesian'}, optional

    Returns
    -------

    Notes
    -----
    The k-points given by the Monkhorst-Pack method read:

        k_{n1,...,ndim} = \\sum_{i=1}^{dim} bi*(2*ni-Ni-1)/(2*Ni),

    where bi is the i-th reciprocal lattice vector, Ni is the total number of sites
    in the i-th direction and ni are integers

    Original reference:
    H. J. Monkhorst and J. D. Pack. Special points for brillouin-zone integrations.
    Phys. Rev. B, 13:5188–5192, Jun 1976. doi:10.1103/PhysRevB.13.5188.

    Important!
    To include the origin (k=[0,0,0]), consider an odd number of points per direction.
    [see TODO (1)]

    TODO:
    1) Sometimes some shift of the k-points is include as parameter to solve this
    problem of the origin not been including for even `number_of_kpts`.
    Review works:
    · https://www.smcm.iqfr.csic.es/docs/vasp/node58.html
    · https://sisl.readthedocs.io/en/latest/api/generated/sisl.physics.MonkhorstPack.html
    · https://github.com/zerothi/sisl/releases
    2) The employed sampling does not restrict to the first Brillouin zone. It would
    be nice to implement some algorithm to map back those points that lie outside
    the first Brillouin zone into it.

    See Also
    --------

    Examples
    --------
    >>> N = 2
    >>> b1 = [np.sqrt(3)/2, +0.5, 0]
    >>> b2 = [np.sqrt(3)/2, -0.5, 0]
    >>> MonkhorstPack(N, b1, b2)
    [[-0.25 -0.25]
    [-0.25  0.25]
    [ 0.25 -0.25]
    [ 0.25  0.25]]
    """

    # Define the i-th component of the k-point
    ki = lambda ni, Ni: (2 * ni - Ni - 1) / (2 * Ni)

    if b3 is None:  # 2D CASE ...................................................
        # Promote `N` to an np.array with the dimensions of the lattice.
        if np.isscalar(N):
            N = np.repeat(N, 2)
        # Get the number of kpoints in each direction
        N1 = N[0]
        N2 = N[1]
        # Promote `b1`,`b2` to numpy.arrays in case they are not
        b1 = np.array(b1)
        b2 = np.array(b2)
        # Use map() to create the equivalent of Table[] in Mathematica
        # .. define the iteration ranges
        iteration_range1 = np.arange(1, N1 + 1)
        iteration_range2 = np.arange(1, N2 + 1)
        # .. define function to iterate
        if coordinates == "lattice":

            def set_k_point(n1, n2):
                return [ki(n1, N1), ki(n2, N2), 0]

        elif coordinates == "cartesian":

            def set_k_point(n1, n2):
                return ki(n1, N1) * b1 + ki(n2, N2) * b2

        else:
            raise Exception("\ncoordinates must be 'lattice' or 'cartesian'")
        # .. combine map() and list() to create Table[]-like function
        output2D_aux = list(
            map(
                lambda n1: list(map(lambda n2: set_k_point(n1, n2), iteration_range2)),
                iteration_range1,
            )
        )
        # Reshape the output so that is a np.array with shape (N1*N2,3)
        output = np.array(output2D_aux).reshape(-1, 3)

    else:  # 3D CASE ..........................................................
        # Promote `N` to an np.array with the dimensions of the lattice.
        if np.isscalar(N):
            N = np.repeat(N, 3)
        # Get the number of kpoints in each direction
        N1 = N[0]
        N2 = N[1]
        N3 = N[2]
        # Promote `b1`,`b2`,`b3` to numpy.arrays in case they are not
        b1 = np.array(b1)
        b2 = np.array(b2)
        b2 = np.array(b3)
        # Use map() to create the equivalent of Table[] in Mathematica
        # .. define the iteration ranges
        iteration_range1 = np.arange(1, N1 + 1)
        iteration_range2 = np.arange(1, N2 + 1)
        iteration_range3 = np.arange(1, N3 + 1)
        # .. define function to iterate
        if coordinates == "lattice":

            def set_k_point(n1, n2, n3):
                return [ki(n1, N1), ki(n2, N2), ki(n3, N3)]

        elif coordinates == "cartesian":

            def set_k_point(n1, n2, n3):
                return ki(n1, N1) * b1 + ki(n2, N2) * b2 + ki(n3, N3) * b3

        else:
            raise Exception("\ncoordinates must be 'lattice' or 'cartesian'")
        # .. combine map() and list() to create Table[]-like function
        output3D_aux = list(
            map(
                lambda n1: list(
                    map(
                        lambda n2: list(
                            map(lambda n3: set_k_point(n1, n2, n3), iteration_range3)
                        ),
                        iteration_range2,
                    )
                ),
                iteration_range1,
            )
        )
        # Reshape the output so that is a np.array with shape (N1*N2*N3,3)
        output = np.array(output3D_aux).reshape(-1, 3)

    # ...
    return output


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


# ****** FUNCTION DEFINITION: make_light_matter_Hamiltonian() ******


def make_light_matter_Hamiltonian(
    model, emitter_detunings=[0], couplings=[[[0.5], [0]]]
):
    # ...
    #
    # The idea is to create a sparce matrix (this is relevant to speed up calculations or even made
    # them attainable). To do that you need to provide the non-zero matrix elements H_{ij} and their
    # corresponding the indexes i,j.
    #
    # First, I create the bath Hamiltonian H_bath, given by the sum of H_onsite and H_hoppings. To do
    # that, I use the ouptuts from a finite pythonTB model. The emitter's and interaction's Hamiltonians
    # all together using the `emitter_detuning` and `couplings` inputs.
    #
    # ...

    ###  Definitions
    N_orbitals = np.shape(model._orb)[0]
    N_emitters = len(couplings)

    H_dimension = N_orbitals + N_emitters

    if np.size(emitter_detunings) == 1 and N_emitters > 1:
        emitter_detunings = emitter_detunings * N_emitters
    # ...
    # NOTE: if a single emitter_detunings is given, but more than one emitter is considered, we assume all  to
    #       emitters to feature the same detuning value.
    # ...

    ### Bath Hamiltonian ......................................................

    ## On-site energies ...

    # define the on-site Hamiltonian as a sparce matrix
    v_onsite = model._site_energies
    i_onsite = np.arange(0, N_orbitals, 1) + N_emitters
    j_onsite = np.arange(0, N_orbitals, 1) + N_emitters

    H_onsite = csc_matrix(
        (v_onsite, (i_onsite, j_onsite)), shape=(H_dimension, H_dimension)
    )

    ## Hopping terms ...

    # define the hopping Hamiltonian as a sparce matrix
    hopping_vals_aux1 = np.array(model._hoppings)[:, 0]  # get the hopping values
    hopping_indx_aux1 = np.array(model._hoppings)[:, [1, 2]]  # get the hopping indexes
    hopping_vals_aux2 = np.conjugate(
        hopping_vals_aux1
    )  # conjugate of the hopping values
    hopping_indx_aux2 = hopping_indx_aux1[:, [1, 0]]  # exchange the i and j indexes

    hopping_vals = np.hstack([hopping_vals_aux1, hopping_vals_aux2])
    hopping_indx = np.vstack([hopping_indx_aux1, hopping_indx_aux2]) + N_emitters

    # ...
    # NOTE_1: The aux quantities are defined to make the Hamiltonian Hermitian. The `model._hoppings`
    #         output accounts for the coupling between sites in one direction, but the conjugate term
    #         must be considered as well.
    # NOTE_2: The "+ N_emitters" in `hopping_indx` is there to shift the indexes so that H_bath occupies
    #         the lower right corner in the total Hamiltonian.
    # ...

    # define sparse hoppings matrix
    v_hopping = hopping_vals
    i_hopping = hopping_indx[:, 0]
    j_hopping = hopping_indx[:, 1]

    H_hoppings = csc_matrix(
        (v_hopping, (i_hopping, j_hopping)), shape=(H_dimension, H_dimension)
    )

    ## Sum both parts
    H_bath = H_onsite + H_hoppings

    ### Emitters + coupling Hamiltonian .......................................

    # define the emitters and interaction Hamiltonians as a sparce matrix

    # define the non-zero values and indexes for the emitters' Hamiltonian
    v_emitters = np.array(emitter_detunings)
    i_emitters = np.arange(0, N_emitters, 1)
    j_emitters = np.arange(0, N_emitters, 1)

    # define the non-zero values and indexes for the interactions' Hamiltonian
    coupling_vals_list = []
    coupling_indx_i_list = []
    coupling_indx_j_list = []
    for i in range(N_emitters):
        coupling_vals_list.append(couplings[i][0])
        coupling_indx_i_list.append(
            list(i * np.ones(np.shape(couplings[i][1]), dtype=int))
        )
        coupling_indx_j_list.append(list(np.array(couplings[i][1]) + N_emitters))

    coupling_vals_aux1 = np.hstack(coupling_vals_list)
    coupling_indx_i_aux1 = np.hstack(coupling_indx_i_list)
    coupling_indx_j_aux1 = np.hstack(coupling_indx_j_list)

    coupling_vals_aux2 = np.conjugate(coupling_vals_aux1)
    coupling_indx_i_aux2 = coupling_indx_j_aux1
    coupling_indx_j_aux2 = coupling_indx_i_aux1

    v_coupling = np.hstack([coupling_vals_aux1, coupling_vals_aux2])
    i_coupling = np.hstack([coupling_indx_i_aux1, coupling_indx_i_aux2])
    j_coupling = np.hstack([coupling_indx_j_aux1, coupling_indx_j_aux2])

    # ...
    # NOTE: I use similar procedure to the one use to create the H_hoppings matrix. This depends on
    #       the particular way the coupling object is defined though. Therefore, if we change that,
    #       we will need to revisit this part.
    # ...

    # merge the emitters' and interactions' values and indexes
    v_all = np.hstack((v_emitters, v_coupling))
    i_all = np.hstack((i_emitters, i_coupling))
    j_all = np.hstack((j_emitters, j_coupling))

    H_emitters_and_coupling = csc_matrix(
        (v_all, (i_all, j_all)), shape=(H_dimension, H_dimension)
    )

    ### Total Hamiltonian
    H_total = H_bath + H_emitters_and_coupling

    ### Output ...
    return H_total


# ****** FUNCTION DEFINITION: set_local_coupling() ******


def set_local_coupling(model, emitter_positions=[[0, 0]], g_values=[0.3]):
    ### Some definitions...

    # get lattice vectors
    lattice_vectors = model._lat

    # get orbitals
    orbitals = model._orb

    # get number of emitters
    N_emitters = np.shape(emitter_positions)[0]

    # ...
    # NOTE: if a single g_value is given, but more than one emitter is considered, we assume all emitters to
    #       feature the same coupling constant.
    # ...

    if np.size(g_values) == 1 and N_emitters > 1:
        g_values = g_values * N_emitters

    ### Find positions (i.e. orbitals in Cartesian coordinates)
    positions = find_positions(orbitals, lattice_vectors)

    ### Find indexes where emitters are placed
    emitter_positions_indexes = np.zeros(N_emitters, dtype=int)
    for i in range(N_emitters):
        emitter_positions_indexes[i] = np.where(
            (positions == emitter_positions[i]).all(1)
        )[0][0]

    ### Define the coupling values, given by g_values
    couplings = []
    for i in range(N_emitters):
        couplings.append([])
        couplings[i].append([g_values[i]])
        couplings[i].append([emitter_positions_indexes[i]])

    ### Output ...
    return couplings


# ****** AUXILIARY FUNCTIONS ******
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


def binary_entropy(p):
    """
    Compute the binary entropy function for probabilities p.

    The binary entropy is defined as:
        H(p) = -p * log(p) - (1 - p) * log(1 - p)

    Parameters
    ----------
    p : array_like
        Input probabilities. Each value should lie in [0, 1].

    Returns
    -------
    ndarray
        The binary entropy evaluated element-wise on `p`.

    Notes
    -----
    - Uses natural logarithm (base e). For base-2 entropy,
      replace `np.log` with `np.log2`.
    - Values of `p` are clipped to [1e-12, 1 - 1e-12] to avoid
      evaluating log(0).
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)  # avoid log(0) errors
    return -p * np.log(p) - (1 - p) * np.log(1 - p)
