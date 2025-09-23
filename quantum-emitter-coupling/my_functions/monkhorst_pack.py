import numpy as np


def monkhorst_pack_sampling(N, b1, b2, b3=None, coordinates="lattice"):
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
