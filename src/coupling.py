import numpy as np


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
    exponential_factor = np.exp(-1j * (kpts @ rpts.T))

    # Output
    gr0r = prefactor * np.transpose(dipole_times_field) @ exponential_factor

    return gr0r
