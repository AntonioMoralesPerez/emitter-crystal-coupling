import numpy as np

from .constants import K_CONST, LORENTZIAN_BROADENING


def pad_to_3d(array: np.ndarray) -> np.ndarray:
    """
    Pad a 2D array with zero columns until it has exactly 3 columns.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (N,) or (N, M) where M <= 3.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3), with zero-padding if M < 3.

    Raises
    ------
    ValueError
        If the input array has more than 3 columns.
    """
    array = np.atleast_2d(array)

    if array.shape[1] > 3:
        raise ValueError(f"Array has {array.shape[1]} columns, expected at most 3.")

    if array.shape[1] < 3:
        pad = np.zeros((array.shape[0], 3 - array.shape[1]))
        array = np.hstack([array, pad])

    return array


def compute_reciprocal_space_coupling(
    fr0_n_ks: np.ndarray,
    dipole_moment: np.ndarray,
    omega_ks_n: np.ndarray,
    const: float = K_CONST,
) -> np.ndarray:
    """
    Compute the reciprocal-space coupling g_{n,k}(r_0) between a quantum emitter
    at position r_0 and the photonic Bloch modes.

    Parameters
    ----------
    fr0_n_ks : np.ndarray, shape (Nb, Nk, 3)
        Electric field profile at the emitter position for each mode.
        First dimension indexes bands, second indexes k-points, third is vector
        components.
    dipole_moment : np.ndarray, shape (3,)
        Dipole moment vector normalized by the dipole unit.
    omega_ks_n : np.ndarray, shape (Nk, Nb)
        Mode frequencies in units of c/a.
    const : float, optional
        Prefactor constant in the definition of g_{n,k}(r_0).
        Default is `K_CONST`.

    Returns
    -------
    gr0_n_ks : np.ndarray, shape (Nb, Nk)
        Reciprocal-space coupling g_{n,k}(r_0) for each band and k-point.

    Raises
    ------
    ValueError
        If `fr0_n_ks` does not have 3 vector components in its last dimension.
        If `dipole_moment` is not a 3-component vector.
        If `omega_ks_n` does not have shape (Nk, Nb).

    Notes
    -----
    The mode-resolved coupling is defined as:

        g_{n,k}(r_0) = -i * const * sqrt(ω_{n,k} / (2 N_k)) * (p·f_{n,k}(r_0))

    where N_k is the number of k-points, p is the dipole moment, and f_{n,k}(r_0)
    is the electric field profile at the emitter position.
    """
    n_bands, n_kpoints, n_components = fr0_n_ks.shape

    # Input validation
    if n_components != 3:
        raise ValueError(
            f"Last dimension of `fr0_n_ks` must be 3 (vector components), got {n_components}."
        )
    if dipole_moment.shape != (3,):
        raise ValueError(
            f"Dipole moment must be a 3-component vector, got shape {dipole_moment.shape}."
        )
    if omega_ks_n.shape != (n_kpoints, n_bands):
        raise ValueError(
            f"`omega_ks_n` must have shape (Nk, Nb)=({n_kpoints}, {n_bands}), got {omega_ks_n.shape}."
        )

    # Compute dipole · field, shape: (Nb, Nk)
    dipole_times_field = np.einsum("i,bki->bk", dipole_moment, fr0_n_ks)

    # Compute prefactor, shape: (Nb, Nk)
    prefactor = (-1j * const * np.sqrt(omega_ks_n / (2 * n_kpoints))).T

    # Reciprocal-space coupling g_{n,k}(r_0), shape: (Nb, Nk)
    return prefactor * dipole_times_field


def compute_real_space_coupling(
    fr0_n_ks: np.ndarray,
    dipole_moment: np.ndarray,
    omega_ks_n: np.ndarray,
    ks: np.ndarray,
    rs: np.ndarray,
    const: float = K_CONST,
) -> np.ndarray:
    """
    Compute the real-space coupling g_{n,r}(r_0) from reciprocal-space couplings
    g_{n,k}(r_0) via a discrete Fourier transform over k-points.

    Parameters
    ----------
    fr0_n_ks : np.ndarray, shape (Nb, Nk, 3)
        Electric field profile at the emitter position for each mode.
        First dimension indexes bands, second indexes k-points, third is vector
        components.
    dipole_moment : np.ndarray, shape (3,)
        Dipole moment vector normalized by the dipole unit.
    omega_ks_n : np.ndarray, shape (Nk, Nb)
        Mode frequencies in units of c/a.
    ks : np.ndarray, shape (Nk, 3)
        k-point coordinates. Shorter vectors are padded with zeros.
    rs : np.ndarray, shape (Nr, 3)
        Real-space points where g_r(r0) is evaluated. Shorter vectors are padded
        with zeros.
    const : float, optional
        Prefactor constant in the definition of g_{n,k}(r_0).
        Default is `K_CONST`.

    Returns
    -------
    gr0_n_rs : np.ndarray, shape (Nb, Nr)
        Real-space coupling g_{n,r}(r_0) for each band and r-point.

    Raises
    ------
    ValueError
        If `fr0_n_ks` does not have 3 vector components in its last dimension.
        If `dipole_moment` is not a 3-component vector.
        If `omega_ks_n` does not have shape (Nk, Nb).
        If the number of k-points in `ks` does not match `Nk`.

    Notes
    -----
    The real-space coupling is computed as:

        g_{n,r}(r_0) = 1/sqrt(N_k) Σ_k g_{n,k}(r_0) * exp(-i k·r)

    where g_{n,k}(r_0) is given by `compute_reciprocal_space_coupling`.
    """
    ks = pad_to_3d(ks)  # (Nk, 3)
    rs = pad_to_3d(rs)  # (Nr, 3)

    n_kpoints = fr0_n_ks.shape[1]

    if ks.shape[0] != n_kpoints:
        raise ValueError(
            f"Number of k-points in `ks` ({ks.shape[0]}) does not match Nk={n_kpoints}."
        )

    # Reciprocal-space coupling g_{n,k}(r_0), shape: (Nb, Nk)
    gr0_n_ks = compute_reciprocal_space_coupling(
        fr0_n_ks, dipole_moment, omega_ks_n, const=const
    )

    # Fourier factor: 1/sqrt(Nk) * exp(-i k·r), shape: (Nk, Nr)
    exponential_factor = np.exp(-1j * np.dot(ks, rs.T)) / np.sqrt(n_kpoints)

    # Sum over k-points via matrix multiplication, shape: (Nb, Nr)
    return np.dot(gr0_n_ks, exponential_factor)


def compute_decay_rate(
    omegas: np.ndarray,
    omega_ks: np.ndarray,
    fr0_ks: np.ndarray,
    dipole_moment: np.ndarray,
    eta: float = LORENTZIAN_BROADENING,
    const: float = K_CONST,
) -> np.ndarray:
    """
    Compute the frequency-resolved spontaneous-emission decay rate Γ(ω) at a
    given emitter position via Fermi's golden rule, using Lorentzian broadening
    for the Dirac delta.

    Parameters
    ----------
    omegas : np.ndarray
        Frequencies at which to evaluate Γ in units of c/a.
        Can be a scalar or 1D array.
    omega_ks : np.ndarray, shape (Nk,)
        Frequencies of the modes in units of c/a.
    fr0_ks : np.ndarray, shape (Nk, 3)
        Electric field profiles at the emitter position.
        Last dimension indexes vector components (x, y, z).
    dipole_moment : np.ndarray, shape (3,)
        Dipole moment vector normalized by the dipole unit.
    eta : float, optional
        Lorentzian broadening parameter. Default is `LORENTZIAN_BROADENING`.
    const : float, optional
        Prefactor constant in the definition of g_k(r_0).
        Default is `K_CONST`.

    Returns
    -------
    gamma : float or np.ndarray
        Decay rate evaluated at the frequencies `omegas`, in units of c/a.
        Returns a scalar if `omegas` is a scalar, otherwise an array of shape (N_omega,).

    Raises
    ------
    ValueError
        If `fr0_ks` does not have shape (Nk, 3).
        If `dipole_moment` is not a 3-component vector.

    Notes
    -----
    Fermi's golden rule gives:

        Γ(ω) = 2π Σ_k |g_k(r_0)|^2 δ(ω − ω_k)

    where the Dirac delta is replaced by a Lorentzian kernel of width η.
    The computation uses the relation:

        |g_k(r_0)|^2 = const^2 * (ω_k / (2 N_k)) * |p·f_k(r_0)|^2

    and replaces the Dirac delta by a normalized Lorentzian:

        δ(ω - ω_k) → (1/π) * η / ((ω - ω_k)^2 + η^2)
    """
    is_scalar = np.isscalar(omegas)
    omegas = np.atleast_1d(omegas)

    (n_kpoints,) = omega_ks.shape

    # Input validation
    if fr0_ks.shape != (n_kpoints, 3):
        raise ValueError(
            f"`fr0_ks` must have shape (Nk, 3)=({n_kpoints}, 3), got {fr0_ks.shape}."
        )

    # Promote to single-band shapes expected by the k-space coupling routine,
    # call it, then squeeze the band axis back out.
    gk_single_band = compute_reciprocal_space_coupling(
        fr0_n_ks=fr0_ks[np.newaxis, :, :],
        dipole_moment=dipole_moment,
        omega_ks_n=omega_ks[:, np.newaxis],
        const=const,
    )[0]  # shape: (Nk,)

    # |g_k|^2, shape: (Nk,)
    g_squared = np.abs(gk_single_band) ** 2

    # Lorentzian kernel via broadcasting, shape: (N_omega, Nk)
    delta = omegas[:, np.newaxis] - omega_ks[np.newaxis, :]
    lorentzian = (1 / np.pi) * eta / (delta**2 + eta**2)

    # Γ(ω) = 2π Σ_k |g_k|^2 δ_η(ω - ω_k), shape: (N_omega,)
    gamma = 2 * np.pi * np.sum(g_squared * lorentzian, axis=1)

    return gamma[0] if is_scalar else gamma
