import numpy as np

from constants import K_CONST, LORENTIAN_BROADENING


def compute_gr0r(
    E_k_r0: np.ndarray,
    dipole_moment: np.ndarray,
    omega_k: np.ndarray,
    kpts: np.ndarray,
    rpts: np.ndarray,
    const: float = K_CONST,
) -> np.ndarray:
    """
    Compute the real-space coupling g_r(r0) from reciprocal-space couplings g_k(r0)
    via a discrete Fourier transform over k-points.

    The mode-resolved coupling is defined as

        g_k(r0) = -i * const * sqrt(ω_k / (2 N)) * (p · E_k(r0)),

    where N is the number of k-points, p is the dipole moment, and E_k(r0)
    is the electric field at the emitter position. The real-space coupling is

        g(r) = Σ_k g_k(r0) * exp(-i k · r).

    Parameters
    ----------
    E_k_r0 : ndarray, shape (Nb, Nk, 3)
        Electric field at the emitter position for each mode. First dimension
        indexes bands, second dimension k-points, third is vector components.
    dipole_moment : ndarray, shape (3,)
        Dipole moment vector.
    omega_k : ndarray, shape (Nk, Nb)
        Mode frequencies.
    kpts : ndarray, shape (Nk, 3)
        k-point coordinates. Shorter vectors are padded with zeros.
    rpts : ndarray, shape (Nr, 3)
        Real-space points where g_r(r0) is evaluated. Shorter vectors are padded with zeros.
    const : float, optional
        Prefactor constant in the definition of g_k. Default is `K_CONST`.

    Returns
    -------
    gr0r : ndarray, shape (Nb, Nr)
        Real-space coupling g(r) for each band and r-point.
    """

    # --- Helper: pad arrays to 3D ---
    def pad_to_3d(array: np.ndarray) -> np.ndarray:
        array = np.atleast_2d(array)
        if array.shape[1] < 3:
            pad = np.zeros((array.shape[0], 3 - array.shape[1]))
            array = np.hstack([array, pad])
        return array

    kpts = pad_to_3d(kpts)  # Nk x 3
    rpts = pad_to_3d(rpts)  # Nr x 3

    Nb, Nk, comp = E_k_r0.shape

    # --- Input sanity checks ---
    assert (
        comp == 3
    ), f"Last dimension of E_k_r0 must be 3 (vector components), got {comp}."
    assert dipole_moment.shape == (3,), "Dipole moment must be a 3-component vector."
    assert omega_k.shape == (
        Nk,
        Nb,
    ), f"omega_k must have shape (Nk, Nb), got {omega_k.shape}."
    assert (
        kpts.shape[0] == Nk
    ), f"Number of k-points {kpts.shape[0]} does not match Nk={Nk}."

    # --- Compute dipole · field: shape (Nb, Nk) ---
    # Vectorized dot product over last dimension
    dipole_times_field = np.einsum("i,bki->bk", dipole_moment, E_k_r0)

    # --- Prefactor: shape (Nk, Nb) ---
    prefactor = -1j * const * np.sqrt(omega_k / (2 * Nk))
    # Transpose to match dipole_times_field: Nb x Nk
    prefactor = prefactor.T

    # --- Compute g_k(r0): Nb x Nk ---
    gk = prefactor * dipole_times_field

    # --- Fourier factor: exp(-i k·r) Nk x Nr ---
    exponential_factor = np.exp(-1j * np.dot(kpts, rpts.T))  # Nk x Nr

    # --- Sum over k-points: result Nb x Nr ---
    gr0r = np.dot(gk, exponential_factor)  # (Nb x Nk) · (Nk x Nr) -> Nb x Nr

    return gr0r


def compute_ldos(
    omega_eval: np.ndarray,
    omega_k: np.ndarray,
    E_k_r0: np.ndarray,
    p_vec: np.ndarray,
    eta: float = LORENTIAN_BROADENING,
    const: float = K_CONST,
) -> np.ndarray:
    """
    Compute the local density of states (LDOS) at a given emitter position
    using Lorentzian broadening.

    The LDOS is approximated as:

        LDOS(ω) = Σ_{k,n} |g_{k,n}(r0)|^2 δ(ω − ω_{k,n}),

    where the Dirac delta is replaced by a Lorentzian kernel of width eta.

    Parameters
    ----------
    omega_eval : array_like
        Frequencies at which to evaluate the LDOS. Can be a scalar or 1D array.
    omega_k : ndarray, shape (Nk,)
        Frequencies of the modes. Nk = number of k-points.
    E_k_r0 : ndarray, shape (Nk, 3)
        Electric field at the emitter position. Last dimension indexes vector
        components (x, y, z).
    p_vec : ndarray, shape (3,)
        Dipole moment vector.
    eta : float, optional
        Lorentzian broadening parameter η. Default is `LORENTIAN_BROADENING`.
    const : float, optional
        Prefactor constant in the definition of g_k. Default is `K_CONST`.

    Returns
    -------
    ldos : float or ndarray
        LDOS evaluated at the frequencies `omega_eval`. Returns a scalar if
        `omega_eval` is a scalar, otherwise an array of shape (Nω,).

    Notes
    -----
    The computation uses the relation:

        |g_k(r0)|^2 = const^2 * (ω_k / (2 Nk)) * |p · E_k(r0)|^2

    and replaces the Dirac delta by a normalized Lorentzian:

        δ(ω - ω_k) → (1/π) * η / ((ω - ω_k)^2 + η^2)
    """
    # --- Input validation ---
    omega_eval = np.atleast_1d(omega_eval)
    (Nk,) = omega_k.shape
    assert E_k_r0.shape == (Nk, 3), "E_k_r0 must have shape (Nk, 3)"
    assert p_vec.shape == (3,), "p_vec must be a 3-component vector"

    # --- Dipole-field overlap: |p · E|^2, shape (Nk,) ---
    overlaps = np.abs(np.einsum("ki,i", E_k_r0, p_vec)) ** 2

    # --- Prefactor from g_k definition ---
    prefactor = const**2 * omega_k / (2 * Nk)  # shape (Nk,)

    # --- Lorentzian kernel ---
    Δ = omega_eval[:, None] - omega_k[None, :]  # shape (Nω, Nk)
    L = (1 / np.pi) * eta / (Δ**2 + eta**2)

    # --- LDOS: sum over k and bands ---
    ldos = np.sum(prefactor * overlaps * L, axis=1)

    # --- Return scalar if input was scalar ---
    return ldos[0] if np.isscalar(omega_eval) else ldos
