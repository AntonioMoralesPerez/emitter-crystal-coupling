import numpy as np

# Define constants
E = 1.602176634e-19  # Elementary charge [C]
HBAR = 1.054571817e-34  # Reduced Planck constant [J·s]
C = 299792458  # Speed of light [m/s]
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity [F/m]

K_CONST: float = E / np.sqrt(C * HBAR * EPSILON_0)  # Constant prefactor, dimensionless

LORENTIAN_BROADENING: float = 1e-3  # Broadening factor for LDOS calculations


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
    E_k_r0 : ndarray, shape (3, Nk, Nb)
        Electric field at the emitter position for each mode. The first dimension
        indexes vector components (x, y, z), second dimension is k-points,
        third is bands.
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

    Notes
    -----
    * g_k is computed per (k, band), then summed over k with the Fourier factor.
    * The function assumes uniform discretization of k-points; if non-uniform
      weights are needed, modify the prefactor accordingly.
    """

    # --- Helper: pad arrays to 3D ---
    def pad_to_3d(array):
        array = np.atleast_2d(array)
        if array.shape[1] < 3:
            pad = np.zeros((array.shape[0], 3 - array.shape[1]))
            array = np.hstack([array, pad])
        return array

    kpts = pad_to_3d(kpts)
    rpts = pad_to_3d(rpts)

    Nk = kpts.shape[0]

    # --- Input sanity checks ---
    assert (
        E_k_r0.shape[0] == 3
    ), "First dimension of E_k_r0 must be 3 (vector components)."
    assert dipole_moment.shape == (3,), "Dipole moment must be a 3-component vector."
    assert (
        omega_k.shape[0] == Nk
    ), "omega_k first dimension must match number of k-points."

    # --- Dipole · Electric field: shape (Nk, Nb) ---
    dipole_times_field = np.tensordot(dipole_moment, E_k_r0, axes=([0], [0]))

    # --- Prefactor: shape (Nk, Nb) ---
    prefactor = -1j * const * np.sqrt(omega_k / (2 * Nk))

    # --- Compute g_k(r0) ---
    gk = prefactor * dipole_times_field  # shape (Nk, Nb)

    # --- Fourier factor: exp(-i k·r) shape (Nk, Nr) ---
    exponential_factor = np.exp(-1j * np.dot(kpts, rpts.T))  # shape (Nk, Nr)

    # --- Fourier transform over k-points: result (Nb, Nr) ---
    gr0r = np.dot(gk.T, exponential_factor)

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
    omega_k : ndarray, shape (Nk, Nb)
        Frequencies of the modes. Nk = number of k-points, Nb = number of bands.
    E_k_r0 : ndarray, shape (Nk, Nb, 3)
        Electric field at the emitter position for each mode. Last dimension
        indexes vector components (x, y, z).
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
    Nk, Nb = omega_k.shape
    assert E_k_r0.shape == (Nk, Nb, 3), "E_k_r0 must have shape (Nk, Nb, 3)"
    assert p_vec.shape == (3,), "p_vec must be a 3-component vector"

    # --- Dipole-field overlap: |p · E|^2, shape (Nk, Nb) ---
    overlaps = (
        np.abs(np.tensordot(E_k_r0, p_vec, axes=([2], [0]))) ** 2
    )  # shape (Nk, Nb)

    # --- Prefactor from g_k definition ---
    prefactor = const**2 * omega_k / (2 * Nk)  # shape (Nk, Nb)

    # --- Lorentzian kernel ---
    Δ = omega_eval[:, None, None] - omega_k[None, :, :]  # shape (Nω, Nk, Nb)
    L = (1 / np.pi) * eta / (Δ**2 + eta**2)

    # --- LDOS: sum over k and bands ---
    ldos = np.sum(prefactor * overlaps * L, axis=(1, 2))

    # --- Return scalar if input was scalar ---
    return ldos[0] if np.isscalar(omega_eval) else ldos
