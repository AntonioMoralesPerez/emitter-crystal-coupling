import numpy as np

from .constants import K_CONST, LORENTIAN_BROADENING


# --- Helper: pad arrays to 3D ---
def pad_to_3d(array: np.ndarray) -> np.ndarray:
    array = np.atleast_2d(array)
    if array.shape[1] < 3:
        pad = np.zeros((array.shape[0], 3 - array.shape[1]))
        array = np.hstack([array, pad])
    return array


def compute_real_space_coupling(
    fr0_n_ks: np.ndarray,
    dipole_moment: np.ndarray,
    ω_ks_n: np.ndarray,
    ks: np.ndarray,
    rs: np.ndarray,
    const: float = K_CONST,
) -> np.ndarray:
    """
    Compute the real-space coupling g_{n,r}(r_0) from reciprocal-space couplings
    g_{n,k}(r_0) via a discrete Fourier transform over k-points.

    Parameters
    ----------
    fr0_n_ks : ndarray, shape (Nb, Nk, 3)
        Electric field's profile at the emitter position for each mode.
        First dimension indexes bands, second dimension k-points, third is vector
        components.
    dipole_moment : ndarray, shape (3,)
        Dipole moment vector normalized by the dipole unit.
    ω_ks_n : ndarray, shape (Nk, Nb)
        Mode frequencies in units of c/a.
    ks : ndarray, shape (Nk, 3)
        k-point coordinates. Shorter vectors are padded with zeros.
    rs : ndarray, shape (Nr, 3)
        Real-space points where g_r(r0) is evaluated. Shorter vectors are padded with zeros.
    const : float, optional
        Prefactor constant in the definition of g_{n,k}(r_0).
        Default is `K_CONST`.

    Returns
    -------
    gr0_n_rs : ndarray, shape (Nb, Nr)
        Real-space coupling g_{n,r}(r_0) for each band and r-point.

    Notes
    -----
    The mode-resolved coupling is defined as:

        g_{n,k}(r_0) = -i * const * sqrt(ω_{n,k} / (2 N_k)) * (p·f_{n,k}(r_0)),
    where N_k is the number of k-points, p is the dipole moment, and f_{n,k}(r_0)
    is the electric field's profile at the emitter position.

    The real-space coupling is computed by:

        g_{n,r}(r_0) = 1/sqrt(N_k) Σ_k g_{n,k}(r_0) * exp(-i k·r).
    """

    ks = pad_to_3d(ks)  # Nk x 3
    rs = pad_to_3d(rs)  # Nr x 3

    N_b, N_k, N_comp = fr0_n_ks.shape

    # --- Input sanity checks ---
    assert (
        N_comp == 3
    ), f"Last dimension of `fr0_n_ks` must be 3 (vector components), got {N_comp}."
    assert dipole_moment.shape == (3,), "Dipole moment must be a 3-component vector."
    assert ω_ks_n.shape == (
        N_k,
        N_b,
    ), f"`ω_ks_n` must have shape (N_k, N_b), got {ω_ks_n.shape}."
    assert (
        ks.shape[0] == N_k
    ), f"Number of k-points {ks.shape[0]} does not match N_k={N_k}."

    # --- Compute dipole · field: shape (Nb, Nk) ---
    # Vectorized dot product over last dimension
    dipole_times_field = np.einsum("i,bki->bk", dipole_moment, fr0_n_ks)

    # --- Prefactor: shape (Nk, Nb) ---
    prefactor = -1j * const * np.sqrt(ω_ks_n / (2 * N_k))
    # Transpose to match dipole_times_field: Nb x Nk
    prefactor = prefactor.T

    # --- Compute g_k(r0): Nb x Nk ---
    gr0_n_ks = prefactor * dipole_times_field

    # --- Fourier factor: exp(-i k·r) Nk x Nr ---
    exponential_factor = np.exp(-1j * np.dot(ks, rs.T))  # Nk x Nr

    # --- Sum over k-points: result Nb x Nr ---
    gr0_n_rs = np.dot(gr0_n_ks, exponential_factor)  # (Nb x Nk) · (Nk x Nr) -> Nb x Nr

    return gr0_n_rs


def compute_ldos(
    omega_eval: np.ndarray,
    omega_ks: np.ndarray,
    E_ks_r0: np.ndarray,
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
    omega_ks : ndarray, shape (Nk,)
        Frequencies of the modes. Nk = number of k-points.
    E_ks_r0 : ndarray, shape (Nk, 3)
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
    # --- Input validation
    omega_eval = np.atleast_1d(omega_eval)
    (Nk,) = omega_ks.shape
    (Nω,) = omega_eval.shape

    if not E_ks_r0.shape == (Nk, 3):
        raise ValueError("`E_ks_r0` must have shape (Nk, 3)")
    if not p_vec.shape == (3,):
        raise ValueError("`p_vec` must be a 3-component vector")

    # --- Dipole-field overlap: |p · E|^2, shape (Nk,)
    overlaps = np.abs(np.einsum("ki,i", E_ks_r0, p_vec)) ** 2

    # --- Prefactor from g_k definition
    prefactor = const**2 * omega_ks / (2 * Nk)  # shape (Nk,)

    # --- Lorentzian kernel
    Δ = np.zeros((Nω, Nk))
    for i, omega in enumerate(omega_eval):
        Δ[i] = omega - omega_ks

    L = (1 / np.pi) * eta / (Δ**2 + eta**2)  # shape (Nω, Nk)

    # --- LDOS: sum over k and bands
    ldos = np.sum(prefactor * overlaps * L, axis=1)

    # --- Return scalar if input was scalar
    return ldos[0] if np.isscalar(omega_eval) else ldos
