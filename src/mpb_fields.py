import numpy as np

from .constants import K_CONST, LORENTIAN_BROADENING


def compute_gr0r(
    E_k_r0: np.ndarray,
    omega_k: np.ndarray,
    kpts: np.ndarray,
    rpts: np.ndarray,
    const: float = K_CONST,
) -> np.ndarray:
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
