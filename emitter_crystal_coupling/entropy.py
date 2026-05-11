import numpy as np
import warnings


def binary_entropy(p: float | np.ndarray, base: float = np.e) -> np.ndarray:
    """
    Compute the binary entropy function H(p) for probabilities p.

    The binary entropy is defined as:

        H(p) = -p * log(p) - (1 - p) * log(1 - p)

    Parameters
    ----------
    p : float or np.ndarray
        Input probability or array of probabilities. Values should lie in [0, 1].
    base : float, optional
        Logarithm base. Default is the natural logarithm (np.e).
        For base-2 entropy use `base=2`.

    Returns
    -------
    np.ndarray
        Binary entropy evaluated element-wise on `p`.

    Notes
    -----
    Values of `p` are clipped to [1e-12, 1 - 1e-12] to avoid log(0) errors.
    The result is in nats if `base=np.e`, or in bits if `base=2`.

    Examples
    --------
    >>> binary_entropy(0.5)
    0.6931471805599453
    >>> binary_entropy(0.5, base=2)
    1.0
    >>> binary_entropy(np.array([0.1, 0.5, 0.9]))
    array([0.32508297, 0.69314718, 0.32508297])
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return (-p * np.log(p) - (1 - p) * np.log(1 - p)) / np.log(base)


def quadrant_entropy(
    x: np.ndarray,
    y: np.ndarray,
    amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the quadrant-resolved Shannon entropy of the photonic bath amplitudes.

    Partitions the bath sites into four quadrants of the (x, y) plane and
    computes the Shannon entropy of the probability distribution over them.
    Sites exactly on the axes (x=0 or y=0) are excluded from all quadrants.

    Parameters
    ----------
    x : np.ndarray, shape (Nr,)
        x-coordinates of the bath sites in lattice units.
    y : np.ndarray, shape (Nr,)
        y-coordinates of the bath sites in lattice units.
    amplitudes : np.ndarray, shape (Nt, Nr), complex
        Bath amplitudes C_r(t) at each site and time step.

    Returns
    -------
    S : np.ndarray, shape (Nt,)
        Shannon entropy of the quadrant probability distribution at each time step.
    p : np.ndarray, shape (Nt, 4)
        Normalized quadrant probabilities in order [++, -+, --, +-].

    Raises
    ------
    ValueError
        If the total probability is zero at any time step.

    Notes
    -----
    The quadrant probabilities are computed as:

        P_q(t) = Σ_{r ∈ q} |C_r(t)|²

    and normalized to sum to 1 at each time step. The entropy is then:

        S(t) = -Σ_q p_q(t) * log(p_q(t))

    For a uniform distribution over 4 quadrants, S = log(4).
    For a uniform distribution over 2 quadrants, S = log(2).
    """
    masks = np.stack(
        [
            (x > 0) & (y > 0),  # quadrant ++
            (x < 0) & (y > 0),  # quadrant -+
            (x < 0) & (y < 0),  # quadrant --
            (x > 0) & (y < 0),  # quadrant +-
        ]
    )  # shape: (4, Nr)

    weights = np.abs(amplitudes) ** 2  # shape: (Nt, Nr)
    P = weights @ masks.T  # shape: (Nt, 4)

    total = P.sum(axis=1, keepdims=True)

    zero_mask = (total == 0).ravel()
    if np.any(zero_mask):
        zero_steps = np.where(zero_mask)[0]
        warnings.warn(
            f"Total bath probability is zero at time steps {zero_steps.tolist()} "
            "(e.g. t=0 when all population is in the emitter). "
            "Entropy will be NaN at these steps.",
            RuntimeWarning,
            stacklevel=2,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        p = P / total
        S = -np.nansum(p * np.log(p), axis=1)

    return S, p
