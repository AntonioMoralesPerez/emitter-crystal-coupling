import numpy as np
from typing import Union


def binary_entropy(p: Union[float, np.ndarray], base: float = np.e) -> np.ndarray:
    """
    Compute the binary entropy function H(p) for probabilities p.

    The binary entropy is defined as:
        H(p) = -p * log(p) - (1 - p) * log(1 - p)

    Parameters
    ----------
    p : float or array_like
        Input probability or array of probabilities. Values should lie in [0, 1].
    base : float, optional
        Logarithm base. Default is natural logarithm (np.e). For base-2 entropy, use `base=2`.

    Returns
    -------
    entropy : ndarray
        Binary entropy evaluated element-wise on `p`.

    Notes
    -----
    - Values of `p` are clipped to [1e-12, 1 - 1e-12] to avoid log(0) errors.
    - The result is in "nats" if `base=np.e`, or in "bits" if `base=2`.

    Examples
    --------
    >>> binary_entropy(0.5)
    0.6931471805599453  # natural log
    >>> binary_entropy(0.5, base=2)
    1.0
    >>> binary_entropy(np.array([0.1, 0.5, 0.9]))
    array([0.32508297, 0.69314718, 0.32508297])
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)  # avoid log(0)
    return (-p * np.log(p) - (1 - p) * np.log(1 - p)) / np.log(base)
