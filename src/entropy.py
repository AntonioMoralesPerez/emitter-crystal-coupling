import numpy as np


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
