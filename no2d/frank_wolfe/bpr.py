from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def bpr_density_smooth(
    time: ArrayLike,
    x: ArrayLike,
    criticalDensity: ArrayLike,
    params: ArrayLike,
) -> NDArray[np.float64]:
    time = np.asarray(time, dtype=float)
    x = np.asarray(x, dtype=float)
    criticalDensity = np.asarray(criticalDensity, dtype=float)
    params = np.asarray(params, dtype=float)

    if time.ndim != 1 or x.ndim != 1 or criticalDensity.ndim != 1:
        raise ValueError("time, x, criticalDensity must be 1D arrays")
    if params.ndim != 2 or params.shape[1] != 3:
        raise ValueError("params must be (n, 3): [alpha, beta, eps]")
    if not (time.size == x.size == criticalDensity.size == params.shape[0]):
        raise ValueError("time, x, criticalDensity, params must have same length")

    alpha = params[:, 0]
    beta = params[:, 1]

    eps_vals = np.unique(params[:, 2])
    if eps_vals.size != 1:
        raise ValueError(f"params[:,2] must contain a single eps value; got {eps_vals}")
    eps = float(eps_vals[0])

    ratio = x / criticalDensity
    ratio = np.maximum(ratio, eps)

    return time * (1.0 + alpha * np.power(ratio, beta))
