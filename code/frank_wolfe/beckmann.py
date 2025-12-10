from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def beckmann_min_ue(
    L: float,
    density: ArrayLike,
    density_y: ArrayLike,
    time: ArrayLike,
    params: ArrayLike,
    criticalDensity: ArrayLike,
) -> float:
    density = np.asarray(density, dtype=float)
    density_y = np.asarray(density_y, dtype=float)
    time = np.asarray(time, dtype=float)
    params = np.asarray(params, dtype=float)
    criticalDensity = np.asarray(criticalDensity, dtype=float)

    if density.ndim != 1 or density_y.ndim != 1 or time.ndim != 1 or criticalDensity.ndim != 1:
        raise ValueError("density, density_y, time, criticalDensity must be 1D arrays")
    if params.ndim != 2 or params.shape[1] != 3:
        raise ValueError("params must be (n, 3): [alpha, beta, eps]")
    n = density.size
    if not (density_y.size == time.size == criticalDensity.size == params.shape[0] == n):
        raise ValueError("All inputs must have same length")

    alpha = params[:, 0]
    beta = params[:, 1]

    eps_vals = np.unique(params[:, 2])
    if eps_vals.size != 1:
        raise ValueError(f"params[:,2] must contain a single eps value; got {eps_vals}")
    eps = float(eps_vals[0])

    x = density + float(L) * (density_y - density)

    T1 = np.zeros_like(x, dtype=float)
    for ii in range(T1.size):
        if x[ii] / criticalDensity[ii] >= eps:
            T1[ii] = time[ii] * x[ii] * (
                1.0 + (alpha[ii] / (beta[ii] + 1.0)) * (x[ii] ** beta[ii]) * (criticalDensity[ii] ** beta[ii])
            )
        else:
            T1[ii] = time[ii] * x[ii] * (1.0 + alpha[ii] * (eps ** beta[ii]))

    return float(np.sum(T1))
