from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def beckmann_user_equilibrium_minimiser(
    step: float,
    density: NDArray[np.float64],
    density_y: NDArray[np.float64],
    time: NDArray[np.float64],
    params: NDArray[np.float64],
    critical_density: NDArray[np.float64]
) -> float:
    alpha = params[:, 0]
    beta = params[:, 1]
    eps = params[0, 2]

    x = density + step * (density_y - density)

    travel_time = np.zeros_like(x, dtype=float)
    for ii in range(travel_time.size):
        ratio = x[ii] / critical_density[ii]

        r = ratio if ratio >= eps else eps

        travel_time[ii] = time[ii] * x[ii] * (1.0 + (alpha[ii] / (beta[ii] + 1.0)) * (r ** beta[ii]))

    return np.sum(travel_time)
