from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bpr_density_smooth(
    time: NDArray[np.float64],
    x: NDArray[np.float64],
    critical_density: NDArray[np.float64],
    params: NDArray[np.float64],
) -> NDArray[np.float64]:
    alpha = params[:, 0]
    beta = params[:, 1]
    eps = params[0, 2]

    ratio = x / critical_density
    ratio = np.maximum(ratio, eps)

    return time * (1.0 + alpha * np.power(ratio, beta))
