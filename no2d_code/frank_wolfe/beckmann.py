import numpy as np
from numpy.typing import NDArray


def beckmann_objective_ue(
    flow: NDArray[np.float64],
    time: NDArray[np.float64],
    params: NDArray[np.float64],
    capacity: NDArray[np.float64],
) -> float:
    alpha = params[:, 0]
    beta = params[:, 1]

    x = flow
    term = x + (alpha / (beta + 1.0)) * (np.power(x, beta + 1.0) / np.power(capacity, beta))
    return float(np.sum(time * term))


def beckmann_line_search_objective_ue(
    step: float,
    flow: NDArray[np.float64],
    flow_y: NDArray[np.float64],
    time: NDArray[np.float64],
    params: NDArray[np.float64],
    capacity: NDArray[np.float64],
) -> float:
    x = flow + step * (flow_y - flow)
    return beckmann_objective_ue(x, time, params, capacity)
