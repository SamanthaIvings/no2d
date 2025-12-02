from __future__ import annotations

import numpy as np


def bpr_density_smooth(
    free_flow_time_h: np.ndarray,
    density: np.ndarray,
    critical_density: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    bpr_eps: float,
) -> np.ndarray:
    ratio = density / critical_density
    if bpr_eps > 0:
        ratio = np.maximum(ratio, bpr_eps)
    return free_flow_time_h * (1.0 + alpha * np.power(ratio, beta))


def beckmann_objective(
    step: float,
    density: np.ndarray,
    density_y: np.ndarray,
    free_flow_time_h: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    critical_density: np.ndarray,
    bpr_eps: float,
) -> float:
    x = density + step * (density_y - density)
    ratio = x / critical_density
    ratio_clipped = np.maximum(ratio, bpr_eps) if bpr_eps > 0 else ratio
    mask = ratio >= bpr_eps if bpr_eps > 0 else np.ones_like(ratio, dtype=bool)

    term = np.empty_like(x, dtype=float)
    term[mask] = 1.0 + (alpha[mask] / (beta[mask] + 1.0)) * np.power(ratio_clipped[mask], beta[mask])
    if bpr_eps > 0:
        term[~mask] = 1.0 + alpha[~mask] * (bpr_eps ** beta[~mask])
    else:
        term[~mask] = 1.0 + (alpha[~mask] / (beta[~mask] + 1.0)) * np.power(ratio_clipped[~mask], beta[~mask])

    return float(np.sum(free_flow_time_h * x * term))
