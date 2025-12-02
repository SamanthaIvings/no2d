from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from code.models import NetworkParameters, ODData
from code.frank_wolfe.cost_functions import bpr_density_smooth, beckmann_objective
from code.frank_wolfe.settings import FrankWolfeSettings, format_seconds, log, should_log_iteration
from code.frank_wolfe.shortest_path import all_or_nothing_assignment


def frank_wolfe_user_equilibrium(
    network: NetworkParameters,
    od_data: ODData,
    settings: FrankWolfeSettings,
) -> Tuple[np.ndarray, float, float, int, float]:
    start_total = time.perf_counter()

    flow = np.zeros(network.n_edges, dtype=float)
    travel_time = _travel_time_from_flow(network, flow)

    log("Initial all-or-nothing assignment...", enabled=settings.verbose_progress_log)
    flow = all_or_nothing_assignment(network, od_data, travel_time, settings)

    best_dual_bound = 0.0
    crit1 = np.inf
    crit2 = np.inf

    iteration = 0
    while _not_converged(crit1, crit2, settings):
        iteration += 1
        if iteration >= settings.step_limit:
            log(f"Max iterations reached: {iteration}", enabled=settings.verbose_progress_log)
            break

        step_start = time.perf_counter()
        flow, travel_time, best_dual_bound, crit1, crit2, line_search_fun, step = _fw_iteration(
            network=network,
            od_data=od_data,
            settings=settings,
            flow=flow,
            travel_time=travel_time,
            best_dual_bound=best_dual_bound,
        )

        if settings.verbose_progress_log and should_log_iteration(iteration, settings):
            _log_progress(
                iteration=iteration,
                crit1=crit1,
                crit2=crit2,
                step=step,
                best_dual_bound=best_dual_bound,
                line_search_fun=line_search_fun,
                flow=flow,
                travel_time=travel_time,
                step_seconds=time.perf_counter() - step_start,
                total_seconds=time.perf_counter() - start_total,
            )

        if _early_stop(crit1, crit2, settings, iteration):
            log(
                f"Converged (tol={settings.result_difference_tolerance}) at iter {iteration}",
                enabled=settings.verbose_progress_log,
            )
            break

    return flow, float(crit1), float(crit2), int(iteration), float(best_dual_bound)


def _fw_iteration(
    network: NetworkParameters,
    od_data: ODData,
    settings: FrankWolfeSettings,
    flow: np.ndarray,
    travel_time: np.ndarray,
    best_dual_bound: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    target_flow = all_or_nothing_assignment(network, od_data, travel_time, settings)
    search_direction = target_flow - flow

    density = _density_from_flow(network, flow)
    target_density = _density_from_flow(network, target_flow)

    grad_time = bpr_density_smooth(
        network.free_flow_time_h, density, network.critical_density, network.alpha, network.beta, network.bpr_eps
    )

    total_cost = float(flow @ travel_time)
    linearized_cost = total_cost + float(grad_time @ search_direction)

    best_dual_bound = max(best_dual_bound, linearized_cost)
    crit1 = _relative_gap(total_cost, best_dual_bound)

    line_search = minimize_scalar(
        lambda s: beckmann_objective(
            s,
            density,
            target_density,
            network.free_flow_time_h,
            network.alpha,
            network.beta,
            network.critical_density,
            network.bpr_eps,
        ),
        bounds=(0.0, 1.0),
        method="bounded",
    )
    step = float(line_search.x)

    flow = flow + step * search_direction
    travel_time = _travel_time_from_flow(network, flow)

    new_total_cost = float(flow @ travel_time)
    crit2 = _relative_gap(new_total_cost, best_dual_bound)

    return flow, travel_time, best_dual_bound, float(crit1), float(crit2), float(line_search.fun), float(step)


def _travel_time_from_flow(network: NetworkParameters, flow: np.ndarray) -> np.ndarray:
    density = _density_from_flow(network, flow)
    return bpr_density_smooth(
        network.free_flow_time_h, density, network.critical_density, network.alpha, network.beta, network.bpr_eps
    )


def _density_from_flow(network: NetworkParameters, flow: np.ndarray) -> np.ndarray:
    return flow * network.critical_density / network.capacity


def _relative_gap(value: float, bound: float) -> float:
    return abs(value - bound) / bound if bound > 0 else np.inf


def _not_converged(crit1: float, crit2: float, settings: FrankWolfeSettings) -> bool:
    tol = settings.result_difference_tolerance
    return (abs(crit1) > tol) or (abs(crit2) > tol)


def _early_stop(crit1: float, crit2: float, settings: FrankWolfeSettings, iteration: int) -> bool:
    tol = settings.result_difference_tolerance
    return (iteration > 100) and ((abs(crit1) < tol) or (abs(crit2) < tol))


def _log_progress(
    iteration: int,
    crit1: float,
    crit2: float,
    step: float,
    best_dual_bound: float,
    line_search_fun: float,
    flow: np.ndarray,
    travel_time: np.ndarray,
    step_seconds: float,
    total_seconds: float,
) -> None:
    total_cost = float(flow @ travel_time)
    avg_seconds = total_seconds / max(iteration, 1)
    log(
        "Iter {:6d} | step={:.6f} | crit1={:.3e} crit2={:.3e} | "
        "T={:.6e} Tnew={:.6e} LBD={:.6e} | LSfun={:.6e} | "
        "iter={} avg={} total={}".format(
            iteration,
            step,
            crit1,
            crit2,
            total_cost,
            total_cost,
            best_dual_bound,
            line_search_fun,
            format_seconds(step_seconds),
            format_seconds(avg_seconds),
            format_seconds(total_seconds),
        ),
        enabled=True,
    )
