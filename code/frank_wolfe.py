from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from models import NetworkParameters, ODData


@dataclass(frozen=True)
class FrankWolfeSettings:
    result_difference_tolerance: float = 1e-5
    step_limit: int = 200
    verbose_progress_log: bool = True
    log_print_frequency: int = 10
    aon_verbose: bool = False
    origins_progress_every: int = 25


def _fmt_sec(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds/60:.2f}m"
    return f"{seconds/3600:.2f}h"


def _log(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


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


def _min_edge_per_pair(
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], int]]:
    edge_id = np.arange(len(weights), dtype=int)
    key = from_nodes.astype(np.int64) * np.int64(n_nodes) + to_nodes.astype(np.int64)

    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    u_s = from_nodes[order]
    v_s = to_nodes[order]
    w_s = weights[order]
    id_s = edge_id[order]

    u_out: List[int] = []
    v_out: List[int] = []
    w_out: List[float] = []
    id_out: List[int] = []

    i = 0
    m = len(key_s)
    while i < m:
        j = i + 1
        best_j = i
        best_w = w_s[i]
        while j < m and key_s[j] == key_s[i]:
            if w_s[j] < best_w:
                best_w = w_s[j]
                best_j = j
            j += 1
        u_out.append(int(u_s[best_j]))
        v_out.append(int(v_s[best_j]))
        w_out.append(float(best_w))
        id_out.append(int(id_s[best_j]))
        i = j

    pair_to_edge = {(u, v): eid for u, v, eid in zip(u_out, v_out, id_out)}
    return np.asarray(u_out, int), np.asarray(v_out, int), np.asarray(w_out, float), pair_to_edge


def _build_adjacency(
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
) -> Tuple[csr_matrix, Dict[Tuple[int, int], int]]:
    u, v, w, pair_to_edge = _min_edge_per_pair(from_nodes, to_nodes, weights, n_nodes)
    return csr_matrix((w, (u, v)), shape=(n_nodes, n_nodes)), pair_to_edge


def _reconstruct_edge_path(
    predecessors: np.ndarray,
    origin: int,
    destination: int,
    pair_to_edge: Dict[Tuple[int, int], int],
) -> List[int]:
    if destination == origin or predecessors[destination] < 0:
        return []

    nodes_rev = [destination]
    cur = destination
    while cur != origin:
        cur = int(predecessors[cur])
        if cur < 0:
            return []
        nodes_rev.append(cur)

    nodes = nodes_rev[::-1]
    return [pair_to_edge[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]


def all_or_nothing_assignment(
    network: NetworkParameters,
    od_data: ODData,
    link_travel_time_h: np.ndarray,
    settings: FrankWolfeSettings,
) -> np.ndarray:
    start = time.perf_counter()
    adjacency, pair_to_edge = _build_adjacency(
        network.from_nodes, network.to_nodes, link_travel_time_h, network.n_nodes
    )

    aon_flow = np.zeros(network.n_edges, dtype=float)
    unique_origins = np.unique(od_data.origins)

    for i, origin in enumerate(unique_origins, start=1):
        _, predecessors = dijkstra(adjacency, directed=True, indices=int(origin), return_predecessors=True)

        trips_from_origin = np.where(od_data.origins == origin)[0]
        for k in trips_from_origin:
            edge_path = _reconstruct_edge_path(
                predecessors, int(origin), int(od_data.destinations[k]), pair_to_edge
            )
            if edge_path:
                aon_flow[edge_path] += od_data.demand[k]

        if settings.aon_verbose and (
            i == 1 or i % settings.origins_progress_every == 0 or i == len(unique_origins)
        ):
            _log(f"  AoN: origin {i}/{len(unique_origins)} ({_fmt_sec(time.perf_counter() - start)} elapsed)", enabled=True)

    if settings.aon_verbose:
        _log(f"  AoN: finished in {_fmt_sec(time.perf_counter() - start)}", enabled=True)

    return aon_flow


def need_to_print_log(iteration: int, settings: FrankWolfeSettings) -> bool:
    return iteration == 1 or iteration % settings.log_print_frequency == 0


def frank_wolfe_user_equilibrium(
    network: NetworkParameters,
    od_data: ODData,
    settings: FrankWolfeSettings,
) -> Tuple[np.ndarray, float, float, int, float]:
    start_total = time.perf_counter()

    flow = np.zeros(network.n_edges, dtype=float)
    density = flow * network.critical_density / network.capacity
    travel_time = bpr_density_smooth(
        network.free_flow_time_h, density, network.critical_density, network.alpha, network.beta, network.bpr_eps
    )

    _log("Initial all-or-nothing assignment...", enabled=settings.verbose_progress_log)
    flow = all_or_nothing_assignment(network, od_data, travel_time, settings)

    best_dual_bound = 0.0
    crit1 = np.inf
    crit2 = np.inf

    density = flow * network.critical_density / network.capacity
    travel_time = bpr_density_smooth(
        network.free_flow_time_h, density, network.critical_density, network.alpha, network.beta, network.bpr_eps
    )

    iteration = 0
    while (abs(crit1) > settings.result_difference_tolerance) or (abs(crit2) > settings.result_difference_tolerance):
        iteration += 1
        if iteration >= settings.step_limit:
            _log(f"Max iterations reached: {iteration}", enabled=settings.verbose_progress_log)
            break

        start_iter = time.perf_counter()

        target_flow = all_or_nothing_assignment(network, od_data, travel_time, settings)
        search_direction = target_flow - flow

        density = flow * network.critical_density / network.capacity
        target_density = target_flow * network.critical_density / network.capacity

        grad_time = bpr_density_smooth(
            network.free_flow_time_h, density, network.critical_density, network.alpha, network.beta, network.bpr_eps
        )

        total_cost = float(flow @ travel_time)
        linearized_cost = total_cost + float(grad_time @ search_direction)

        best_dual_bound = max(best_dual_bound, linearized_cost)
        crit1 = abs(total_cost - best_dual_bound) / best_dual_bound if best_dual_bound > 0 else np.inf

        line_search = minimize_scalar(
            lambda s: beckmann_objective(
                s, density, target_density,
                network.free_flow_time_h, network.alpha, network.beta, network.critical_density, network.bpr_eps
            ),
            bounds=(0.0, 1.0),
            method="bounded",
        )
        step = float(line_search.x)

        flow = flow + step * search_direction
        density = density + step * (target_density - density)

        travel_time = bpr_density_smooth(
            network.free_flow_time_h, density, network.critical_density, network.alpha, network.beta, network.bpr_eps
        )

        new_total_cost = float(flow @ travel_time)
        crit2 = abs(new_total_cost - best_dual_bound) / best_dual_bound if best_dual_bound > 0 else np.inf

        iter_time = time.perf_counter() - start_iter
        total_time = time.perf_counter() - start_total
        avg_time = total_time / max(iteration, 1)

        if settings.verbose_progress_log and need_to_print_log(iteration, settings):
            _log(
                "Iter {:6d} | step={:.6f} | crit1={:.3e} crit2={:.3e} | "
                "T={:.6e} Tnew={:.6e} LBD={:.6e} | LSfun={:.6e} | "
                "iter={} avg={} total={}".format(
                    iteration, step, crit1, crit2,
                    total_cost, new_total_cost, best_dual_bound,
                    float(line_search.fun),
                    _fmt_sec(iter_time), _fmt_sec(avg_time), _fmt_sec(total_time),
                ),
                enabled=True,
            )

        if (abs(crit1) < settings.result_difference_tolerance or abs(crit2) < settings.result_difference_tolerance) and iteration > 100:
            _log(f"Converged (tol={settings.result_difference_tolerance}) at iter {iteration}", enabled=settings.verbose_progress_log)
            break

    return flow, float(crit1), float(crit2), int(iteration), float(best_dual_bound)
