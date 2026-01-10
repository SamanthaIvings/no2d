from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

from no2d_code.frank_wolfe.beckmann import beckmann_user_equilibrium_minimiser
from no2d_code.frank_wolfe.bpr import bpr_density_smooth
from no2d_code.frank_wolfe.frank_wolfe_classes import FWRunConfig, FWResult
from no2d_code.frank_wolfe.shortestpathtree import shortestpathtree_edges_cell, Digraph


def frank_wolfe_ue_solver(
    demand: np.ndarray,
    graph: Digraph,
    origin_destination: np.ndarray,
    config: FWRunConfig,
) -> FWResult:
    time = graph.free_flow_travel_h
    capacity = graph.capacity
    critical_density = graph.critical_density
    params = graph.bpr_params

    eps = config.eps

    flow0 = np.zeros(graph.u.size, dtype=float)
    density0 = flow0 * critical_density / capacity
    traveltime0 = bpr_density_smooth(graph.free_flow_travel_h, density0, critical_density, params)

    graph.weight = traveltime0

    flow = flow0.copy()

    for i in range(graph.n_nodes):
        E = shortestpathtree_edges_cell(graph, i)
        if i == 0:
            e_store = [None] * graph.n_nodes
        e_store[i] = E

    for i in range(origin_destination.shape[0]):
        s = int(origin_destination[i, 2])
        t = int(origin_destination[i, 3])
        E1 = e_store[s]
        edgepath = E1[t]
        if edgepath:
            flow[np.asarray(edgepath, dtype=int)] += float(demand[i])

    LBD = 0.0

    crit1 = float("inf")
    crit2 = float("inf")

    crit1Best = crit1
    crit2Best = crit2
    ue_flows_best = flow.copy()
    iter_best = 0
    LBDBest = LBD

    density = flow * critical_density / capacity

    traveltime = bpr_density_smooth(time, density, critical_density, params)
    graph.weight = traveltime

    criteria_log = np.loadtxt(config.crit_log_name, delimiter=",")
    critBests = np.loadtxt(config.crit_bests_name, delimiter=",")

    step = 0
    while abs(crit1) > eps or abs(crit2) > eps:
        step += 1

        if step == config.steplimit:
            ue_flows, xa_gi_metrics = build_edge_od_incidence_matrix(e_store, flow, graph, origin_destination)
            break

        flow_y = np.zeros(flow.size, dtype=float)
        e_store, xa_gi_metrics = precompute_shortest_path_trees(e_store, flow, graph, origin_destination)

        for i in range(origin_destination.shape[0]):
            s = int(origin_destination[i, 2])
            t = int(origin_destination[i, 3])
            E1 = e_store[s]
            edgepath = E1[t]

            if edgepath:
                indexes = np.asarray(edgepath, dtype=int)
                flow_y[indexes] += float(demand[i])
                xa_gi_metrics[indexes, i] = 1.0

        flow_p = flow_y - flow
        density = flow * critical_density / capacity
        density_y = flow_y * critical_density / capacity
        density_p = density_y - density

        gradT = bpr_density_smooth(time, density, critical_density, params)
        T = float(np.dot(flow, traveltime))
        T_Bar = T + float(np.dot(gradT, flow_p))

        LBD = max(LBD, T_Bar)
        crit1 = abs(T - LBD) / LBD

        crit1Best, crit2Best, ue_flows_best, iter_best, LBDBest, critBests = track_best_solution(
            crit1, crit2, flow, step, LBD, crit1Best, crit2Best,
            ue_flows_best, iter_best, LBDBest, critBests, config
        )

        if abs(crit1) < eps:
            ue_flows = flow.copy()
            criteria_log[step, :] = np.array([crit1, crit2], dtype=float)
            np.savetxt(config.crit_log_name, criteria_log, delimiter=",")
            print(f"First convergence check met, iter={step}")
            break

        fun = lambda step: beckmann_user_equilibrium_minimiser(step, density, density_y, time, params, critical_density)
        res = minimize_scalar(fun, bounds=(0.0, 1.0), method="bounded")
        step_new = float(res.x)

        flow = flow + step_new * flow_p
        density = density + step_new * density_p

        traveltime = bpr_density_smooth(time, density, critical_density, params)
        graph.weight = traveltime

        T_new = float(np.dot(flow, traveltime))
        crit2 = abs(T_new - LBD) / LBD

        print(f"Iteration: {step} Step Size: {step_new}")
        print(f"Crit1: {crit1} Crit2: {crit2}")
        print(f"Flow difference norm: {float(np.linalg.norm(flow_p))}")
        print(f"Travel Time Update: {float(np.mean(traveltime))}")

        crit1Best, crit2Best, ue_flows_best, iter_best, LBDBest, critBests = track_best_solution(
            crit1, crit2, flow, step, LBD, crit1Best, crit2Best,
            ue_flows_best, iter_best, LBDBest, critBests, config
        )

        if abs(crit2) < eps:
            ue_flows, xa_gi_metrics = build_edge_od_incidence_matrix(e_store, flow, graph, origin_destination)

            criteria_log[step, :] = np.array([crit1, crit2], dtype=float)
            np.savetxt(config.crit_log_name, criteria_log, delimiter=",")
            print(f"Second convergence check met, iter={step}")
            break

        if (step % 100) == 0:
            with open(config.txt_name, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()}: completed iteration {step}.\n")

        criteria_log[step, :] = np.array([crit1, crit2], dtype=float)
        np.savetxt(config.crit_log_name, criteria_log, delimiter=",")

    return FWResult(
        flows=ue_flows,
        flows_best=ue_flows_best,
        crit1=float(crit1),
        crit2=float(crit2),
        crit1_best=float(crit1Best),
        crit2_best=float(crit2Best),
        iterations=int(step),
        iter_best=int(iter_best),
        Xa_Gi=xa_gi_metrics,
        crit_log=criteria_log,
        crit_bests=critBests,
        LBD=float(LBD),
        LBD_best=float(LBDBest),
    )


def precompute_shortest_path_trees(e_store, flow, graph, origin_destination):
    xa_gi_metrics = np.zeros((flow.size, origin_destination.shape[0]), dtype=float)

    for i in range(graph.n_nodes):
        E = shortestpathtree_edges_cell(graph, i)
        if i == 0:
            e_store = [None] * graph.n_nodes
        e_store[i] = E
    return e_store, xa_gi_metrics


def build_edge_od_incidence_matrix(e_store, flow, graph, origin_destination):
    e_store, xa_gi_metrics = precompute_shortest_path_trees(e_store, flow, graph, origin_destination)

    for i in range(origin_destination.shape[0]):
        s = int(origin_destination[i, 2])
        t = int(origin_destination[i, 3])
        E1 = e_store[s]
        edgepath = E1[t]
        if edgepath:
            xa_gi_metrics[np.asarray(edgepath, dtype=int), i] = 1.0

    ue_flows = flow.copy()
    return ue_flows, xa_gi_metrics


def track_best_solution(
        crit1, crit2, flow, L, LBD, crit1_best, crit2_best,
        ue_flows_best, iter_best, LBD_best, crit_bests, config
):
    if (crit1 <= crit1_best) and (crit2 <= crit2_best):
        crit1_best = crit1
        crit2_best = crit2
        ue_flows_best = flow.copy()
        iter_best = L
        LBD_best = LBD

        crit_bests = np.atleast_2d(crit_bests)
        row = np.array([crit1_best, crit2_best, iter_best], dtype=float)
        crit_bests = np.vstack([crit_bests, row])
        np.savetxt(config.crit_bests_name, crit_bests, delimiter=",")

    return crit1_best, crit2_best, ue_flows_best, iter_best, LBD_best, crit_bests
