from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

from no2d_code.frank_wolfe.beckmann import beckmann_objective_ue, beckmann_line_search_objective_ue
from no2d_code.frank_wolfe.bpr import bpr_flow
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
    params = graph.bpr_params

    eps = config.eps

    flow0 = np.zeros(graph.u.size, dtype=float)
    traveltime0 = bpr_flow(graph.free_flow_travel_h, flow0, capacity, params)
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

    ratio = flow / capacity
    print("min ratio:", float(np.min(ratio)))
    print("count ratio < eps:", int(np.sum(ratio < params[0, 2])))

    traveltime = bpr_flow(time, flow, capacity, params)
    graph.weight = traveltime

    criteria_log = np.loadtxt(config.crit_log_name, delimiter=",")
    critBests = np.loadtxt(config.crit_bests_name, delimiter=",")

    best_gap = float("inf")
    prev_obj = beckmann_objective_ue(flow, time, params, capacity)

    step = 0
    while crit1 > eps:
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

        traveltime = bpr_flow(time, flow, capacity, params)

        fw_gap = float(np.dot(traveltime, -flow_p))
        if fw_gap < 0.0:
            fw_gap = 0.0

        tot_tt = float(np.dot(traveltime, flow))
        crit1 = fw_gap / max(tot_tt, 1e-12)

        print(f"FW gap: {fw_gap:.6e}  rel_gap: {crit1:.6e}  tot_tt: {tot_tt:.6e}")

        if crit1 < best_gap:
            best_gap = crit1
            ue_flows_best = flow.copy()
            iter_best = step
            LBDBest = LBD
            crit1Best = crit1
            crit2Best = crit2

            critBests = np.atleast_2d(critBests)
            critBests = np.vstack([critBests, np.array([crit1Best, crit2Best, iter_best], dtype=float)])
            np.savetxt(config.crit_bests_name, critBests, delimiter=",")

        if crit1 < eps:
            print(f"First convergence check met: crit1={crit1:.6e}, eps={eps:.6e}, iter={step}")
            ue_flows = flow.copy()
            criteria_log[step, :] = np.array([crit1, crit2], dtype=float)
            np.savetxt(config.crit_log_name, criteria_log, delimiter=",")
            break

        fun = lambda s: beckmann_line_search_objective_ue(s, flow, flow_y, time, params, capacity)

        f0 = float(fun(0.0))
        f1 = float(fun(1.0))
        print(f"line search: f(0)={f0:.6e} f(1)={f1:.6e}")

        res = minimize_scalar(fun, bounds=(0.0, 1.0), method="bounded")
        step_new = float(res.x)

        f_star = float(fun(step_new))
        print(f"line search: f(step*)={f_star:.6e}")

        ls_tol = 1e-10 * max(1.0, abs(f0))
        if f_star > f0 + ls_tol:
            raise RuntimeError(f"Line search did not decrease objective: f(step*)={f_star} > f(0)={f0}")

        flow = flow + step_new * flow_p

        ratio = flow / capacity
        print("min ratio:", float(np.min(ratio)))
        print("count ratio < eps:", int(np.sum(ratio < params[0, 2])))

        traveltime = bpr_flow(time, flow, capacity, params)
        graph.weight = traveltime

        move_norm = float(np.linalg.norm(step_new * flow_p))
        base_norm = max(float(np.linalg.norm(flow)), 1e-12)
        crit2 = move_norm / base_norm

        obj_new = beckmann_objective_ue(flow, time, params, capacity)
        obj_tol = 1e-10 * max(1.0, abs(prev_obj))
        if obj_new > prev_obj + obj_tol:
            raise RuntimeError(f"Objective increased: {obj_new} > {prev_obj}")
        prev_obj = obj_new

        print(f"Iteration: {step} Step Size: {step_new}")
        print(f"Crit1: {crit1} Crit2: {crit2}")
        print(f"Flow difference norm: {float(np.linalg.norm(flow_p))}")
        print(f"Travel Time Update: {float(np.mean(traveltime))}")

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
