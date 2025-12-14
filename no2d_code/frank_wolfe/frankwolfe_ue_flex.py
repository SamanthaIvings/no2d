from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

from no2d_code.frank_wolfe.beckmann import beckmann_min_ue
from no2d_code.frank_wolfe.bpr import bpr_density_smooth
from no2d_code.frank_wolfe.frank_wolfe_classes import FWRunConfig, FWResult
from no2d_code.frank_wolfe.shortestpathtree import (
    Digraph,
    SPTWorkspace,
    spt_predecessors,
    aon_flow_from_tree,
    write_xagi_from_tree,
)


@dataclass(frozen=True)
class ODGrouped:
    origins: np.ndarray
    start: np.ndarray
    dests: np.ndarray
    demands: np.ndarray
    cols: np.ndarray


def _group_od_by_origin(origin_destination: np.ndarray, demand: np.ndarray) -> ODGrouped:
    s = origin_destination[:, 2].astype(np.int64, copy=False)
    t = origin_destination[:, 3].astype(np.int64, copy=False)
    q = demand.astype(np.float64, copy=False)
    cols = np.arange(s.size, dtype=np.int64)

    order = np.argsort(s, kind="mergesort")
    s = s[order]
    t = t[order]
    q = q[order]
    cols = cols[order]

    uniq, start = np.unique(s, return_index=True)
    start = np.append(start, s.size).astype(np.int64, copy=False)

    return ODGrouped(origins=uniq, start=start, dests=t, demands=q, cols=cols)


def _aon_flow_all(
    graph: Digraph,
    od: ODGrouped,
    edge_cost: np.ndarray,
    ws: SPTWorkspace,
) -> np.ndarray:
    flow_y = np.zeros(edge_cost.size, dtype=np.float64)

    for k in range(od.origins.size):
        origin = int(od.origins[k])
        a = int(od.start[k])
        b = int(od.start[k + 1])

        ws.node_demand.fill(0.0)
        np.add.at(ws.node_demand, od.dests[a:b], od.demands[a:b])

        dist, pred_node, pred_edge = spt_predecessors(graph, origin, edge_cost=edge_cost, ws=ws)
        aon_flow_from_tree(origin, dist, pred_node, pred_edge, ws.node_demand, ws.edge_flow)
        flow_y += ws.edge_flow

    return flow_y


def _build_xagi(
    graph: Digraph,
    od: ODGrouped,
    edge_cost: np.ndarray,
    ws: SPTWorkspace,
) -> np.ndarray:
    Xa_Gi = np.zeros((edge_cost.size, od.cols.size), dtype=np.float64)

    for k in range(od.origins.size):
        origin = int(od.origins[k])
        a = int(od.start[k])
        b = int(od.start[k + 1])

        _, pred_node, pred_edge = spt_predecessors(graph, origin, edge_cost=edge_cost, ws=ws)
        write_xagi_from_tree(origin, pred_node, pred_edge, od.dests[a:b], od.cols[a:b], Xa_Gi)

    return Xa_Gi


def FrankWolfe_UE_Flex(
    demand: np.ndarray,
    graph: Digraph,
    origin_destination: np.ndarray,
    config: FWRunConfig,
) -> FWResult:
    time = graph.free_flow_travel_h
    capacity = graph.capacity
    criticalDensity = graph.critical_density
    params = graph.bpr_params
    if params is None:
        raise ValueError("Digraph.bpr_params must be set before calling FrankWolfe_UE_Flex.")

    eps = config.eps
    stepbreak = config.stepbreak
    txtName = config.txt_name
    critLogName = config.crit_log_name
    critBestsName = config.crit_bests_name

    od = _group_od_by_origin(origin_destination, demand)
    ws = SPTWorkspace.create(graph.n_nodes, graph.u.size)

    flow0 = np.zeros(graph.u.size, dtype=np.float64)
    density0 = flow0 * criticalDensity / capacity
    traveltime0 = bpr_density_smooth(time, density0, criticalDensity, params)

    flow = _aon_flow_all(graph, od, traveltime0, ws)

    LBD = 0.0
    L = 0

    crit1 = float("inf")
    crit2 = float("inf")

    crit1Best = crit1
    crit2Best = crit2
    UEflowsBest = flow.copy()
    iter_best = 0
    LBDBest = LBD

    density = flow * criticalDensity / capacity
    traveltime = bpr_density_smooth(time, density, criticalDensity, params)

    critLog = np.loadtxt(critLogName, delimiter=",")
    critBests = np.loadtxt(critBestsName, delimiter=",")

    Xa_Gi = np.zeros((flow.size, origin_destination.shape[0]), dtype=np.float64)

    while abs(crit1) > eps or abs(crit2) > eps:
        L += 1

        if L == stepbreak:
            Xa_Gi = _build_xagi(graph, od, traveltime, ws)
            UEflows = flow.copy()
            break

        flow_y = _aon_flow_all(graph, od, traveltime, ws)

        flow_p = flow_y - flow
        density = flow * criticalDensity / capacity
        density_y = flow_y * criticalDensity / capacity
        density_p = density_y - density

        gradT = bpr_density_smooth(time, density, criticalDensity, params)
        T = float(np.dot(flow, traveltime))
        T_Bar = T + float(np.dot(gradT, flow_p))

        LBD = max(LBD, T_Bar)
        crit1 = abs(T - LBD) / LBD

        if (crit1 <= crit1Best) and (crit2 <= crit2Best):
            crit1Best = crit1
            crit2Best = crit2
            UEflowsBest = flow.copy()
            iter_best = L
            LBDBest = LBD
            if critBests.ndim == 1:
                critBests = critBests.reshape(1, -1)
            critBests = np.vstack([critBests, np.array([crit1Best, crit2Best, iter_best], dtype=float)])
            np.savetxt(critBestsName, critBests, delimiter=",")

        if abs(crit1) < eps:
            Xa_Gi = _build_xagi(graph, od, traveltime, ws)
            UEflows = flow.copy()
            critLog[L, :] = np.array([crit1, crit2], dtype=float)
            np.savetxt(critLogName, critLog, delimiter=",")
            print(f"First convergence check met, iter={L}")
            break

        fun = lambda step: beckmann_min_ue(step, density, density_y, time, params, criticalDensity)
        res = minimize_scalar(fun, bounds=(0.0, 1.0), method="bounded")
        step_new = float(res.x)

        flow = flow + step_new * flow_p
        density = density + step_new * density_p

        traveltime = bpr_density_smooth(time, density, criticalDensity, params)
        T_new = float(np.dot(flow, traveltime))
        crit2 = abs(T_new - LBD) / LBD

        print(f"Iteration: {L} Step Size: {step_new}")
        print(f"Crit1: {crit1} Crit2: {crit2}")
        print(f"Flow difference norm: {float(np.linalg.norm(flow_p))}")
        print(f"Travel Time Update: {float(np.mean(traveltime))}")

        if (crit1 <= crit1Best) and (crit2 <= crit2Best):
            crit1Best = crit1
            crit2Best = crit2
            UEflowsBest = flow.copy()
            iter_best = L
            LBDBest = LBD
            if critBests.ndim == 1:
                critBests = critBests.reshape(1, -1)
            critBests = np.vstack(
                [critBests, np.array([crit1Best, crit2Best, iter_best], dtype=float)]
            )
            np.savetxt(critBestsName, critBests, delimiter=",")

        if abs(crit2) < eps:
            Xa_Gi = _build_xagi(graph, od, traveltime, ws)
            UEflows = flow.copy()
            critLog[L, :] = np.array([crit1, crit2], dtype=float)
            np.savetxt(critLogName, critLog, delimiter=",")
            print(f"Second convergence check met, iter={L}")
            break

        if (L % 100) == 0:
            with open(txtName, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()}: completed iteration {L}.\n")

        critLog[L, :] = np.array([crit1, crit2], dtype=float)
        np.savetxt(critLogName, critLog, delimiter=",")

    return FWResult(
        flows=UEflows,
        flows_best=UEflowsBest,
        crit1=float(crit1),
        crit2=float(crit2),
        crit1_best=float(crit1Best),
        crit2_best=float(crit2Best),
        iterations=int(L),
        iter_best=int(iter_best),
        Xa_Gi=Xa_Gi,
        crit_log=critLog,
        crit_bests=critBests,
        LBD=float(LBD),
        LBD_best=float(LBDBest),
    )
