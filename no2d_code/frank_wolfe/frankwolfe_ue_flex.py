from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

from no2d_code.frank_wolfe.beckmann import beckmann_min_ue
from no2d_code.frank_wolfe.bpr import bpr_density_smooth
from no2d_code.frank_wolfe.frank_wolfe_classes import FWRunConfig, FWResult
from no2d_code.frank_wolfe.shortestpathtree import shortestpathtree_edges_cell, Digraph


def FrankWolfe_UE_Flex(
    demand: np.ndarray,
    graph: Digraph,
    OD_list: np.ndarray,
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

    flow0 = np.zeros(graph.u.size, dtype=float)
    density0 = flow0 * criticalDensity / capacity
    traveltime0 = bpr_density_smooth(graph.free_flow_travel_h, density0, criticalDensity, params)

    graph.weight = traveltime0

    flow = flow0.copy()

    for i in range(graph.n_nodes):
        E = shortestpathtree_edges_cell(graph, i)
        if i == 0:
            E_store = [None] * graph.n_nodes
        E_store[i] = E

    for i in range(OD_list.shape[0]):
        s = int(OD_list[i, 2])
        t = int(OD_list[i, 3])
        E1 = E_store[s]
        edgepath = E1[t]
        if edgepath:
            flow[np.asarray(edgepath, dtype=int)] += float(demand[i])

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
    graph.weight = traveltime

    critLog = np.loadtxt(critLogName, delimiter=",")
    critBests = np.loadtxt(critBestsName, delimiter=",")

    while abs(crit1) > eps or abs(crit2) > eps:
        L = L + 1

        if L == stepbreak:
            Xa_Gi = np.zeros((flow.size, OD_list.shape[0]), dtype=float)

            for i in range(graph.n_nodes):
                E = shortestpathtree_edges_cell(graph, i)
                if i == 0:
                    E_store = [None] * graph.n_nodes
                E_store[i] = E

            for i in range(OD_list.shape[0]):
                s = int(OD_list[i, 2])
                t = int(OD_list[i, 3])
                E1 = E_store[s]
                edgepath = E1[t]
                if edgepath:
                    Xa_Gi[np.asarray(edgepath, dtype=int), i] = 1.0

            UEflows = flow.copy()
            break

        flow_y = np.zeros(flow.size, dtype=float)
        Xa_Gi = np.zeros((flow.size, OD_list.shape[0]), dtype=float)

        for i in range(graph.n_nodes):
            E = shortestpathtree_edges_cell(graph, i)
            if i == 0:
                E_store = [None] * graph.n_nodes
            E_store[i] = E

        for i in range(OD_list.shape[0]):
            s = int(OD_list[i, 2])
            t = int(OD_list[i, 3])
            E1 = E_store[s]
            edgepath = E1[t]

            if edgepath:
                idxs = np.asarray(edgepath, dtype=int)
                flow_y[idxs] += float(demand[i])
                Xa_Gi[idxs, i] = 1.0

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
            critBests = np.vstack(
                [critBests, np.array([crit1Best, crit2Best, iter_best], dtype=float)]
            )
            np.savetxt(critBestsName, critBests, delimiter=",")

        if abs(crit1) < eps:
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
        graph.weight = traveltime

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
            Xa_Gi = np.zeros((flow.size, OD_list.shape[0]), dtype=float)

            for i in range(graph.n_nodes):
                E = shortestpathtree_edges_cell(graph, i)
                if i == 0:
                    E_store = [None] * graph.n_nodes
                E_store[i] = E

            for i in range(OD_list.shape[0]):
                s = int(OD_list[i, 2])
                t = int(OD_list[i, 3])
                E1 = E_store[s]
                edgepath = E1[t]
                if edgepath:
                    Xa_Gi[np.asarray(edgepath, dtype=int), i] = 1.0

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
