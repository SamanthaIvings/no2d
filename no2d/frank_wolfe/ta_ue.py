from __future__ import annotations

import numpy as np

from no2d.frank_wolfe.IO_operations import (
    load_edges, load_nodes, load_od_list, load_demand,
    init_ue_logs, save_ue_results
)
from no2d.frank_wolfe.frankwolfe_ue_flex import FrankWolfe_UE_Flex
from no2d.frank_wolfe.shortestpathtree import Digraph


def TA_UE(
    tol: float = 58.6,
    parentDir: str = "../../data/",
) -> None:
    edges = load_edges(parentDir)
    _nodes = load_nodes(parentDir)

    highway = edges["highway"].astype(str).to_numpy()

    u0 = edges["u"].to_numpy(dtype=int)
    v0 = edges["v"].to_numpy(dtype=int)

    length_m = edges["length"].to_numpy(dtype=float)
    speedlimit_kmh = edges["speedlim"].to_numpy(dtype=float)
    capacity = edges["capacity"].to_numpy(dtype=float)
    criticalDensity = edges["criticalDensity"].to_numpy(dtype=float)

    u = u0 + 1
    v = v0 + 1

    n_nodes = int(max(u.max(), v.max()) + 1)

    G = Digraph.from_edge_arrays(
        u=u,
        v=v,
        length_m=length_m,
        speedlimit_kmh=speedlimit_kmh,
        capacity=capacity,
        critical_density=criticalDensity,
        n_nodes=n_nodes,
        weight=length_m,
    )

    roadClass = np.empty(edges.shape[0], dtype=object)
    for ii in range(edges.shape[0]):
        h = highway[ii]
        if ("motorway" in h) or ("trunk" in h):
            roadClass[ii] = "highway"
        elif ("tertiary" in h) or ("unclassified" in h):
            roadClass[ii] = "rural"
        else:
            roadClass[ii] = "urban"

    alpha = np.full(G.u.size, 0.15, dtype=float)
    beta = np.full(G.u.size, 4.0, dtype=float)
    params = np.column_stack([alpha, beta, np.zeros(edges.shape[0], dtype=float)])

    TimeBinPeriods = ["DAY"]

    eps = 1e-5
    steplimit = 125000

    txtName, critLogName, critBestsName = init_ue_logs(parentDir, steplimit)

    OD_list = load_od_list(parentDir, tol)
    demand = load_demand(parentDir)

    inds = np.where(OD_list[:, 0] == OD_list[:, 1])[0]
    OD_list = np.delete(OD_list, inds, axis=0)
    demand = np.delete(demand, inds, axis=0)

    OD_list[:, 2] += 1
    OD_list[:, 3] += 1

    assert OD_list[:, 2].min() >= 0 and OD_list[:, 3].min() >= 0
    assert OD_list[:, 2].max() < n_nodes and OD_list[:, 3].max() < n_nodes

    UEflows = []
    UEflowsBest = []

    for i in range(len(TimeBinPeriods)):
        print(f"Time bin ...{i+1}")
        print("Starting user-equilibrium Frank-Wolfe...")

        (
            UEflows_i,
            crit1_UE,
            crit2_UE,
            L_UE,
            _,
            _critLog,
            _critBests,
            UEflowsBest_i,
            crit1_UE_Best,
            crit2_UE_Best,
            iter_UE,
            LBD_UE,
            LBD_UE_Best,
        ) = FrankWolfe_UE_Flex(
            demand=demand,
            time=G.free_flow_travel_h,
            topo_graph=G,
            OD_list=OD_list,
            eps=eps,
            capacity=G.capacity,
            criticalDensity=G.critical_density,
            params=params,
            stepbreak=steplimit,
            txtName=txtName,
            critLogName=critLogName,
            critBestsName=critBestsName,
        )

        UEflows.append(UEflows_i)
        UEflowsBest.append(UEflowsBest_i)

    UEflows = np.column_stack(UEflows) if UEflows else np.empty((G.u.size, 0), dtype=float)
    UEflowsBest = np.column_stack(UEflowsBest) if UEflowsBest else np.empty((G.u.size, 0), dtype=float)

    save_ue_results(
        parent_dir=parentDir,
        UEflows=UEflows,
        UEflowsBest=UEflowsBest,
        crit1_UE=crit1_UE,
        crit2_UE=crit2_UE,
        crit1_UE_Best=crit1_UE_Best,
        crit2_UE_Best=crit2_UE_Best,
        L_UE=L_UE,
        iter_UE=iter_UE,
        LBD_UE=LBD_UE,
        LBD_UE_Best=LBD_UE_Best,
    )


if __name__ == "__main__":
    TA_UE()
