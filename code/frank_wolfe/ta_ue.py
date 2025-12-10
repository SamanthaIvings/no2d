from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

from code.frank_wolfe import FrankWolfe_UE_Flex
from code.frank_wolfe.shortestpathtree import Digraph


def TA_UE(
    tol: float = 58.6,
    parentDir: str = "../../data/",
) -> None:
    inputDir = os.path.join(parentDir, "inputs/")
    outputDir = os.path.join(parentDir, "outputs/")

    edges = pd.read_csv(os.path.join(inputDir, "edges.csv"))
    _nodes = pd.read_csv(os.path.join(inputDir, "nodes.csv"))

    highway = edges["highway"].astype(str).to_numpy()

    u0              = edges["u"].to_numpy(dtype=int)
    v0              = edges["v"].to_numpy(dtype=int)

    length_m        = edges["length"].to_numpy(dtype=float)
    speedlimit_kmh  = edges["speedlim"].to_numpy(dtype=float)
    capacity        = edges["capacity"].to_numpy(dtype=float)
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
        h = str(highway[ii])
        if ("motorway" in h) or ("trunk" in h):
            roadClass[ii] = "highway"
        else:
            if ("tertiary" in h) or ("unclassified" in h):
                roadClass[ii] = "rural"
            else:
                roadClass[ii] = "urban"

    alpha = np.full(G.u.size, 0.15, dtype=float)
    beta = np.full(G.u.size, 4.0, dtype=float)
    params = np.column_stack([alpha, beta, np.zeros(edges.shape[0], dtype=float)])

    TimeBinPeriods = ["DAY"]

    eps = 1e-5
    steplimit = 125000

    txtName = os.path.join(parentDir, "Out_TA_HPC_UE.txt")
    with open(txtName, "w", encoding="utf-8") as f:
        f.write(f"File created: {datetime.now().isoformat()}.\n")

    critLogName = os.path.join(parentDir, "All_crit1_crit2_UE.csv")
    np.savetxt(critLogName, np.full((steplimit + 1, 2), np.inf, dtype=float), delimiter=",")

    critBestsName = os.path.join(parentDir, "Best_crit1_crit2_UE.csv")
    np.savetxt(critBestsName, np.array([[np.inf, np.inf, np.inf]], dtype=float), delimiter=",")

    OD_list = np.loadtxt(os.path.join(inputDir, f"OD_list_tol{tol}.csv"), delimiter=",", skiprows=1)
    demand = np.loadtxt(os.path.join(inputDir, "demand.csv"), delimiter=",", skiprows=1)

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
            critBestsName=critBestsName
        )

        UEflows.append(UEflows_i)
        UEflowsBest.append(UEflowsBest_i)

    UEflows = np.column_stack(UEflows) if UEflows else np.empty((G.u.size, 0), dtype=float)
    UEflowsBest = np.column_stack(UEflowsBest) if UEflowsBest else np.empty((G.u.size, 0), dtype=float)

    os.makedirs(outputDir, exist_ok=True)

    np.savetxt(os.path.join(outputDir, "UE_flow.csv"), UEflows, delimiter=",")
    np.savetxt(os.path.join(outputDir, "UE_flow_best.csv"), UEflowsBest, delimiter=",")

    np.savetxt(os.path.join(outputDir, "UE_crit1and2.csv"), np.array([crit1_UE, crit2_UE], dtype=float), delimiter=",")
    np.savetxt(
        os.path.join(outputDir, "UE_crit1and2_best.csv"),
        np.array([crit1_UE_Best, crit2_UE_Best], dtype=float),
        delimiter=",",
    )

    np.savetxt(os.path.join(outputDir, "UE_L.csv"), np.array([L_UE], dtype=float), delimiter=",")
    np.savetxt(os.path.join(outputDir, "UE_L_best.csv"), np.array([iter_UE], dtype=float), delimiter=",")
    np.savetxt(os.path.join(outputDir, "UE_LBD.csv"), np.array([LBD_UE], dtype=float), delimiter=",")
    np.savetxt(os.path.join(outputDir, "UE_LBD_best.csv"), np.array([LBD_UE_Best], dtype=float), delimiter=",")


if __name__ == "__main__":
    TA_UE()
