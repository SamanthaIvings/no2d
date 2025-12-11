# no2d_code/frank_wolfe/ta_ue.py
from __future__ import annotations

import numpy as np

from no2d_code.frank_wolfe import FrankWolfe_UE_Flex
from no2d_code.frank_wolfe.IO_operations import (
    load_edges, load_nodes, load_od_list, load_demand,
    init_ue_logs, save_ue_results,
)
from no2d_code.frank_wolfe.frank_wolfe_classes import FWResult, FWRunConfig
from no2d_code.frank_wolfe.shortestpathtree import Digraph


def TA_UE(
    tol: float = 58.6,
    parentDir: str = "../../data/",
) -> None:
    edges = load_edges(parentDir)
    _nodes = load_nodes(parentDir)

    u0 = edges["u"].to_numpy(dtype=int)
    v0 = edges["v"].to_numpy(dtype=int)

    length_m = edges["length"].to_numpy(dtype=float)
    speedlimit_kmh = edges["speedlim"].to_numpy(dtype=float)
    capacity = edges["capacity"].to_numpy(dtype=float)
    criticalDensity = edges["criticalDensity"].to_numpy(dtype=float)

    u = u0 + 1
    v = v0 + 1

    n_nodes = int(max(u.max(), v.max()) + 1)

    bpr_params = np.tile([0.15, 4.0, 0.0], (edges.shape[0], 1))

    G = Digraph.from_edge_arrays(
        u=u,
        v=v,
        length_m=length_m,
        speedlimit_kmh=speedlimit_kmh,
        capacity=capacity,
        critical_density=criticalDensity,
        n_nodes=n_nodes,
        weight=length_m,
        bpr_params=bpr_params
    )

    TimeBinPeriods = ["DAY"]

    steplimit = 125000

    txtName, critLogName, critBestsName = init_ue_logs(parentDir, steplimit)

    OD_list = load_od_list(parentDir, tol)
    demand = load_demand(parentDir)

    inds = np.where(OD_list[:, 0] == OD_list[:, 1])[0]
    OD_list = np.delete(OD_list, inds, axis=0)
    demand = np.delete(demand, inds, axis=0)

    OD_list[:, 2] += 1
    OD_list[:, 3] += 1

    run_cfg = FWRunConfig(
        eps=1e-5,
        stepbreak=steplimit,
        txt_name=txtName,
        crit_log_name=critLogName,
        crit_bests_name=critBestsName,
    )

    UEflows = []
    UEflowsBest = []
    last_result: FWResult | None = None

    for i in range(len(TimeBinPeriods)):
        print(f"Time bin ...{i+1}")
        print("Starting user-equilibrium Frank-Wolfe...")

        result = FrankWolfe_UE_Flex(
            demand=demand,
            graph=G,
            OD_list=OD_list,
            config=run_cfg,
        )

        UEflows.append(result.flows)
        UEflowsBest.append(result.flows_best)
        last_result = result

    UEflows = np.column_stack(UEflows) if UEflows else np.empty((G.u.size, 0), dtype=float)
    UEflowsBest = np.column_stack(UEflowsBest) if UEflowsBest else np.empty((G.u.size, 0), dtype=float)

    if last_result is None:
        raise RuntimeError("No Frank–Wolfe iterations were run")

    save_ue_results(
        parent_dir=parentDir,
        UEflows=UEflows,
        UEflowsBest=UEflowsBest,
        result=last_result,
    )


if __name__ == "__main__":
    TA_UE()
