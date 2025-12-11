from __future__ import annotations

import numpy as np

from no2d_code.frank_wolfe import FrankWolfe_UE_Flex
from no2d_code.frank_wolfe.IO_operations import (
    load_edges, load_od_list, load_demand,
    init_ue_logs, save_ue_results,
)
from no2d_code.frank_wolfe.frank_wolfe_classes import FWResult, FWRunConfig
from no2d_code.frank_wolfe.shortestpathtree import Digraph


def find_transport_assignment_user_equilibrium(tol: float = 58.6, parentDir: str = "../../data/"):
    edges = load_edges(parentDir)
    graph = Digraph.from_edges(edges)

    TimeBinPeriods = ["DAY"]

    steplimit = 125000

    txtName, critLogName, critBestsName = init_ue_logs(parentDir, steplimit)

    OD_list = load_od_list(parentDir, tol)
    demand = load_demand(parentDir)

    inds = np.where(OD_list[:, 0] == OD_list[:, 1])[0]
    OD_list = np.delete(OD_list, inds, axis=0)
    demand = np.delete(demand, inds, axis=0)

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
            graph=graph,
            OD_list=OD_list,
            config=run_cfg,
        )

        UEflows.append(result.flows)
        UEflowsBest.append(result.flows_best)
        last_result = result

    UEflows = np.column_stack(UEflows) if UEflows else np.empty((graph.u.size, 0), dtype=float)
    UEflowsBest = np.column_stack(UEflowsBest) if UEflowsBest else np.empty((graph.u.size, 0), dtype=float)

    if last_result is None:
        raise RuntimeError("No Frank–Wolfe iterations were run")

    save_ue_results(
        parent_dir=parentDir,
        UEflows=UEflows,
        UEflowsBest=UEflowsBest,
        result=last_result,
    )


if __name__ == "__main__":
    find_transport_assignment_user_equilibrium()
