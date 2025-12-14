from __future__ import annotations

import numpy as np

from no2d_code.frank_wolfe import FrankWolfe_UE_Flex
from no2d_code.frank_wolfe.IO_operations import load_edges, load_filtered_od_and_demand, init_ue_logs, save_ue_results
from no2d_code.frank_wolfe.frank_wolfe_classes import FWResult, FWRunConfig
from no2d_code.frank_wolfe.shortestpathtree import Digraph


def find_transport_assignment_user_equilibrium(tol: float = 58.6, parent_folder: str = "../../data/"):
    edges = load_edges(parent_folder)
    graph = Digraph.from_edges(edges)

    time_bin_period = ["DAY"]
    steplimit = 125000

    origin_destination, demand = load_filtered_od_and_demand(parent_folder, tol)

    txtName, critLogName, critBestsName = init_ue_logs(parent_folder, steplimit)

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

    for i in range(len(time_bin_period)):
        print(f"Time bin ...{i+1}")
        print("Starting user-equilibrium Frank-Wolfe...")

        result = FrankWolfe_UE_Flex(
            demand=demand,
            graph=graph,
            origin_destination=origin_destination,
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
        parent_dir=parent_folder,
        UEflows=UEflows,
        UEflowsBest=UEflowsBest,
        result=last_result,
    )


if __name__ == "__main__":
    find_transport_assignment_user_equilibrium()
