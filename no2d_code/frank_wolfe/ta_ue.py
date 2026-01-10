from __future__ import annotations

import numpy as np

from no2d_code.frank_wolfe import frank_wolfe_ue_solver
from no2d_code.frank_wolfe.IO_operations import load_edges, load_filtered_od_and_demand, init_ue_logs, save_ue_results
from no2d_code.frank_wolfe.frank_wolfe_classes import FWResult, FWRunConfig
from no2d_code.frank_wolfe.shortestpathtree import Digraph


def find_transport_assignment_user_equilibrium(tol: float = 58.6, parent_directory: str = "../../data/"):
    edges = load_edges(parent_directory)
    graph = Digraph.from_edges(edges)

    time_bin_periods = ["DAY"]
    step_limit = 125000

    origin_destination, demand = load_filtered_od_and_demand(parent_directory, tol)

    txt_name, crit_log_name, crit_bests_name = init_ue_logs(parent_directory, step_limit)

    run_cfg = FWRunConfig(
        eps=1e-5,
        steplimit=step_limit,
        txt_name=txt_name,
        crit_log_name=crit_log_name,
        crit_bests_name=crit_bests_name
    )

    ue_flows = []
    ue_flows_best = []
    last_result: FWResult | None = None

    for i in range(len(time_bin_periods)):
        print(f"Time bin ...{i+1}")
        print("Starting user-equilibrium Frank-Wolfe...")

        result = frank_wolfe_ue_solver(
            demand=demand,
            graph=graph,
            origin_destination=origin_destination,
            config=run_cfg,
        )

        ue_flows.append(result.flows)
        ue_flows_best.append(result.flows_best)
        last_result = result

    ue_flows = np.column_stack(ue_flows) if ue_flows else np.empty((graph.u.size, 0), dtype=float)
    ue_flows_best = np.column_stack(ue_flows_best) if ue_flows_best else np.empty((graph.u.size, 0), dtype=float)

    if last_result is None:
        raise RuntimeError("No Frank–Wolfe iterations were run")

    save_ue_results(
        parent_dir=parent_directory,
        UEflows=ue_flows,
        UEflowsBest=ue_flows_best,
        result=last_result,
    )


if __name__ == "__main__":
    find_transport_assignment_user_equilibrium()
