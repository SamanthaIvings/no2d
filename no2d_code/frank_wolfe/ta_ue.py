from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from no2d_code.frank_wolfe.IO_operations import (
    init_ue_logs,
    load_edges,
    load_filtered_od_and_demand,
    save_ue_results,
)
from no2d_code.frank_wolfe.bpr import bpr_flow
from no2d_code.frank_wolfe.frank_wolfe_classes import FWResult, FWRunConfig
from no2d_code.frank_wolfe.frankwolfe_ue_flex import frank_wolfe_ue_solver
from no2d_code.frank_wolfe.shortestpathtree import Digraph
from no2d_code.visualisation.fw_flow_plotter import (
    compute_aon_flow,
    plot_fw_flow_comparison,
)


def _check_and_prepare_paths(parent_dir: Path) -> tuple[Path, Path]:
    nodes_csv = parent_dir / "inputs/nodes.csv"
    if not nodes_csv.exists():
        raise FileNotFoundError(f"nodes.csv not found at: {nodes_csv}")

    plots_dir = parent_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    return nodes_csv, plots_dir


def _load_problem_data(
    parent_directory: str,
    tol: float,
    eps: float,
) -> tuple[Digraph, np.ndarray, np.ndarray]:
    edges = load_edges(parent_directory)
    graph = Digraph.from_edges(edges, eps)

    origin_destination, demand = load_filtered_od_and_demand(parent_directory, tol)
    return graph, origin_destination, demand


def _build_run_config(
    parent_directory: str,
    eps: float,
    step_limit: int,
) -> FWRunConfig:
    txt_name, crit_log_name, crit_bests_name = init_ue_logs(parent_directory, step_limit)
    return FWRunConfig(
        eps=eps,
        steplimit=step_limit,
        txt_name=txt_name,
        crit_log_name=crit_log_name,
        crit_bests_name=crit_bests_name,
    )


def _run_time_bins(
    time_bin_periods: List[str],
    graph: Digraph,
    demand: np.ndarray,
    origin_destination: np.ndarray,
    run_cfg: FWRunConfig,
    nodes_csv: Path,
    plots_dir: Path,
):
    ue_flows: List[np.ndarray] = []
    ue_flows_best: List[np.ndarray] = []
    last_result: Optional[FWResult] = None

    for i, tag in enumerate(time_bin_periods):
        print(f"Time bin ...{i + 1}")
        print("Starting user-equilibrium Frank-Wolfe...")

        flow0 = np.zeros(graph.u.size, dtype=float)
        graph.weight = bpr_flow(
            graph.free_flow_travel_h,
            flow0,
            graph.capacity,
            graph.bpr_params,
        )

        aon_flow = compute_aon_flow(
            graph=graph,
            demand=demand,
            origin_destination=origin_destination,
        )

        result = frank_wolfe_ue_solver(
            demand=demand,
            graph=graph,
            origin_destination=origin_destination,
            config=run_cfg,
        )

        ue_flows.append(result.flows)
        ue_flows_best.append(result.flows_best)
        last_result = result

        out_png = plots_dir / f"fw_flow_compare_{tag}.png"
        plot_fw_flow_comparison(
            graph=graph,
            flow_init=aon_flow,
            flow_final=result.flows,
            nodes=nodes_csv,
            out_path=out_png,
        )
        print(f"Saved plot: {out_png}")

    if last_result is None:
        raise RuntimeError("No Frank–Wolfe iterations were run")

    n_edges = graph.u.size
    ue_flows_arr = (
        np.column_stack(ue_flows)
        if ue_flows
        else np.empty((n_edges, 0), dtype=float)
    )
    ue_flows_best_arr = (
        np.column_stack(ue_flows_best)
        if ue_flows_best
        else np.empty((n_edges, 0), dtype=float)
    )

    return ue_flows_arr, ue_flows_best_arr, last_result


def find_transport_assignment_user_equilibrium(tol: float = 58.6, parent_directory: str = "../../data/", eps = 1e-6):
    time_bin_periods = ["DAY"]
    step_limit = 125000

    parent_dir = Path(parent_directory)
    nodes_csv, plots_dir = _check_and_prepare_paths(parent_dir)

    graph, origin_destination, demand = _load_problem_data(
        parent_directory=parent_directory,
        tol=tol,
        eps=eps,
    )

    run_cfg = _build_run_config(
        parent_directory=parent_directory,
        eps=eps,
        step_limit=step_limit,
    )

    ue_flows, ue_flows_best, last_result = _run_time_bins(
        time_bin_periods=time_bin_periods,
        graph=graph,
        demand=demand,
        origin_destination=origin_destination,
        run_cfg=run_cfg,
        nodes_csv=nodes_csv,
        plots_dir=plots_dir,
    )

    save_ue_results(
        parent_dir=parent_directory,
        UEflows=ue_flows,
        UEflowsBest=ue_flows_best,
        result=last_result,
    )


if __name__ == "__main__":
    find_transport_assignment_user_equilibrium(eps=1e-6)
