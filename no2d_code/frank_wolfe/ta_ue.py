from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

import numpy as np

from no2d_code.frank_wolfe import filepath_configs as fc
from no2d_code.frank_wolfe.IO_operations import (
    init_ue_logs,
    load_edges,
    load_filtered_od_and_demand,
    save_ue_results,
    has_ue_cache_pickle,
    load_ue_cache_pickle,
    save_ue_cache_pickle,
)
from no2d_code.frank_wolfe.bpr import bpr_flow
from no2d_code.frank_wolfe.frank_wolfe_classes import FWResult, FWRunConfig
from no2d_code.frank_wolfe.frankwolfe_ue_flex import frank_wolfe_ue_solver
from no2d_code.frank_wolfe.shortestpathtree import Digraph, shortestpathtree_edges_cell
from no2d_code.frank_wolfe.octt_mapping import octt_from_traveltime
from no2d_code.visualisation.fw_flow_plotter import (
    plot_fw_flow_comparison,
    plot_edge_value_comparison,
)


# ------------------------------------------------------------------------------
# Paths / IO
# ------------------------------------------------------------------------------

def _check_and_prepare_paths(parent_dir: Path) -> tuple[Path, Path, Path]:
    nodes_csv = Path(fc.input_path(str(parent_dir), fc.NODES_CSV))
    if not nodes_csv.exists():
        raise FileNotFoundError(f"nodes.csv not found at: {nodes_csv}")

    plots_dir = parent_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = Path(fc.outputs_dir(str(parent_dir)))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return nodes_csv, plots_dir, outputs_dir


def _load_input_data(
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


# ------------------------------------------------------------------------------
# AoN / SPT utilities (unique origins)
# ------------------------------------------------------------------------------

def _compute_all_or_nothing_flow_unique_origins(
    *,
    demand: np.ndarray,
    origin_destination: np.ndarray,
    graph: Digraph,
) -> np.ndarray:
    origins = np.unique(origin_destination[:, 2].astype(int))
    print(f"[AoN] Unique origins: {origins.size}")

    e_store: Dict[int, List[List[int]]] = {}
    for j, s in enumerate(origins):
        if j % 200 == 0 or j == origins.size - 1:
            print(f"[AoN] Building SPT {j+1}/{origins.size}")
        e_store[int(s)] = shortestpathtree_edges_cell(graph, int(s))

    flow = np.zeros(graph.u.size, dtype=float)

    hit = 0
    for i in range(origin_destination.shape[0]):
        s = int(origin_destination[i, 2])
        t = int(origin_destination[i, 3])
        store = e_store.get(s)
        if store is None:
            continue
        path = store[t]
        if path:
            hit += 1
            d = float(demand[i])
            for e in path:
                flow[int(e)] += d

    print(f"[AoN] Done. Paths found for {hit}/{origin_destination.shape[0]} ODs.")
    return flow


def _compute_od_costs_unique_origins(
    *,
    origin_destination: np.ndarray,
    edge_cost: np.ndarray,
    graph: Digraph,
) -> np.ndarray:
    origins = np.unique(origin_destination[:, 2].astype(int))

    e_store: Dict[int, List[List[int]]] = {}
    for s in origins:
        e_store[int(s)] = shortestpathtree_edges_cell(graph, int(s))

    od_cost = np.full(origin_destination.shape[0], np.inf, dtype=float)
    for i in range(origin_destination.shape[0]):
        s = int(origin_destination[i, 2])
        t = int(origin_destination[i, 3])
        path = e_store[s][t]
        if path:
            od_cost[i] = float(np.sum(edge_cost[np.asarray(path, dtype=int)]))

    return od_cost


def _run_time_bins(
    *,
    time_bin_periods: List[str],
    graph: Digraph,
    demand: np.ndarray,
    origin_destination: np.ndarray,
    run_cfg: FWRunConfig,
    nodes_csv: Path,
    plots_dir: Path,
    outputs_dir: Path,
    compute_octt: bool,
    use_cache: bool,
    overwrite_cache: bool,
    debug_octt: bool,
    parent_directory: str,
    tol: float,
):
    ue_flows = []
    ue_flows_best = []
    last_result: Optional[FWResult] = None

    for tag in time_bin_periods:
        print(f"Time bin ... ({tag})")

        flow0 = np.zeros(graph.u.size, dtype=float)
        graph.weight = bpr_flow(
            graph.free_flow_travel_h,
            flow0,
            graph.capacity,
            graph.bpr_params,
        )

        print("[AoN] Computing initial all-or-nothing flow...")
        aon_flow = _compute_all_or_nothing_flow_unique_origins(
            demand=demand,
            origin_destination=origin_destination,
            graph=graph,
        )

        if use_cache and not overwrite_cache and has_ue_cache_pickle(parent_directory, tag):
            UEflows, UEflowsBest, result, meta = load_ue_cache_pickle(parent_directory, tag)
            print(f"[FW] Loaded cache: {meta}")
        else:
            result = frank_wolfe_ue_solver(
                demand=demand,
                graph=graph,
                origin_destination=origin_destination,
                config=run_cfg,
            )
            UEflows = result.flows
            UEflowsBest = result.flows_best

            if use_cache:
                save_ue_cache_pickle(
                    parent_dir=parent_directory,
                    tag=tag,
                    UEflows_col=UEflows,
                    UEflowsBest_col=UEflowsBest,
                    result=result,
                    meta={
                        "tol": tol,
                        "eps": run_cfg.eps,
                        "steplimit": run_cfg.steplimit,
                    },
                )

        ue_flows.append(np.asarray(UEflows))
        ue_flows_best.append(np.asarray(UEflowsBest))
        last_result = result

        # ------------------------------------------------------------------
        # Plot sanity check
        # ------------------------------------------------------------------
        nodes_arr = np.genfromtxt(nodes_csv, delimiter=",", names=True)
        node_col = "node" if "node" in nodes_arr.dtype.names else nodes_arr.dtype.names[0]
        node_ids = np.asarray([r[node_col] for r in nodes_arr], dtype=int)

        print("[PLOT] nodes_csv:", nodes_csv)
        print("[PLOT] node_ids min/max/count:",
              int(node_ids.min()), int(node_ids.max()), node_ids.size)
        print("[PLOT] graph expects node ids 0..", graph.n_nodes - 1)

        if node_ids.min() != 0 or node_ids.max() != graph.n_nodes - 1:
            raise ValueError(
                "nodes.csv does not match Digraph node ids. "
                "You are plotting with the wrong nodes file."
            )

        out_png = plots_dir / f"fw_flow_compare_{tag}.png"
        plot_fw_flow_comparison(
            graph=graph,
            flow_init=aon_flow,
            flow_final=np.asarray(UEflows),
            nodes=nodes_csv,
            out_path=out_png,
        )

        if compute_octt:
            tt_aon = bpr_flow(graph.free_flow_travel_h, aon_flow, graph.capacity, graph.bpr_params)
            tt_ue = bpr_flow(graph.free_flow_travel_h, np.asarray(UEflows), graph.capacity, graph.bpr_params)

            octt_aon = octt_from_traveltime(tt_aon, debug=debug_octt)
            octt_ue = octt_from_traveltime(tt_ue, debug=debug_octt)

            out_octt_png = plots_dir / f"fw_octt_compare_{tag}.png"
            plot_edge_value_comparison(
                graph=graph,
                value_init=octt_aon,
                value_final=octt_ue,
                nodes=nodes_csv,
                out_path=out_octt_png,
                title_init="AoN OCTT",
                title_final="UE OCTT",
                cbar_label="OCTT",
                delta_label="ΔOCTT",
            )

            graph.weight = octt_ue
            od_octt = _compute_od_costs_unique_origins(
                origin_destination=origin_destination,
                edge_cost=octt_ue,
                graph=graph,
            )

            np.save(outputs_dir / f"octt_edge_{tag}.npy", octt_ue)
            np.save(outputs_dir / f"od_octt_{tag}.npy", od_octt)

    return (
        np.column_stack(ue_flows),
        np.column_stack(ue_flows_best),
        last_result,
    )


def find_transport_assignment_user_equilibrium(
    *,
    tol: float = 100.0,
    parent_directory: str = "../../data",
    eps: float = 1e-6,
    compute_octt: bool = True,
    use_cache: bool = True,
    overwrite_cache: bool = False,
    debug_octt: bool = False,
):
    time_bin_periods = ["DAY"]
    step_limit = 200

    parent_dir = Path(parent_directory)
    nodes_csv, plots_dir, outputs_dir = _check_and_prepare_paths(parent_dir)

    graph, origin_destination, demand = _load_input_data(
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
        outputs_dir=outputs_dir,
        compute_octt=compute_octt,
        use_cache=use_cache,
        overwrite_cache=overwrite_cache,
        debug_octt=debug_octt,
        parent_directory=parent_directory,
        tol=tol,
    )

    save_ue_results(
        parent_dir=parent_directory,
        UEflows=ue_flows,
        UEflowsBest=ue_flows_best,
        result=last_result,
    )


if __name__ == "__main__":
    find_transport_assignment_user_equilibrium(
        compute_octt=True,
        debug_octt=False,
        use_cache=False,
        overwrite_cache=True,
    )
