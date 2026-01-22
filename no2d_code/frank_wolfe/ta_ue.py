from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

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
    compute_aon_flow,
    plot_fw_flow_comparison,
    plot_edge_value_comparison,
)


def _check_and_prepare_paths(parent_dir: Path) -> tuple[Path, Path, Path]:
    nodes_csv = parent_dir / "inputs/nodes.csv"
    if not nodes_csv.exists():
        raise FileNotFoundError(f"nodes.csv not found at: {nodes_csv}")

    plots_dir = parent_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = parent_dir / "outputs"
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


def _compute_od_costs_from_shortest_path_trees(
    origin_destination: np.ndarray,
    e_store: List[List[List[int]]],
    edge_cost: np.ndarray,
) -> np.ndarray:
    od_cost = np.full(origin_destination.shape[0], np.inf, dtype=float)
    for i in range(origin_destination.shape[0]):
        s = int(origin_destination[i, 2])
        t = int(origin_destination[i, 3])
        edgepath = e_store[s][t]
        if edgepath:
            idx = np.asarray(edgepath, dtype=int)
            od_cost[i] = float(np.sum(edge_cost[idx]))
    return od_cost


def _build_shortest_path_trees_all_origins(graph: Digraph) -> List[List[List[int]]]:
    e_store: List[List[List[int]]] = [None] * graph.n_nodes  # type: ignore[assignment]
    for i in range(graph.n_nodes):
        e_store[i] = shortestpathtree_edges_cell(graph, i)
    return e_store


def _save_octt_outputs(
    outputs_dir: Path,
    tag: str,
    *,
    octt_edge: np.ndarray,
    od_octt: np.ndarray,
) -> None:
    np.save(outputs_dir / f"octt_edge_{tag}.npy", octt_edge)
    np.save(outputs_dir / f"od_octt_{tag}.npy", od_octt)


def _p(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, [0, 1, 5, 50, 95, 99, 100])


def _run_time_bins(
    time_bin_periods: List[str],
    graph: Digraph,
    demand: np.ndarray,
    origin_destination: np.ndarray,
    run_cfg: FWRunConfig,
    nodes_csv: Path,
    plots_dir: Path,
    outputs_dir: Path,
    *,
    compute_octt: bool,
    use_cache: bool,
    overwrite_cache: bool,
    debug_octt: bool,
    parent_directory: str,
    tol: float,
):
    ue_flows: List[np.ndarray] = []
    ue_flows_best: List[np.ndarray] = []
    last_result: Optional[FWResult] = None

    for i, tag in enumerate(time_bin_periods):
        print(f"Time bin ...{i + 1} ({tag})")

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

        loaded_from_cache = False
        if use_cache and (not overwrite_cache) and has_ue_cache_pickle(parent_directory, tag):
            UEflows_col, UEflowsBest_col, result, meta = load_ue_cache_pickle(parent_directory, tag)
            loaded_from_cache = True
            print(f"Loaded UE cache for tag={tag}: {meta}")
        else:
            print("Starting user-equilibrium Frank-Wolfe...")
            result = frank_wolfe_ue_solver(
                demand=demand,
                graph=graph,
                origin_destination=origin_destination,
                config=run_cfg,
            )
            UEflows_col = result.flows
            UEflowsBest_col = result.flows_best

            if use_cache:
                save_ue_cache_pickle(
                    parent_dir=parent_directory,
                    tag=tag,
                    UEflows_col=UEflows_col,
                    UEflowsBest_col=UEflowsBest_col,
                    result=result,
                    meta={
                        "tol": float(tol),
                        "eps": float(run_cfg.eps),
                        "steplimit": int(run_cfg.steplimit),
                        "octt_mapping": "octt_mapping.octt_from_traveltime",
                    },
                )
                print(f"Saved UE cache for tag={tag}")

        ue_flows.append(np.asarray(UEflows_col, dtype=float))
        ue_flows_best.append(np.asarray(UEflowsBest_col, dtype=float))
        last_result = result

        out_png = plots_dir / f"fw_flow_compare_{tag}.png"
        plot_fw_flow_comparison(
            graph=graph,
            flow_init=aon_flow,
            flow_final=np.asarray(UEflows_col, dtype=float),
            nodes=nodes_csv,
            out_path=out_png,
        )
        print(f"Saved plot: {out_png}")

        if compute_octt:
            traveltime_aon = bpr_flow(
                graph.free_flow_travel_h,
                aon_flow,
                graph.capacity,
                graph.bpr_params,
            )
            traveltime_ue = bpr_flow(
                graph.free_flow_travel_h,
                np.asarray(UEflows_col, dtype=float),
                graph.capacity,
                graph.bpr_params,
            )

            octt_aon = octt_from_traveltime(traveltime_aon, debug=debug_octt)
            octt_ue = octt_from_traveltime(traveltime_ue, debug=debug_octt)

            if debug_octt:
                print("octt_delta pct:", _p(octt_ue - octt_aon))

            out_octt_png = plots_dir / f"fw_octt_compare_{tag}.png"
            plot_edge_value_comparison(
                graph=graph,
                value_init=octt_aon,
                value_final=octt_ue,
                nodes=nodes_csv,
                out_path=out_octt_png,
                title_init="Initial (AoN) OCTT",
                title_final="Final (UE) OCTT",
                cbar_label="OCTT",
                delta_label="OCTT change",
            )
            print(f"Saved plot: {out_octt_png}")

            octt_edge = octt_ue
            prev_weight = graph.weight
            graph.weight = octt_edge
            e_store_octt = _build_shortest_path_trees_all_origins(graph)
            od_octt = _compute_od_costs_from_shortest_path_trees(
                origin_destination=origin_destination,
                e_store=e_store_octt,
                edge_cost=octt_edge,
            )
            graph.weight = prev_weight

            _save_octt_outputs(
                outputs_dir=outputs_dir,
                tag=tag,
                octt_edge=octt_edge,
                od_octt=od_octt,
            )
            print(f"Saved OCTT arrays for tag={tag}")

            rho_aon = aon_flow / np.maximum(graph.capacity, 1e-12)
            rho_ue = np.asarray(UEflows_col, dtype=float) / np.maximum(graph.capacity, 1e-12)

            rho_clip = np.percentile(rho_ue, 99)
            rho_aon = np.clip(rho_aon, 0.0, rho_clip)
            rho_ue = np.clip(rho_ue, 0.0, rho_clip)

            out_png = plots_dir / f"fw_density_compare_{tag}.png"
            plot_edge_value_comparison(
                graph=graph,
                value_init=rho_aon,
                value_final=rho_ue,
                nodes=nodes_csv,
                out_path=out_png,
                title_init="Initial (AoN) density",
                title_final="Final (UE) density",
                cbar_label="density",
                delta_label="Density change",
            )
            print(f"Saved plot: {out_png}")

        if loaded_from_cache:
            print("UE loaded from cache (no FW run).")

    if last_result is None:
        raise RuntimeError("No Frank–Wolfe iterations were run")

    n_edges = graph.u.size
    ue_flows_arr = np.column_stack(ue_flows) if ue_flows else np.empty((n_edges, 0), dtype=float)
    ue_flows_best_arr = np.column_stack(ue_flows_best) if ue_flows_best else np.empty((n_edges, 0), dtype=float)

    return ue_flows_arr, ue_flows_best_arr, last_result


def find_transport_assignment_user_equilibrium(
    tol: float = 58.6,
    parent_directory: str = "../../data/",
    eps: float = 1e-6,
    *,
    compute_octt: bool = True,
    use_cache: bool = True,
    overwrite_cache: bool = False,
    debug_octt: bool = False,
):
    time_bin_periods = ["DAY"]
    step_limit = 100

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
        debug_octt=True,
        use_cache=True,
        overwrite_cache=False,
    )
