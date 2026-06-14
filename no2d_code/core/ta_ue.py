from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

import numpy as np

from no2d_code.core import filepath_configs as fc
from no2d_code.solver.IO_operations import (
    init_ue_logs,
    load_edges,
    load_filtered_od_and_demand,
    save_ue_results,
    has_ue_cache_pickle,
    load_ue_cache_pickle,
    save_ue_cache_pickle,
)
from no2d_code.solver.bpr import bpr_flow
from no2d_code.solver.frank_wolfe_classes import FWResult, FWRunConfig
from no2d_code.solver.frankwolfe_ue_flex import solve_frank_wolfe_user_equilibrium
from no2d_code.solver.digraph import Digraph
from no2d_code.core.octt_mapping import (
    octt_from_traveltime as _octt_legacy,
    att_bu_from_octt as _att_bu_legacy,
)
from no2d_code.core.survey_octt_mapping import SurveyOCTTPipeline
from no2d_code.solver.shortest_path_tree_builder import get_shortest_path_tree_edges_cell
from no2d_code.visualisation.fw_flow_plotter import (
    plot_fw_flow_comparison,
    plot_edge_value_comparison,
    plot_fw_lsoa_flow_comparison,
    plot_lsoa_value_comparison,
    build_edge_lsoa_map,
)
from no2d_code.visualisation.lsoa_metric_plotter import plot_lsoa_value_state


def _resolve_octt_fns(octt_mode: str, survey_xlsx: Optional[str] = None):
    if octt_mode == "legacy":
        return _octt_legacy, _att_bu_legacy
    if octt_mode == "survey":
        if survey_xlsx is None:
            raise ValueError("octt_mode='survey' requires survey_xlsx path")
        pipe = SurveyOCTTPipeline(survey_xlsx)
        pipe.print_report()
        return pipe.octt_from_traveltime, pipe.att_bu_from_octt
    raise ValueError(f"Unknown octt_mode {octt_mode!r}")


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


def _resolve_lsoa_polygons_path(
    parent_dir: Path,
    tol: float,
    lsoa_polygons_path: Optional[Path],
) -> Path:
    if lsoa_polygons_path is not None:
        lsoa_path = Path(lsoa_polygons_path)
        if not lsoa_path.exists():
            raise FileNotFoundError(f"LSOA polygons not found: {lsoa_path}")
        return lsoa_path

    candidates = [
        parent_dir / "inputs" / "fw_inputs_sy_simplified" / f"lDists_tol{tol}.csv",
        parent_dir / "inputs" / "fw_inputs_sy_simplified" / "lDists.csv",
        parent_dir / "inputs" / "geo" / "LSOA_2011_EW_BGC.zip",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "LSOA polygons not found. Provide lsoa_polygons_path "
        "or add lDists.csv / lDists_tol{tol}.csv under data/inputs."
    )


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
        e_store[int(s)] = get_shortest_path_tree_edges_cell(graph, int(s))

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
        e_store[int(s)] = get_shortest_path_tree_edges_cell(graph, int(s))

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
    plot_lsoa: bool,
    lsoa_polygons_path: Optional[Path],
    edge_lsoa_map_path: Optional[Path],
    lsoa_name_filter: Optional[List[str]],
    octt_mode: str,
    survey_xlsx: Optional[str],
):
    ue_flows_col = []
    ue_flows_best_col = []
    last_result: Optional[FWResult] = None

    lsoa_polygons = None
    edge_lsoa_map = None
    if plot_lsoa:
        lsoa_polygons = _resolve_lsoa_polygons_path(
            Path(parent_directory), tol, lsoa_polygons_path,
        )
        if edge_lsoa_map_path is not None:
            edge_lsoa_map = Path(edge_lsoa_map_path)
        else:
            edge_lsoa_map = outputs_dir / f"edge_lsoa_map_tol{tol}.csv"
        if not edge_lsoa_map.exists():
            print(f"[LSOA] Building edge->LSOA map: {edge_lsoa_map}")
            build_edge_lsoa_map(
                graph, nodes=nodes_csv, lsoa_polygons=lsoa_polygons,
                out_path=edge_lsoa_map, lsoa_name_filter=lsoa_name_filter,
            )

    # resolve OCTT functions once before loop
    octt_fn = att_bu_fn = None
    if compute_octt:
        octt_fn, att_bu_fn = _resolve_octt_fns(octt_mode, survey_xlsx)

    for tag in time_bin_periods:
        print(f"Time bin ... ({tag})")

        flow0 = np.zeros(graph.u.size, dtype=float)
        graph.weight = bpr_flow(
            graph.free_flow_travel_h, flow0, graph.capacity, graph.bpr_params,
        )

        print("[AoN] Computing initial all-or-nothing flow...")
        aon_flow = _compute_all_or_nothing_flow_unique_origins(
            demand=demand, origin_destination=origin_destination, graph=graph,
        )

        cache_ok = False
        if use_cache and not overwrite_cache and has_ue_cache_pickle(parent_directory, tag):
            ue_flow, ue_flow_best, result, meta = load_ue_cache_pickle(parent_directory, tag)
            print(f"[FW] Loaded cache: {meta}")
            ue_flow = np.asarray(ue_flow)
            ue_flow_best = np.asarray(ue_flow_best)
            if ue_flow.size == graph.u.size and ue_flow_best.size == graph.u.size:
                cache_ok = True
            else:
                print(f"[FW] Cache flow length mismatch; recomputing.")

        if not cache_ok:
            result = solve_frank_wolfe_user_equilibrium(
                demand=demand, graph=graph,
                origin_destination=origin_destination, config=run_cfg,
            )
            ue_flow = result.flows
            ue_flow_best = result.flows_best

            if use_cache:
                save_ue_cache_pickle(
                    parent_dir=parent_directory, tag=tag,
                    UEflows_col=ue_flow, UEflowsBest_col=ue_flow_best,
                    result=result,
                    meta={"tol": tol, "eps": run_cfg.eps, "steplimit": run_cfg.steplimit},
                )

        ue_flows_col.append(np.asarray(ue_flow))
        ue_flows_best_col.append(np.asarray(ue_flow_best))
        last_result = result

        nodes_arr = np.genfromtxt(nodes_csv, delimiter=",", names=True)
        node_col = "node" if "node" in nodes_arr.dtype.names else nodes_arr.dtype.names[0]
        node_ids = np.asarray([r[node_col] for r in nodes_arr], dtype=int)

        if node_ids.min() != 0 or node_ids.max() != graph.n_nodes - 1:
            raise ValueError("nodes.csv does not match Digraph node ids.")

        out_pdf = plots_dir / f"fw_flow_change_{tag}.pdf"
        plot_fw_flow_comparison(
            graph=graph, flow_init=aon_flow, flow_final=np.asarray(ue_flow),
            nodes=nodes_csv, out_path=out_pdf, delta_only=True, cbar_numbers=False,
        )

        if plot_lsoa:
            plot_fw_lsoa_flow_comparison(
                graph=graph, flow_init=aon_flow, flow_final=np.asarray(ue_flow),
                nodes=nodes_csv, lsoa_polygons=lsoa_polygons,
                edge_lsoa_map=edge_lsoa_map,
                out_path=plots_dir / f"fw_lsoa_flow_change_{tag}.png",
                lsoa_name_filter=lsoa_name_filter,
            )

        if compute_octt:
            tt_aon = bpr_flow(graph.free_flow_travel_h, aon_flow, graph.capacity, graph.bpr_params)
            tt_ue = bpr_flow(graph.free_flow_travel_h, np.asarray(ue_flow), graph.capacity, graph.bpr_params)

            octt_aon = octt_fn(tt_aon, debug=debug_octt)
            octt_ue = octt_fn(tt_ue, debug=debug_octt)

            plot_edge_value_comparison(
                graph=graph, value_init=octt_aon, value_final=octt_ue,
                nodes=nodes_csv, out_path=plots_dir / f"fw_octt_change_{tag}.pdf",
                title_init="AoN OCTT", title_final="UE OCTT",
                cbar_label="OCTT", delta_label="ΔOCTT",
                delta_only=True, cbar_numbers=False,
            )

            if plot_lsoa:
                plot_lsoa_value_comparison(
                    graph=graph, value_init=octt_aon, value_final=octt_ue,
                    nodes=nodes_csv, lsoa_polygons=lsoa_polygons,
                    edge_lsoa_map=edge_lsoa_map,
                    out_path=plots_dir / f"fw_lsoa_octt_change_{tag}.png",
                    lsoa_name_filter=lsoa_name_filter, cbar_label="Change in OCTT",
                )

            att_aon, bu_aon = att_bu_fn(octt_aon)
            att_ue, bu_ue = att_bu_fn(octt_ue)

            plot_edge_value_comparison(
                graph=graph, value_init=att_aon, value_final=att_ue,
                nodes=nodes_csv, out_path=plots_dir / f"fw_att_change_{tag}.pdf",
                title_init="AoN ATT", title_final="UE ATT",
                cbar_label="ATT", delta_label="ΔATT",
                delta_only=True, cbar_numbers=False,
            )

            plot_edge_value_comparison(
                graph=graph, value_init=bu_aon, value_final=bu_ue,
                nodes=nodes_csv, out_path=plots_dir / f"fw_bu_change_{tag}.pdf",
                title_init="AoN BU", title_final="UE BU",
                cbar_label="BU", delta_label="ΔBU",
                delta_only=True, cbar_numbers=False,
            )

            if plot_lsoa:
                plot_lsoa_value_comparison(
                    graph=graph, value_init=att_aon, value_final=att_ue,
                    nodes=nodes_csv, lsoa_polygons=lsoa_polygons,
                    edge_lsoa_map=edge_lsoa_map,
                    out_path=plots_dir / f"fw_lsoa_att_change_{tag}.png",
                    lsoa_name_filter=lsoa_name_filter, cbar_label="Change in ATT",
                )

                plot_lsoa_value_comparison(
                    graph=graph, value_init=bu_aon, value_final=bu_ue,
                    nodes=nodes_csv, lsoa_polygons=lsoa_polygons,
                    edge_lsoa_map=edge_lsoa_map,
                    out_path=plots_dir / f"fw_lsoa_bu_change_{tag}.png",
                    lsoa_name_filter=lsoa_name_filter, cbar_label="Change in BU",
                )

                plot_lsoa_value_state(
                    value=att_ue, lsoa_polygons=lsoa_polygons,
                    edge_lsoa_map=edge_lsoa_map,
                    out_path=plots_dir / f"lsoa_att_ue_{tag}.png",
                    title="ATT after optimisation (UE)", cbar_label="ATT",
                    agg="mean", lsoa_name_filter=lsoa_name_filter,
                )

                plot_lsoa_value_state(
                    value=bu_ue, lsoa_polygons=lsoa_polygons,
                    edge_lsoa_map=edge_lsoa_map,
                    out_path=plots_dir / f"lsoa_bu_ue_{tag}.png",
                    title="BU after optimisation (UE)", cbar_label="BU",
                    agg="mean", lsoa_name_filter=lsoa_name_filter,
                )

            graph.weight = octt_ue
            od_octt = _compute_od_costs_unique_origins(
                origin_destination=origin_destination, edge_cost=octt_ue, graph=graph,
            )

            np.save(outputs_dir / f"octt_edge_{tag}.npy", octt_ue)
            np.save(outputs_dir / f"att_edge_{tag}.npy", att_ue)
            np.save(outputs_dir / f"bu_edge_{tag}.npy", bu_ue)
            np.save(outputs_dir / f"od_octt_{tag}.npy", od_octt)

    return (
        np.column_stack(ue_flows_col),
        np.column_stack(ue_flows_best_col),
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
    plot_lsoa: bool = True,
    lsoa_polygons_path: Optional[str] = None,
    edge_lsoa_map_path: Optional[str] = None,
    lsoa_name_filter: Optional[List[str]] = None,
    octt_mode: str = "survey",
    survey_xlsx: Optional[str] = None,
):
    time_bin_periods = ["DAY"]
    step_limit = 30

    parent_dir = Path(parent_directory)
    nodes_csv, plots_dir, outputs_dir = _check_and_prepare_paths(parent_dir)

    graph, origin_destination, demand = _load_input_data(
        parent_directory=parent_directory, tol=tol, eps=eps,
    )

    run_cfg = _build_run_config(
        parent_directory=parent_directory, eps=eps, step_limit=step_limit,
    )

    ue_flows, ue_flows_best, last_result = _run_time_bins(
        time_bin_periods=time_bin_periods,
        graph=graph, demand=demand,
        origin_destination=origin_destination, run_cfg=run_cfg,
        nodes_csv=nodes_csv, plots_dir=plots_dir, outputs_dir=outputs_dir,
        compute_octt=compute_octt, use_cache=use_cache,
        overwrite_cache=overwrite_cache, debug_octt=debug_octt,
        parent_directory=parent_directory, tol=tol,
        plot_lsoa=plot_lsoa,
        lsoa_polygons_path=None if lsoa_polygons_path is None else Path(lsoa_polygons_path),
        edge_lsoa_map_path=None if edge_lsoa_map_path is None else Path(edge_lsoa_map_path),
        lsoa_name_filter=lsoa_name_filter,
        octt_mode=octt_mode,
        survey_xlsx=survey_xlsx,
    )

    save_ue_results(
        parent_dir=parent_directory,
        UEflows=ue_flows, UEflowsBest=ue_flows_best, result=last_result,
    )


if __name__ == "__main__":
    find_transport_assignment_user_equilibrium(
        compute_octt=True,
        debug_octt=False,
        use_cache=False,
        overwrite_cache=True,
        plot_lsoa=False,
        lsoa_name_filter=["Barnsley", "Doncaster", "Rotherham", "Sheffield"],
        octt_mode="survey",
        survey_xlsx="../../data/inputs/FF_Survey_responses.xlsx",
    )