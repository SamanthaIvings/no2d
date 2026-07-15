from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from no2d_code.core import filepath_configs as fc
from no2d_code.solver.IO_operations import init_ue_logs, load_edges, load_filtered_od_and_demand
from no2d_code.solver.all_or_nothing_assignment import compute_all_or_nothing_flow
from no2d_code.solver.bpr import bpr_flow
from no2d_code.solver.digraph import Digraph
from no2d_code.solver.frank_wolfe_classes import FWRunConfig, FWResult
from no2d_code.solver.frankwolfe_ue_flex import solve_frank_wolfe_user_equilibrium
from no2d_code.core.octt_mapping import octt_from_traveltime as _octt_legacy
from no2d_code.solver.shortest_path_tree_builder import get_shortest_path_tree_edges_cell
from no2d_code.core.survey_octt_mapping import SurveyOCTTPipeline
from no2d_code.visualisation.fw_flow_plotter import plot_fw_flow_comparison
from no2d_code.visualisation.fw_stages_comparison import plot_car_stage1_vs_stage2


@dataclass(frozen=True)
class MultimodalConfig:
    tol: float = 100.0
    eps: float = 1e-6
    step_limit: int = 100

    bus_occupancy: float = 25.0

    bus_speed_factor_priority: float = 0.85
    bus_speed_factor_mixed: float = 0.65
    bus_capacity_factor: float = 3.0

    bike_speed_kmh_infra: float = 18.0
    bike_speed_kmh_mixed: float = 13.0
    bike_speed_kmh_bus_priority: float = 15.0
    bike_capacity_factor: float = 8.0

    car_bpr_alpha: float = 0.15
    car_bpr_beta: float = 4.0

    bus_bpr_alpha: float = 0.15
    bus_bpr_beta: float = 4.0

    bike_bpr_alpha: float = 0.08
    bike_bpr_beta: float = 2.0

    amc_c0: float = 1.0
    amc_c2: float = 0.1
    amc_scale: float = 100.0

    ct_k: float = 0.01
    ct_shift: float = 500.0

    bu_base_share: float = 0.1

    octt_mode: str = "survey"
    survey_xlsx: str | None = str(fc.data_path("inputs", "FF_Survey_responses.xlsx"))


def find_transport_assignment_user_equilibrium_multimodal(
    parent_directory: str = str(fc.DATA_DIR),
    cfg: MultimodalConfig = MultimodalConfig(),
    *,
    use_cache_stage1_car: bool = True,
    overwrite_cache_stage1_car: bool = False,
):
    nodes_csv = fc.input_path(parent_directory, fc.NODES_CSV)
    if not os.path.exists(nodes_csv):
        raise FileNotFoundError(f"Nodes csv file not found at: {nodes_csv}")

    plots_dir = os.path.join(parent_directory, "plots")
    outputs_dir = os.path.join(parent_directory, "outputs")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    edges_car = load_edges(parent_directory)
    origin_destination, demand_total = load_filtered_od_and_demand(parent_directory, cfg.tol)

    edges_bus, edges_bike = _load_mode_layers(parent_directory)
    edges_bus, edges_bike = _align_mode_layers(edges_car, edges_bus, edges_bike)

    graph_car = _make_car_graph(edges_car, cfg)
    run_cfg = _build_run_config(parent_directory, cfg)

    print("=== Stage 1: car UE on full demand ===")
    flows_car_1, res_car_1 = _run_single_ue(
        graph=graph_car,
        origin_destination=origin_destination,
        demand=demand_total,
        run_cfg=run_cfg,
    )

    _plot_flow(graph_car, origin_destination, demand_total, flows_car_1, nodes_csv,
               os.path.join(plots_dir, "fw_flow_compare_DAY_car_stage1.png"))

    traveltime_car_1 = bpr_flow(graph_car.free_flow_travel_h, flows_car_1, graph_car.capacity, graph_car.bpr_params)

    if cfg.octt_mode == "legacy":
        octt_fn = _octt_legacy
    else:
        if cfg.survey_xlsx is None:
            raise ValueError("octt_mode='survey' requires survey_xlsx in MultimodalConfig")
        pipe = SurveyOCTTPipeline(cfg.survey_xlsx)
        pipe.print_report()
        octt_fn = pipe.octt_from_traveltime

    octt_edge_car_1 = octt_fn(traveltime_car_1, debug=False)

    od_octt_mean = _compute_od_mean_edge_cost(
        graph=graph_car,
        origin_destination=origin_destination,
        edge_cost=octt_edge_car_1,
    )

    ct_car, bu_people, att_people = _split_od_demand_from_octt(
        demand_total=demand_total,
        od_octt=od_octt_mean,
        cfg=cfg,
    )

    out_split_csv = os.path.join(outputs_dir, "od_split_DAY_stage1.csv")
    _save_od_split_csv(out_split_csv, origin_destination, demand_total, ct_car, bu_people, att_people, od_octt_mean)
    print("Saved OD split:", out_split_csv)

    np.save(os.path.join(outputs_dir, "octt_edge_DAY_car_stage1.npy"), octt_edge_car_1)
    np.save(os.path.join(outputs_dir, "od_octt_mean_DAY_car_stage1.npy"), od_octt_mean)

    print("Stage1 totals:",
          f"total={float(demand_total.sum()):.3f},",
          f"car_CT={float(ct_car.sum()):.3f},",
          f"bus_BU={float(bu_people.sum()):.3f},",
          f"active_ATT={float(att_people.sum()):.3f}")

    print("=== Stage 2: UE per mode using split demand ===")

    graph_car_2 = _make_car_graph(edges_car, cfg)
    flows_car_2, _ = _run_single_ue(
        graph=graph_car_2,
        origin_destination=origin_destination,
        demand=ct_car,
        run_cfg=run_cfg,
    )
    _plot_flow(graph_car_2, origin_destination, ct_car, flows_car_2, nodes_csv,
               os.path.join(plots_dir, "fw_flow_compare_DAY_car_stage2.png"))

    plot_car_stage1_vs_stage2(
        graph=graph_car_2,
        flows_stage1=flows_car_1,
        flows_stage2=flows_car_2,
        nodes_csv=nodes_csv,
        out_png=os.path.join(plots_dir, "car_stage1_vs_stage2.png"),
    )

    graph_bus = _make_bus_graph(edges_car, edges_bus, cfg)
    bus_veh = bu_people / max(cfg.bus_occupancy, 1e-12)
    flows_bus, _ = _run_single_ue(
        graph=graph_bus,
        origin_destination=origin_destination,
        demand=bus_veh,
        run_cfg=run_cfg,
    )
    _plot_flow(graph_bus, origin_destination, bus_veh, flows_bus, nodes_csv,
               os.path.join(plots_dir, "fw_flow_compare_DAY_bus_stage2.png"))

    graph_bike = _make_bike_graph(edges_car, edges_bus, edges_bike, cfg)
    flows_bike, _ = _run_single_ue(
        graph=graph_bike,
        origin_destination=origin_destination,
        demand=att_people,
        run_cfg=run_cfg,
    )
    _plot_flow(graph_bike, origin_destination, att_people, flows_bike, nodes_csv,
               os.path.join(plots_dir, "fw_flow_compare_DAY_bike_stage2.png"))

    report_stage2_stats(
        demand_total=demand_total,
        ct_car=ct_car,
        bu_people=bu_people,
        att_people=att_people,
        flows_car_stage1=flows_car_1,
        flows_car_stage2=flows_car_2,
        flows_bus_stage2=flows_bus,
        flows_bike_stage2=flows_bike,
        cfg=cfg,
        edges_df=edges_car,
        top_k_edges=10,
    )

    out_npz = os.path.join(outputs_dir, "ue_results_multimodal_DAY.npz")
    np.savez(
        out_npz,
        demand_total=demand_total,
        od_octt_mean_stage1=od_octt_mean,
        ct_car=ct_car,
        bu_people=bu_people,
        att_people=att_people,
        flows_car_stage1=flows_car_1,
        flows_car_stage2=flows_car_2,
        flows_bus_stage2=flows_bus,
        flows_bike_stage2=flows_bike,
    )
    print("Saved:", out_npz)


def _build_run_config(parent_directory: str, cfg: MultimodalConfig) -> FWRunConfig:
    txt_name, crit_log_name, crit_bests_name = init_ue_logs(parent_directory, cfg.step_limit)
    return FWRunConfig(
        eps=cfg.eps,
        steplimit=cfg.step_limit,
        txt_name=txt_name,
        crit_log_name=crit_log_name,
        crit_bests_name=crit_bests_name,
    )


def _run_single_ue(
    *,
    graph: Digraph,
    origin_destination: np.ndarray,
    demand: np.ndarray,
    run_cfg: FWRunConfig,
) -> Tuple[np.ndarray, FWResult]:
    graph.weight = bpr_flow(
        graph.free_flow_travel_h,
        np.zeros(graph.u.size),
        graph.capacity,
        graph.bpr_params,
    )

    result = solve_frank_wolfe_user_equilibrium(
        demand=demand,
        graph=graph,
        origin_destination=origin_destination,
        config=run_cfg,
    )

    return np.asarray(result.flows, dtype=float), result


def _plot_flow(
    graph: Digraph,
    origin_destination: np.ndarray,
    demand: np.ndarray,
    flows: np.ndarray,
    nodes_csv: str,
    out_png: str,
):
    from no2d_code.solver.shortest_path_tree_builder import (
        get_unique_origins, build_shortest_path_trees,
    )
    flow0 = np.zeros(graph.u.size, dtype=float)
    graph.weight = bpr_flow(graph.free_flow_travel_h, flow0, graph.capacity, graph.bpr_params)
    origins = get_unique_origins(origin_destination)
    shortest_path_tree = build_shortest_path_trees(graph, origins)
    aon = compute_all_or_nothing_flow(demand, origin_destination, shortest_path_tree, graph.u.size)

    plot_fw_flow_comparison(
        graph=graph,
        flow_init=aon,
        flow_final=flows,
        nodes=nodes_csv,
        out_path=out_png,
    )
    print("Saved plot:", out_png)


def _load_mode_layers(parent_directory: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = os.path.join(parent_directory, fc.DATA_INPUTS_DIR)

    edges_bus = pd.read_pickle(os.path.join(base, "edges_bus.pkl"))
    edges_bike = pd.read_pickle(os.path.join(base, "edges_bike.pkl"))

    return edges_bus, edges_bike


def _align_mode_layers(
    edges_car: pd.DataFrame,
    edges_bus: pd.DataFrame,
    edges_bike: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if edges_bus.shape[0] != edges_car.shape[0] or edges_bike.shape[0] != edges_car.shape[0]:
        raise ValueError("Layer pickles must have same number of rows as inputs/edges.csv")

    for c in ("u", "v", "key"):
        if c in edges_car.columns and c in edges_bus.columns and c in edges_bike.columns:
            if not (edges_car[c].to_numpy() == edges_bus[c].to_numpy()).all():
                raise ValueError("edges_bus row order does not match edges_car (u mismatch)")
            if not (edges_car[c].to_numpy() == edges_bike[c].to_numpy()).all():
                raise ValueError("edges_bike row order does not match edges_car (u mismatch)")

    return edges_bus, edges_bike


def _make_car_graph(edges_car: pd.DataFrame, cfg: MultimodalConfig) -> Digraph:
    g = Digraph.from_edges(edges_car, cfg.eps)
    g.bpr_params = _pack_bpr(cfg.car_bpr_alpha, cfg.car_bpr_beta, cfg.eps, g.u.size)
    return g


def _make_bus_graph(edges_car: pd.DataFrame, edges_bus: pd.DataFrame, cfg: MultimodalConfig) -> Digraph:
    df = edges_car.copy()
    is_priority = edges_bus.get("is_bus_priority", pd.Series(False, index=df.index)).to_numpy(dtype=bool)

    speed = df["speedlim"].to_numpy(dtype=float)
    speed = np.where(is_priority, speed * cfg.bus_speed_factor_priority, speed * cfg.bus_speed_factor_mixed)
    speed = np.maximum(speed, 5.0)
    df["speedlim"] = speed

    df["capacity"] = df["capacity"].to_numpy(dtype=float) * cfg.bus_capacity_factor

    g = Digraph.from_edges(df, cfg.eps)
    g.bpr_params = _pack_bpr(cfg.bus_bpr_alpha, cfg.bus_bpr_beta, cfg.eps, g.u.size)
    return g


def _make_bike_graph(
    edges_car: pd.DataFrame,
    edges_bus: pd.DataFrame,
    edges_bike: pd.DataFrame,
    cfg: MultimodalConfig,
) -> Digraph:
    df = edges_car.copy()

    is_bike = edges_bike.get("is_bike_infra", pd.Series(False, index=df.index)).to_numpy(dtype=bool)
    is_bus_priority = edges_bus.get("is_bus_priority", pd.Series(False, index=df.index)).to_numpy(dtype=bool)

    bike_speed = np.full(df.shape[0], cfg.bike_speed_kmh_mixed, dtype=float)
    bike_speed = np.where(is_bus_priority & (~is_bike), cfg.bike_speed_kmh_bus_priority, bike_speed)
    bike_speed = np.where(is_bike, cfg.bike_speed_kmh_infra, bike_speed)
    bike_speed = np.maximum(bike_speed, 6.0)
    df["speedlim"] = bike_speed

    df["capacity"] = df["capacity"].to_numpy(dtype=float) * cfg.bike_capacity_factor

    g = Digraph.from_edges(df, cfg.eps)
    g.bpr_params = _pack_bpr(cfg.bike_bpr_alpha, cfg.bike_bpr_beta, cfg.eps, g.u.size)
    return g


def _pack_bpr(alpha: float, beta: float, eps: float, n_edges: int) -> np.ndarray:
    out = np.empty((n_edges, 3), dtype=float)
    out[:, 0] = float(alpha)
    out[:, 1] = float(beta)
    out[:, 2] = float(eps)
    return out


def _build_shortest_path_trees_all_origins(graph: Digraph) -> List[List[List[int]]]:
    e_store: List[List[List[int]]] = [None] * graph.n_nodes  # type: ignore[assignment]
    for i in range(graph.n_nodes):
        e_store[i] = get_shortest_path_tree_edges_cell(graph, i)
    return e_store


def _compute_od_mean_edge_cost(
    *,
    graph: Digraph,
    origin_destination: np.ndarray,
    edge_cost: np.ndarray,
) -> np.ndarray:
    prev_weight = graph.weight
    graph.weight = edge_cost
    e_store = _build_shortest_path_trees_all_origins(graph)
    graph.weight = prev_weight

    od_mean = np.full(origin_destination.shape[0], np.inf, dtype=float)
    for i in range(origin_destination.shape[0]):
        s = int(origin_destination[i, 2])
        t = int(origin_destination[i, 3])
        edgepath = e_store[s][t]
        if edgepath:
            idx = np.asarray(edgepath, dtype=int)
            total = float(np.sum(edge_cost[idx]))
            k = int(idx.size)
            od_mean[i] = total / max(k, 1)
    return od_mean


def _split_od_demand_from_octt(
    *,
    demand_total: np.ndarray,
    od_octt: np.ndarray,
    cfg: MultimodalConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    octt = _normalize_od_octt(od_octt, lo_q=0.10, hi_q=0.90)

    amc = cfg.amc_c0 + cfg.amc_c2 * (octt * cfg.amc_scale) ** 2
    ct = demand_total / (1.0 + np.exp(cfg.ct_k * (amc - cfg.ct_shift)))

    bu = cfg.bu_base_share * demand_total - cfg.bu_base_share * ct
    att = demand_total - bu - ct

    ct = np.clip(ct, 0.0, demand_total)
    bu = np.clip(bu, 0.0, demand_total - ct)
    att = np.clip(att, 0.0, demand_total - ct - bu)

    return ct.astype(float), bu.astype(float), att.astype(float)


def _normalize_od_octt(od_octt: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    a = np.asarray(od_octt, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a, dtype=float)

    lo = float(np.quantile(finite, lo_q))
    hi = float(np.quantile(finite, hi_q))
    denom = max(hi - lo, 1e-12)
    z = (a - lo) / denom
    return np.clip(z, 0.0, 1.0)


def _save_od_split_csv(
    out_csv: str,
    origin_destination: np.ndarray,
    demand_total: np.ndarray,
    ct: np.ndarray,
    bu: np.ndarray,
    att: np.ndarray,
    od_octt: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "OD_i": origin_destination[:, 0].astype(int),
            "OD_j": origin_destination[:, 1].astype(int),
            "O_node": origin_destination[:, 2].astype(int),
            "D_node": origin_destination[:, 3].astype(int),
            "demand_total": demand_total.astype(float),
            "od_octt_mean": od_octt.astype(float),
            "CT_car": ct.astype(float),
            "BU_bus_people": bu.astype(float),
            "ATT_active_people": att.astype(float),
        }
    )
    df.to_csv(out_csv, index=False)


def report_stage2_stats(
    *,
    demand_total: np.ndarray,
    ct_car: np.ndarray,
    bu_people: np.ndarray,
    att_people: np.ndarray,
    flows_car_stage1: np.ndarray,
    flows_car_stage2: np.ndarray,
    flows_bus_stage2: np.ndarray,
    flows_bike_stage2: np.ndarray,
    cfg: MultimodalConfig,
    edges_df: pd.DataFrame | None = None,
    top_k_edges: int = 10,
) -> None:
    total = float(np.sum(demand_total))
    ct = float(np.sum(ct_car))
    bu = float(np.sum(bu_people))
    att = float(np.sum(att_people))

    def _pct(x: float) -> float:
        return 100.0 * x / total if total > 0 else 0.0

    print("\n=== Stage 2 statistical report ===")
    print(f"Total OD demand: {total:.3f}")
    print(f"Car CT:   {ct:.3f} ({_pct(ct):.2f}%)")
    print(f"Bus BU:   {bu:.3f} ({_pct(bu):.2f}%)  -> vehicles approx {bu / max(cfg.bus_occupancy, 1e-12):.3f}")
    print(f"Active:   {att:.3f} ({_pct(att):.2f}%)")

    f1 = np.asarray(flows_car_stage1, dtype=float)
    f2 = np.asarray(flows_car_stage2, dtype=float)
    db = np.asarray(flows_bus_stage2, dtype=float)
    dk = np.asarray(flows_bike_stage2, dtype=float)

    if f1.shape != f2.shape:
        raise ValueError("flows_car_stage1 and flows_car_stage2 must have same shape")

    sum_f1 = float(np.sum(f1))
    sum_f2 = float(np.sum(f2))
    drop_abs = sum_f1 - sum_f2
    drop_pct = 100.0 * drop_abs / sum_f1 if sum_f1 > 0 else 0.0

    d = f2 - f1
    abs_d = np.abs(d)

    def _q(a: np.ndarray, q: float) -> float:
        return float(np.quantile(a, q)) if a.size else 0.0

    print("\nCar edge-flow totals (sum over edges):")
    print(f" Stage 1: {sum_f1:.3f}")
    print(f" Stage 2: {sum_f2:.3f}")
    print(f" Drop:    {drop_abs:.3f} ({drop_pct:.2f}%)")

    print("\nEdge-level change stats for car flows (stage2 - stage1):")
    print(f" Mean delta: {float(np.mean(d)):.6f}")
    print(f" Median delta: {float(np.median(d)):.6f}")
    print(f" Mean |delta|: {float(np.mean(abs_d)):.6f}")
    print(f" 90th pct |delta|: {_q(abs_d, 0.90):.6f}")
    print(f" 99th pct |delta|: {_q(abs_d, 0.99):.6f}")
    print(f" Max |delta|: {float(np.max(abs_d)):.6f}")

    if f1.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(f1 > 0, d / f1, np.nan)
        finite_rel = rel[np.isfinite(rel)]
        print("\nRelative change where stage1>0 (delta/stage1):")
        print(f" Median: {float(np.median(finite_rel)):.6f}")
        print(f" 10th pct: {float(np.quantile(finite_rel, 0.10)):.6f}")
        print(f" 90th pct: {float(np.quantile(finite_rel, 0.90)):.6f}")

    print("\nStage 2 other mode edge-flow totals (sum over edges):")
    print(f" Bus vehicles:  {float(np.sum(db)):.3f}")
    print(f" Bike people:   {float(np.sum(dk)):.3f}")

    if edges_df is not None and top_k_edges > 0:
        idx = np.argsort(abs_d)[::-1][:top_k_edges]
        cols = [c for c in ["u", "v", "key", "highway", "name"] if c in edges_df.columns]
        print(f"\nTop {top_k_edges} edges by |car flow change|:")
        for rank, i in enumerate(idx, 1):
            meta = ""
            if cols:
                row = edges_df.iloc[int(i)]
                meta = " | " + ", ".join(f"{c}={row[c]}" for c in cols)
            print(
                f" {rank:02d}. edge_idx={int(i)}  "
                f"stage1={float(f1[i]):.6f}  stage2={float(f2[i]):.6f}  "
                f"delta={float(d[i]):+.6f}{meta}"
            )

    print("=== End report ===\n")


if __name__ == "__main__":
    find_transport_assignment_user_equilibrium_multimodal()
