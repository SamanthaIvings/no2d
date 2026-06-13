"""
ta_ue_factorial_average.py
==========================
Full-factorial averaging of the survey OCTT pipeline over 3^6 = 729
combinations of (mean, +SD, -SD) per case, then standard 2-stage FW
using the averaged OCTT mapping.

Method: probabilistic sensitivity analysis via exhaustive factorial design.
Each of the 6 survey cases has 3 regression variants (mean, mean+SD, mean-SD).
All 729 combinations are evaluated on the fixed Stage-1 travel times, and
the resulting OCTT vectors are averaged element-wise. This averaged OCTT
is then used for mode split and Stage-2 assignment.

Usage:
    python ta_ue_factorial_average.py [--data-dir ../../data]
                                      [--survey-xlsx ../../data/inputs/FF_Survey_responses.xlsx]
                                      [--step-limit 300]
"""

from __future__ import annotations

import argparse
import itertools
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from no2d_code.frank_wolfe.ta_ue_multimodal_extension import (
    MultimodalConfig,
    _load_mode_layers,
    _align_mode_layers,
    _make_car_graph,
    _make_bus_graph,
    _make_bike_graph,
    _build_run_config,
    _run_single_ue,
    _plot_flow,
    _compute_od_mean_edge_cost,
    _split_od_demand_from_octt,
    _save_od_split_csv,
    report_stage2_stats,
)
from no2d_code.frank_wolfe import filepath_configs as fc
from no2d_code.frank_wolfe.IO_operations import load_edges, load_filtered_od_and_demand
from no2d_code.frank_wolfe.bpr import bpr_flow
from no2d_code.visualisation.fw_stages_comparison import plot_car_stage1_vs_stage2


CASE_DEFS = [
    dict(n=1, cols=[f"S1_Congestion_vs_ActiveTransport_Case{i}" for i in range(1, 6)],
         x=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), model="poly"),
    dict(n=2, cols=[f"S2_ActiveTrips_vs_JourneyEnjoyment_Case{i}" for i in range(1, 3)],
         x=np.array([0.2, 1.0]), model="lin"),
    dict(n=3, cols=[f"S3_JourneyEnjoyment_vs_TimeImportance_Case{i}" for i in range(1, 6)],
         x=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), model="sig_dec"),
    dict(n=4, cols=[f"S4_TimeImportance_vs_AltModalChoices_Case{i}" for i in range(1, 6)],
         x=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), model="poly"),
    dict(n=5, cols=[f"S5_AltModalChoices_vs_CarTrips_Case{i}" for i in range(1, 5)],
         x=np.array([0.2, 0.5, 0.7, 1.0]), model="sig_inc"),
    dict(n=6, cols=[f"S6_CarTrips_vs_BusUse_Case{i}" for i in range(1, 3)],
         x=np.array([0.2, 1.0]), model="lin"),
]

VARIANT_SHIFTS = {"mean": 0, "plus_sd": +1, "minus_sd": -1}


def _lin(x, a, b):
    return a * x + b

def _poly(x, a, b, c):
    return a * x**2 + b * x + c

def _sig(x, L, k, x0):
    return L / (1.0 + np.exp(k * (x - x0)))

_MODEL_FN = {"lin": _lin, "poly": _poly, "sig_dec": _sig, "sig_inc": _sig}


def _fit(x, y, model):
    y = np.clip(y, 0.01, 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "lin":
            p, _ = curve_fit(_lin, x, y, maxfev=20000)
            return _lin, p
        if model == "poly":
            p, _ = curve_fit(_poly, x, y, maxfev=20000)
            return _poly, p
        if model == "sig_dec":
            p, _ = curve_fit(_sig, x, y, p0=[max(y)*1.1, 4.0, 0.7],
                             bounds=([0, 0.01, 0], [200, 50, 2]), maxfev=20000)
            return _sig, p
        p, _ = curve_fit(_sig, x, y, p0=[max(y)*1.5, -2.0, 0.6],
                         bounds=([0, -50, 0], [300, -0.01, 2]), maxfev=20000)
        return _sig, p


def build_all_case_fits(survey_xlsx: str) -> Dict[int, Dict[str, Tuple]]:
    df = pd.read_excel(survey_xlsx, sheet_name="Responses")
    fits = {}
    for cdef in CASE_DEFS:
        vals = df[cdef["cols"]].values.astype(float)
        means = np.nanmean(vals, axis=0)
        sds = np.nanstd(vals, axis=0, ddof=1)
        case_fits = {}
        for tag, shift in VARIANT_SHIFTS.items():
            target = np.clip(means + shift * sds, 0.0, 100.0)
            fn, p = _fit(cdef["x"], target, cdef["model"])
            case_fits[tag] = (fn, p)
        fits[cdef["n"]] = case_fits
    return fits


def evaluate_single_combo(
    combo: Tuple[str, ...],
    fits: Dict[int, Dict[str, Tuple]],
    t01: np.ndarray,
) -> np.ndarray:
    fn1, p1 = fits[1][combo[0]]
    att = np.clip(fn1(t01, *p1), 0.0, 100.0)
    att_01 = att / 100.0

    fn2, p2 = fits[2][combo[1]]
    je = np.clip(fn2(att_01, *p2), 0.0, 100.0)
    je_01 = je / 100.0

    fn3, p3 = fits[3][combo[2]]
    octt_raw = np.clip(fn3(je_01, *p3), 0.0, 100.0)
    octt = octt_raw / 100.0

    return octt


def compute_factorial_average_octt(
    fits: Dict[int, Dict[str, Tuple]],
    traveltime_h: np.ndarray,
) -> np.ndarray:
    t = np.asarray(traveltime_h, dtype=float) * 60.0
    t01 = np.clip(t / 10.0, 0.0, 1.0)

    variant_tags = list(VARIANT_SHIFTS.keys())
    combos = list(itertools.product(variant_tags, repeat=6))
    n_combos = len(combos)

    print(f"[FACTORIAL] Evaluating {n_combos} combinations (3^6 = 729)")

    octt_sum = np.zeros_like(t01, dtype=float)
    for combo in combos:
        octt_sum += evaluate_single_combo(combo, fits, t01)

    octt_avg = octt_sum / n_combos

    pcts = lambda a: np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    print(f"[FACTORIAL] Averaged OCTT percentiles: {pcts(octt_avg)}")
    print(f"[FACTORIAL] OCTT range: [{octt_avg.min():.6f}, {octt_avg.max():.6f}]")

    return octt_avg


def run_factorial_average(
    parent_directory: str = "../../data",
    survey_xlsx: str | None = None,
    cfg: MultimodalConfig | None = None,
):
    if cfg is None:
        cfg = MultimodalConfig()
    if survey_xlsx is None:
        survey_xlsx = cfg.survey_xlsx or "../../data/inputs/FF_Survey_responses.xlsx"

    tag = "factorial_avg"
    plots_dir = os.path.join(parent_directory, "plots", "sd_simulations", tag)
    outputs_dir = os.path.join(parent_directory, "outputs", "sd_simulations", tag)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    nodes_csv = fc.input_path(parent_directory, fc.NODES_CSV)
    edges_car = load_edges(parent_directory)
    origin_destination, demand_total = load_filtered_od_and_demand(parent_directory, cfg.tol)
    edges_bus, edges_bike = _load_mode_layers(parent_directory)
    edges_bus, edges_bike = _align_mode_layers(edges_car, edges_bus, edges_bike)

    graph_car = _make_car_graph(edges_car, cfg)
    run_cfg = _build_run_config(parent_directory, cfg)

    print("=== Stage 1: car UE on full demand ===")
    flows_car_1, _ = _run_single_ue(
        graph=graph_car,
        origin_destination=origin_destination,
        demand=demand_total,
        run_cfg=run_cfg,
        parent_directory=parent_directory,
        tag="DAY_car_stage1_factorial",
        use_cache=True,
        overwrite_cache=False,
    )

    _plot_flow(
        graph_car, origin_destination, demand_total, flows_car_1, nodes_csv,
        os.path.join(plots_dir, "fw_flow_car_stage1.png"),
    )

    traveltime_car_1 = bpr_flow(
        graph_car.free_flow_travel_h, flows_car_1,
        graph_car.capacity, graph_car.bpr_params,
    )

    print("\n=== Building 3^6 factorial OCTT average ===")
    fits = build_all_case_fits(survey_xlsx)
    octt_edge = compute_factorial_average_octt(fits, traveltime_car_1)

    od_octt_mean = _compute_od_mean_edge_cost(
        graph=graph_car,
        origin_destination=origin_destination,
        edge_cost=octt_edge,
    )

    ct_car, bu_people, att_people = _split_od_demand_from_octt(
        demand_total=demand_total,
        od_octt=od_octt_mean,
        cfg=cfg,
    )

    _save_od_split_csv(
        os.path.join(outputs_dir, f"od_split_{tag}.csv"),
        origin_destination, demand_total, ct_car, bu_people, att_people, od_octt_mean,
    )

    total = float(demand_total.sum())
    ct_sum = float(ct_car.sum())
    bu_sum = float(bu_people.sum())
    att_sum = float(att_people.sum())
    pct = lambda v: 100.0 * v / total if total > 0 else 0.0
    print(f"\n[{tag}] Mode split:")
    print(f"  Total demand: {total:.0f}")
    print(f"  Car CT:       {ct_sum:.0f} ({pct(ct_sum):.1f}%)")
    print(f"  Bus BU:       {bu_sum:.0f} ({pct(bu_sum):.1f}%)")
    print(f"  Active ATT:   {att_sum:.0f} ({pct(att_sum):.1f}%)")

    print("\n=== Stage 2: UE per mode ===")

    graph_car_2 = _make_car_graph(edges_car, cfg)
    flows_car_2, _ = _run_single_ue(
        graph=graph_car_2,
        origin_destination=origin_destination,
        demand=ct_car,
        run_cfg=run_cfg,
        parent_directory=parent_directory,
        tag=f"DAY_car_stage2_{tag}",
        use_cache=False, overwrite_cache=False,
    )
    _plot_flow(
        graph_car_2, origin_destination, ct_car, flows_car_2, nodes_csv,
        os.path.join(plots_dir, f"fw_flow_car_stage2_{tag}.png"),
    )
    plot_car_stage1_vs_stage2(
        graph=graph_car_2,
        flows_stage1=flows_car_1,
        flows_stage2=flows_car_2,
        nodes_csv=nodes_csv,
        out_png=os.path.join(plots_dir, f"car_stage1_vs_stage2_{tag}.png"),
    )

    graph_bus = _make_bus_graph(edges_car, edges_bus, cfg)
    bus_veh = bu_people / max(cfg.bus_occupancy, 1e-12)
    flows_bus, _ = _run_single_ue(
        graph=graph_bus,
        origin_destination=origin_destination,
        demand=bus_veh,
        run_cfg=run_cfg,
        parent_directory=parent_directory,
        tag=f"DAY_bus_stage2_{tag}",
        use_cache=False, overwrite_cache=False,
    )
    _plot_flow(
        graph_bus, origin_destination, bus_veh, flows_bus, nodes_csv,
        os.path.join(plots_dir, f"fw_flow_bus_stage2_{tag}.png"),
    )

    graph_bike = _make_bike_graph(edges_car, edges_bus, edges_bike, cfg)
    flows_bike, _ = _run_single_ue(
        graph=graph_bike,
        origin_destination=origin_destination,
        demand=att_people,
        run_cfg=run_cfg,
        parent_directory=parent_directory,
        tag=f"DAY_bike_stage2_{tag}",
        use_cache=False, overwrite_cache=False,
    )
    _plot_flow(
        graph_bike, origin_destination, att_people, flows_bike, nodes_csv,
        os.path.join(plots_dir, f"fw_flow_bike_stage2_{tag}.png"),
    )

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

    np.savez(
        os.path.join(outputs_dir, f"ue_results_{tag}.npz"),
        demand_total=demand_total,
        od_octt_mean=od_octt_mean,
        octt_edge=octt_edge,
        ct_car=ct_car,
        bu_people=bu_people,
        att_people=att_people,
        flows_car_stage1=flows_car_1,
        flows_car_stage2=flows_car_2,
        flows_bus_stage2=flows_bus,
        flows_bike_stage2=flows_bike,
    )
    print(f"\n[{tag}] All outputs saved to {outputs_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Full-factorial 3^6 averaged OCTT → 2-stage FW"
    )
    parser.add_argument("--data-dir", default="../../data")
    parser.add_argument("--survey-xlsx", default=None)
    parser.add_argument("--step-limit", type=int, default=None)
    args = parser.parse_args()

    cfg = MultimodalConfig()
    if args.step_limit is not None:
        cfg = MultimodalConfig(step_limit=args.step_limit)

    run_factorial_average(
        parent_directory=args.data_dir,
        survey_xlsx=args.survey_xlsx,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
