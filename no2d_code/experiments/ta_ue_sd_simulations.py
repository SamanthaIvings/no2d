"""
ta_ue_sd_simulations.py
=======================
Run the multimodal UE pipeline three times — once per survey regression
variant — to propagate inter-respondent uncertainty (±1 SD) through the
full OCTT → mode-split → assignment chain.

Variants:
  1. "mean"     — best-fit regression on column means  (baseline)
  2. "plus_sd"  — regression on mean + 1 SD per column (upper bound)
  3. "minus_sd" — regression on mean − 1 SD per column (lower bound)
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from no2d_code.core import filepath_configs as fc
from no2d_code.solver.IO_operations import load_edges, load_filtered_od_and_demand
from no2d_code.solver.bpr import bpr_flow
from no2d_code.core.survey_octt_mapping import SurveyOCTTPipeline
from no2d_code.core.ta_ue_multimodal_extension import (
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
from no2d_code.visualisation.fw_stages_comparison import plot_car_stage1_vs_stage2

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SURVEY DATA → REGRESSION VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Column groups and x-values per case  (matches the report exactly)
CASE_DEFS: List[dict] = [
    {
        "case": 1,
        "cols": [f"S1_Congestion_vs_ActiveTransport_Case{i}" for i in range(1, 6)],
        "x": np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        "model": "polynomial",
    },
    {
        "case": 2,
        "cols": [f"S2_ActiveTrips_vs_JourneyEnjoyment_Case{i}" for i in range(1, 3)],
        "x": np.array([0.2, 1.0]),
        "model": "linear",
    },
    {
        "case": 3,
        "cols": [f"S3_JourneyEnjoyment_vs_TimeImportance_Case{i}" for i in range(1, 6)],
        "x": np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        "model": "inv_sigmoid",
    },
    {
        "case": 4,
        "cols": [f"S4_TimeImportance_vs_AltModalChoices_Case{i}" for i in range(1, 6)],
        "x": np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        "model": "polynomial",
    },
    {
        "case": 5,
        "cols": [f"S5_AltModalChoices_vs_CarTrips_Case{i}" for i in range(1, 5)],
        "x": np.array([0.2, 0.5, 0.7, 1.0]),
        "model": "inv_sigmoid",
    },
    {
        "case": 6,
        "cols": [f"S6_CarTrips_vs_BusUse_Case{i}" for i in range(1, 3)],
        "x": np.array([0.2, 1.0]),
        "model": "linear",
    },
]


# ── candidate model functions ───────────────────────────────────────────────

def _linear(x, a, b):
    return a * x + b

def _polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def _inv_sigmoid(x, L, k, x0):
    return L / (1.0 + np.exp(k * (x - x0)))


_MODEL_FN = {
    "linear": (_linear, 2),
    "polynomial": (_polynomial, 3),
    "inv_sigmoid": (_inv_sigmoid, 3),
}


def _fit_case(x: np.ndarray, y: np.ndarray, model: str) -> Tuple[str, np.ndarray, float]:
    """Fit one case with proper initial guesses, return (model_name, params, R²)."""
    fn, _nparams = _MODEL_FN[model]
    y_clip = np.clip(y, 0.01, 100.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if model == "inv_sigmoid":
                # Detect direction: decreasing (Case 3) vs increasing (Case 5)
                if y_clip[-1] < y_clip[0]:
                    # Decreasing sigmoid: k > 0
                    p0 = [max(y_clip) * 1.1, 4.0, 0.7]
                    bounds = ([0, 0.01, 0.0], [200, 50, 2.0])
                else:
                    # Increasing sigmoid: k < 0
                    p0 = [max(y_clip) * 1.5, -2.0, 0.6]
                    bounds = ([0, -50, 0.0], [300, -0.01, 2.0])
                popt, _ = curve_fit(fn, x, y_clip, p0=p0, bounds=bounds, maxfev=20_000)
            else:
                popt, _ = curve_fit(fn, x, y_clip, maxfev=20_000)
        except RuntimeError:
            popt = np.zeros(_nparams)

    y_pred = fn(x, *popt)
    ss_res = np.sum((y_clip - y_pred) ** 2)
    ss_tot = np.sum((y_clip - np.mean(y_clip)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15) if ss_tot > 1e-10 else 1.0

    return model, popt, r2


@dataclass
class VariantParams:
    """Fitted parameters for all 6 cases under one SD variant."""
    tag: str                          # "mean" | "plus_sd" | "minus_sd"
    case_params: Dict[int, Tuple[str, np.ndarray, float]]  # case# → (model, popt, R²)


def build_variants(survey_xlsx: str) -> List[VariantParams]:
    """Read survey xlsx, compute 3 target sets, fit regressions."""
    df = pd.read_excel(survey_xlsx, sheet_name="Responses")

    variants: List[VariantParams] = []
    for tag, shift_sign in [("mean", 0), ("plus_sd", +1), ("minus_sd", -1)]:
        case_params = {}
        for cdef in CASE_DEFS:
            vals = df[cdef["cols"]].values.astype(float)
            means = np.nanmean(vals, axis=0)
            sds = np.nanstd(vals, axis=0, ddof=1)
            target = np.clip(means + shift_sign * sds, 0.0, 100.0)

            model, popt, r2 = _fit_case(cdef["x"], target, cdef["model"])
            case_params[cdef["case"]] = (model, popt, r2)

        variants.append(VariantParams(tag=tag, case_params=case_params))

    return variants


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PATCHED OCTT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class PatchedSurveyOCTTPipeline:
    """
    Wraps SurveyOCTTPipeline but overrides the fitted parameters with
    those from a specific SD variant.  Only Cases 1–3 feed into the
    OCTT computation chain; Cases 4–6 affect downstream mode split.
    """

    def __init__(self, base_pipeline: SurveyOCTTPipeline, variant: VariantParams):
        self._base = base_pipeline
        self._variant = variant

    def octt_from_traveltime(self, traveltime_h: np.ndarray, *, debug: bool = False) -> np.ndarray:
        """
        Replicate the survey pipeline chain using variant-specific parameters.

        Chain:  traveltime_h → t01 → Case1(ATT) → Case2(JE) → Case3(OCTT)

        Cases 1-3 use the report's normalised 0–100 scale; the final OCTT
        is rescaled to [0, 1] to match the downstream expectation.
        """
        t = np.asarray(traveltime_h, dtype=float) * 60.0  # hours → minutes
        t01 = np.clip(t / 10.0, 0.0, 1.0)                 # normalise to [0,1]

        # Case 1: Congestion (t01) → Active Transport Trips
        c1_model, c1_p, _ = self._variant.case_params[1]
        att = _apply_model(c1_model, c1_p, t01)

        # Normalise ATT to [0,1] for input to Case 2
        att_01 = np.clip(att / 100.0, 0.0, 1.0)

        # Case 2: ATT → Journey Enjoyment
        c2_model, c2_p, _ = self._variant.case_params[2]
        je = _apply_model(c2_model, c2_p, att_01)

        # Normalise JE to [0,1] for input to Case 3
        je_01 = np.clip(je / 100.0, 0.0, 1.0)

        # Case 3: JE → OCTT  (inverse sigmoid: high JE → low OCTT)
        c3_model, c3_p, _ = self._variant.case_params[3]
        octt_raw = _apply_model(c3_model, c3_p, je_01)

        # Scale from [0, 100] to [0, 1]
        octt = np.clip(octt_raw / 100.0, 0.0, 1.0)

        if debug:
            pcts = lambda a: np.percentile(a, [0, 5, 50, 95, 100])
            print(f"[{self._variant.tag}] t01:  {pcts(t01)}")
            print(f"[{self._variant.tag}] ATT:  {pcts(att)}")
            print(f"[{self._variant.tag}] JE:   {pcts(je)}")
            print(f"[{self._variant.tag}] OCTT: {pcts(octt)}")

        return octt

    def print_report(self):
        tag = self._variant.tag
        print(f"\n{'='*60}")
        print(f"  SD Variant: {tag}")
        print(f"{'='*60}")
        for case_num in sorted(self._variant.case_params):
            model, popt, r2 = self._variant.case_params[case_num]
            param_str = ", ".join(f"{v:.4f}" for v in popt)
            print(f"  Case {case_num}: {model:15s}  R²={r2:.4f}  params=[{param_str}]")
        print()


def _apply_model(model: str, params: np.ndarray, x: np.ndarray) -> np.ndarray:
    fn, _ = _MODEL_FN[model]
    return fn(x, *params)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MAIN SIMULATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_sd_simulations(
    parent_directory: str = str(fc.DATA_DIR),
    survey_xlsx: str | None = None,
    cfg: MultimodalConfig | None = None,
):
    if cfg is None:
        cfg = MultimodalConfig()
    if survey_xlsx is None:
        survey_xlsx = cfg.survey_xlsx or str(fc.data_path("inputs", "FF_Survey_responses.xlsx"))

    out_root = os.path.join(parent_directory, "plots", "sd_simulations")
    outputs_dir = os.path.join(parent_directory, "outputs", "sd_simulations")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # ── load network once ────────────────────────────────────────────────────
    nodes_csv = fc.input_path(parent_directory, fc.NODES_CSV)
    edges_car = load_edges(parent_directory)
    origin_destination, demand_total = load_filtered_od_and_demand(parent_directory, cfg.tol)
    edges_bus, edges_bike = _load_mode_layers(parent_directory)
    edges_bus, edges_bike = _align_mode_layers(edges_car, edges_bus, edges_bike)

    graph_car = _make_car_graph(edges_car, cfg)
    run_cfg = _build_run_config(parent_directory, cfg)

    # ── Stage 1 car UE (shared across variants) ─────────────────────────────
    print("=== Stage 1: car UE on full demand (shared) ===")
    flows_car_1, res_car_1 = _run_single_ue(
        graph=graph_car,
        origin_destination=origin_destination,
        demand=demand_total,
        run_cfg=run_cfg,
        parent_directory=parent_directory,
        tag="DAY_car_stage1_shared",
        use_cache=True,
        overwrite_cache=False,
    )
    traveltime_car_1 = bpr_flow(
        graph_car.free_flow_travel_h, flows_car_1,
        graph_car.capacity, graph_car.bpr_params,
    )

    # ── build 3 regression variants ──────────────────────────────────────────
    print("\n=== Building regression variants from survey data ===")
    base_pipeline = SurveyOCTTPipeline(survey_xlsx)
    variants = build_variants(survey_xlsx)

    for var in variants:
        PatchedSurveyOCTTPipeline(base_pipeline, var).print_report()

    # ── run each variant ─────────────────────────────────────────────────────
    for var in variants:
        tag = var.tag
        var_plots = os.path.join(out_root, tag)
        var_outputs = os.path.join(outputs_dir, tag)
        os.makedirs(var_plots, exist_ok=True)
        os.makedirs(var_outputs, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"#  VARIANT: {tag}")
        print(f"{'#'*70}")

        # OCTT mapping with variant-specific regressions
        pipe = PatchedSurveyOCTTPipeline(base_pipeline, var)
        octt_edge = pipe.octt_from_traveltime(traveltime_car_1, debug=True)

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

        # save split CSV
        _save_od_split_csv(
            os.path.join(var_outputs, f"od_split_{tag}.csv"),
            origin_destination, demand_total, ct_car, bu_people, att_people, od_octt_mean,
        )

        print(f"[{tag}] totals: total={demand_total.sum():.1f}, "
              f"car={ct_car.sum():.1f}, bus={bu_people.sum():.1f}, "
              f"active={att_people.sum():.1f}")

        # ── Stage 2: per-mode UE ────────────────────────────────────────────
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
            os.path.join(var_plots, f"fw_flow_car_stage2_{tag}.png"),
        )

        plot_car_stage1_vs_stage2(
            graph=graph_car_2,
            flows_stage1=flows_car_1,
            flows_stage2=flows_car_2,
            nodes_csv=nodes_csv,
            out_png=os.path.join(var_plots, f"car_stage1_vs_stage2_{tag}.png"),
        )

        # Bus
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
            os.path.join(var_plots, f"fw_flow_bus_stage2_{tag}.png"),
        )

        # Bike
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
            os.path.join(var_plots, f"fw_flow_bike_stage2_{tag}.png"),
        )

        # Report
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

        # Save arrays
        np.savez(
            os.path.join(var_outputs, f"ue_results_{tag}.npz"),
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
        print(f"[{tag}] saved to {var_outputs}/")

    # ── comparison summary ───────────────────────────────────────────────────
    _print_comparison_summary(outputs_dir, variants)


def _print_comparison_summary(outputs_dir: str, variants: List[VariantParams]):
    """Print a table comparing mode splits across variants."""
    print(f"\n{'='*70}")
    print("  SD SIMULATION COMPARISON")
    print(f"{'='*70}")
    print(f"{'Variant':<12} {'Car CT':>12} {'Bus BU':>12} {'Active ATT':>12} {'Car %':>8} {'Bus %':>8} {'ATT %':>8}")
    print("-" * 70)

    for var in variants:
        npz_path = os.path.join(outputs_dir, var.tag, f"ue_results_{var.tag}.npz")
        if not os.path.exists(npz_path):
            continue
        d = np.load(npz_path)
        total = float(d["demand_total"].sum())
        ct = float(d["ct_car"].sum())
        bu = float(d["bu_people"].sum())
        att = float(d["att_people"].sum())
        pct = lambda v: 100.0 * v / total if total > 0 else 0.0
        print(f"{var.tag:<12} {ct:>12.1f} {bu:>12.1f} {att:>12.1f} "
              f"{pct(ct):>7.1f}% {pct(bu):>7.1f}% {pct(att):>7.1f}%")

    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run multimodal UE with ±1 SD survey regression variants"
    )
    parser.add_argument("--data-dir", default=str(fc.DATA_DIR),
                        help="Parent data directory")
    parser.add_argument("--survey-xlsx", default=None,
                        help="Path to FF_Survey_responses.xlsx")
    args = parser.parse_args()

    run_sd_simulations(
        parent_directory=args.data_dir,
        survey_xlsx=args.survey_xlsx,
    )


if __name__ == "__main__":
    main()