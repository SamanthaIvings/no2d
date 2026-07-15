from __future__ import annotations

import argparse
import os
import time
from typing import Callable, Dict

import matplotlib as mpl
import numpy as np
import pandas as pd

from no2d_code.solver.IO_operations import load_edges, load_filtered_od_and_demand
from no2d_code.solver.bpr import bpr_flow
from no2d_code.core.octt_mapping import octt_from_traveltime as legacy_octt
from no2d_code.core.survey_octt_mapping import SurveyOCTTPipeline
from no2d_code.core.ta_ue_multimodal_extension import (
    MultimodalConfig,
    _make_car_graph,
    _build_run_config,
    _run_single_ue,
    _compute_od_mean_edge_cost,
    _split_od_demand_from_octt,
)
from no2d_code.core import filepath_configs as fc
from no2d_code.experiments.ta_ue_sd_simulations import (
    build_variants,
    PatchedSurveyOCTTPipeline,
)

mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
})


def _flush_csv(history, csv_path):
    pd.DataFrame(history).to_csv(csv_path, index=False)


def run_iterative_equilibrium(
    parent_directory: str = str(fc.DATA_DIR),
    survey_xlsx: str | None = None,
    cfg: MultimodalConfig | None = None,
    max_outer: int = 15,
    tol: float = 0.002,
    timeout_per_scenario: float = 3600.0,
):
    if cfg is None:
        cfg = MultimodalConfig()
    if survey_xlsx is None:
        survey_xlsx = cfg.survey_xlsx or str(fc.data_path("inputs", "FF_Survey_responses.xlsx"))

    plots_root = os.path.join(parent_directory, "plots", "iterative_eq")
    outputs_root = os.path.join(parent_directory, "outputs", "iterative_eq")
    os.makedirs(plots_root, exist_ok=True)
    os.makedirs(outputs_root, exist_ok=True)

    edges_car = load_edges(parent_directory)
    origin_destination, demand_total = load_filtered_od_and_demand(parent_directory, cfg.tol)
    run_cfg = _build_run_config(parent_directory, cfg)

    print("=== Stage 1: car UE on full demand (shared) ===")
    graph_s1 = _make_car_graph(edges_car, cfg)
    flows_car_s1, _ = _run_single_ue(
        graph=graph_s1,
        origin_destination=origin_destination,
        demand=demand_total,
        run_cfg=run_cfg,
    )
    tt_s1 = bpr_flow(graph_s1.free_flow_travel_h, flows_car_s1, graph_s1.capacity, graph_s1.bpr_params)

    base_pipeline = SurveyOCTTPipeline(survey_xlsx)
    sd_variants = build_variants(survey_xlsx)

    scenarios: Dict[str, Callable] = {}
    scenarios["baseline"] = lambda tt: legacy_octt(tt, debug=False)
    for var in sd_variants:
        pipe = PatchedSurveyOCTTPipeline(base_pipeline, var)
        scenarios[var.tag] = pipe.octt_from_traveltime

    all_histories: Dict[str, pd.DataFrame] = {}
    aborted = False

    for sc_name, octt_fn in scenarios.items():
        if aborted:
            break

        sc_plots = os.path.join(plots_root, sc_name)
        sc_outputs = os.path.join(outputs_root, sc_name)
        os.makedirs(sc_plots, exist_ok=True)
        os.makedirs(sc_outputs, exist_ok=True)

        csv_path = os.path.join(sc_outputs, f"history_{sc_name}.csv")

        print(f"\n{'#'*70}")
        print(f"#  SCENARIO: {sc_name}  (max {max_outer} iters, tol={tol}, timeout={timeout_per_scenario:.0f}s)")
        print(f"{'#'*70}")

        history = []
        tt_current = tt_s1.copy()
        t_start = time.time()

        try:
            for outer in range(max_outer):
                elapsed = time.time() - t_start
                if elapsed > timeout_per_scenario:
                    print(f"  [{sc_name}] TIMEOUT after {elapsed:.0f}s at iter {outer}")
                    break

                octt_edge = octt_fn(tt_current)

                od_octt = _compute_od_mean_edge_cost(
                    graph=graph_s1,
                    origin_destination=origin_destination,
                    edge_cost=octt_edge,
                )

                ct_car, bu_people, att_people = _split_od_demand_from_octt(
                    demand_total=demand_total,
                    od_octt=od_octt,
                    cfg=cfg,
                )

                total = float(demand_total.sum())
                car_sum = float(ct_car.sum())
                bus_sum = float(bu_people.sum())
                att_sum = float(att_people.sum())
                car_pct = car_sum / total
                bus_pct = bus_sum / total
                att_pct = att_sum / total

                octt_med = float(np.median(octt_edge))
                octt_p05 = float(np.percentile(octt_edge, 5))
                octt_p95 = float(np.percentile(octt_edge, 95))

                history.append({
                    "outer_iter": outer,
                    "car_demand": car_sum,
                    "bus_demand": bus_sum,
                    "att_demand": att_sum,
                    "car_pct": car_pct,
                    "bus_pct": bus_pct,
                    "att_pct": att_pct,
                    "octt_median": octt_med,
                    "octt_p05": octt_p05,
                    "octt_p95": octt_p95,
                    "elapsed_s": elapsed,
                })

                _flush_csv(history, csv_path)

                delta_str = ""
                if outer > 0:
                    delta = abs(car_pct - history[-2]["car_pct"])
                    delta_str = f"  delta_car={delta:.5f}"

                print(f"  [{sc_name}] iter={outer}  "
                      f"car={car_pct:.4f}  bus={bus_pct:.4f}  att={att_pct:.4f}  "
                      f"OCTT_med={octt_med:.4f}  t={elapsed:.0f}s{delta_str}")

                if outer > 0 and abs(car_pct - history[-2]["car_pct"]) < tol:
                    print(f"  [{sc_name}] CONVERGED at outer iter {outer}  ({elapsed:.0f}s)")
                    break

                graph_car_k = _make_car_graph(edges_car, cfg)
                flows_car_k, _ = _run_single_ue(
                    graph=graph_car_k,
                    origin_destination=origin_destination,
                    demand=ct_car,
                    run_cfg=run_cfg,
                )

                tt_current = bpr_flow(
                    graph_car_k.free_flow_travel_h, flows_car_k,
                    graph_car_k.capacity, graph_car_k.bpr_params,
                )
            else:
                print(f"  [{sc_name}] max iters ({max_outer}) reached  ({time.time()-t_start:.0f}s)")

        except KeyboardInterrupt:
            print(f"\n  [{sc_name}] INTERRUPTED by user — saving partial results")
            _flush_csv(history, csv_path)
            aborted = True

        if history:
            _flush_csv(history, csv_path)
            all_histories[sc_name] = pd.DataFrame(history)
            print(f"  [{sc_name}] saved {csv_path}  ({len(history)} iters)")

    plot_all_from_disk(plots_root, outputs_root)
    _print_final_summary(all_histories)

    if aborted:
        print("\n  Run was interrupted. Re-plot at any time with:")
        print(f"    python plot_iterative_eq.py --data-dir {parent_directory}\n")


def plot_all_from_disk(plots_root: str, outputs_root: str):
    all_histories = {}
    for sc_name in os.listdir(outputs_root):
        csv_path = os.path.join(outputs_root, sc_name, f"history_{sc_name}.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                all_histories[sc_name] = df
                sc_plots = os.path.join(plots_root, sc_name)
                os.makedirs(sc_plots, exist_ok=True)
                _plot_convergence(df, sc_name, sc_plots)

    if all_histories:
        _plot_all_scenarios(all_histories, plots_root)


def _plot_convergence(df: pd.DataFrame, sc_name: str, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    iters = df["outer_iter"].values

    ax1.plot(iters, df["car_pct"] * 100, "o-", color="#2E5090", label="Car", lw=2)
    ax1.plot(iters, df["bus_pct"] * 100, "s-", color="#C04040", label="Bus", lw=1.5)
    ax1.plot(iters, df["att_pct"] * 100, "^-", color="#2E8B57", label="Active", lw=2)
    ax1.set_xlabel("Outer iteration")
    ax1.set_ylabel("Mode share (%)")
    ax1.set_title(f"{sc_name} — mode share convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(iters, df["octt_p05"], df["octt_p95"], alpha=0.2, color="#2E5090")
    ax2.plot(iters, df["octt_median"], "o-", color="#2E5090", lw=2, label="OCTT median")
    ax2.set_xlabel("Outer iteration")
    ax2.set_ylabel("OCTT")
    ax2.set_title(f"{sc_name} — OCTT distribution per iter")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    png = os.path.join(out_dir, f"convergence_{sc_name}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{sc_name}] plot saved: {png}")


def _plot_all_scenarios(histories: Dict[str, pd.DataFrame], out_dir: str):
    colors = {"baseline": "#555555", "mean": "#2E5090", "plus_sd": "#C04040", "minus_sd": "#2E8B57"}
    styles = {"baseline": "-", "mean": "-", "plus_sd": "--", "minus_sd": "--"}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for sc_name, df in histories.items():
        c = colors.get(sc_name, "#000000")
        ls = styles.get(sc_name, "-")
        ax.plot(df["outer_iter"], df["car_pct"] * 100, f"o{ls}",
                color=c, lw=2, markersize=5, label=sc_name)

    ax.set_xlabel("Outer iteration", fontsize=12)
    ax.set_ylabel("Car mode share (%)", fontsize=12)
    ax.set_title("Iterative mode-choice equilibrium — car share convergence",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png = os.path.join(out_dir, "all_scenarios_car_convergence.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined plot: {png}")


def _print_final_summary(histories: Dict[str, pd.DataFrame]):
    if not histories:
        print("\n  No results to summarise.\n")
        return

    print(f"\n{'='*80}")
    print("  ITERATIVE EQUILIBRIUM — FINAL MODE SPLITS (last completed iter)")
    print(f"{'='*80}")
    print(f"{'Scenario':<12} {'Iters':>6} {'Car %':>8} {'Bus %':>8} {'ATT %':>8} "
          f"{'OCTT med':>10} {'Time':>8}")
    print("-" * 80)

    for sc_name, df in histories.items():
        last = df.iloc[-1]
        n = int(last["outer_iter"]) + 1
        t = last.get("elapsed_s", 0)
        print(f"{sc_name:<12} {n:>6} {last['car_pct']*100:>7.2f}% {last['bus_pct']*100:>7.2f}% "
              f"{last['att_pct']*100:>7.2f}% {last['octt_median']:>10.4f} {t:>7.0f}s")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Iterative mode-choice equilibrium for 4 OCTT scenarios"
    )
    parser.add_argument("--data-dir", default=str(fc.DATA_DIR))
    parser.add_argument("--survey-xlsx", default=None)
    parser.add_argument("--max-outer", type=int, default=15)
    parser.add_argument("--tol", type=float, default=0.002)
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="Wall-clock timeout per scenario in seconds (default 3600)")
    args = parser.parse_args()

    run_iterative_equilibrium(
        parent_directory=args.data_dir,
        survey_xlsx=args.survey_xlsx,
        max_outer=args.max_outer,
        tol=args.tol,
        timeout_per_scenario=args.timeout,
    )


if __name__ == "__main__":
    main()