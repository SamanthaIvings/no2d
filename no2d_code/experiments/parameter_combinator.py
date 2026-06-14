"""
Combinatorial parameter sweep for OCTT mapping functions.
==========================================================

Sweeps control parameters for the three-stage mapping pipeline:

    BPR traveltime  ──►  ATT  ──►  JE  ──►  OCTT  ──►  (ATT_people, BU_people)
The Frank-Wolfe UE solver is run **once** (or loaded from cache).
All 18,150 parameter combinations are evaluated as pure post-processing
on the fixed AoN and UE travel times — no re-solving needed.

Parameter grids (from pink stickers):
─────────────────────────────────────
  Stage 1 — ATT = b_att · t01²
      b_att ∈ (0.0, 1.0],  step 0.1   →  10 values

  Stage 2 — JE = a_je · ATT
      a_je  ∈ [0.6, 2.0],  step 0.1   →  15 values

  Stage 3 — OCTT = 1 / (1 + exp(a_octt · (JE + b_octt)))
      a_octt ∈ [1, 10],    step 1     →  10 values
      b_octt ∈ [-1.0, 0.0], step 0.1  →  11 values

  Total combinations: 10 × 15 × 10 × 11 = 16500
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────
# 1.  PARAMETER GRID DEFINITIONS
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MappingParams:
    """One combination of the four sweep parameters."""
    b_att: float    # ATT coefficient
    a_je: float     # JE slope
    a_octt: float   # OCTT sigmoid steepness
    b_octt: float   # OCTT sigmoid shift

    @property
    def tag(self) -> str:
        """Short unique string for filenames and dict keys."""
        return (
            f"batt{self.b_att:.1f}_aje{self.a_je:.1f}"
            f"_aoctt{self.a_octt:.0f}_boctt{self.b_octt:.1f}"
        )


def build_parameter_grid() -> List[MappingParams]:
    """Build the full cartesian product of parameter values."""
    b_att_vals  = np.round(np.arange(0.1, 1.0 + 1e-9, 0.1), 1)   # 10
    a_je_vals   = np.round(np.arange(0.6, 2.0 + 1e-9, 0.1), 1)   # 15
    a_octt_vals = np.arange(1, 11, dtype=float)                          # 10
    b_octt_vals = np.round(np.arange(-1.0, 0.0 + 1e-9, 0.1), 1)  # 11

    grid = [
        MappingParams(b_att=b, a_je=a, a_octt=ao, b_octt=bo)
        for b, a, ao, bo in itertools.product(
            b_att_vals, a_je_vals, a_octt_vals, b_octt_vals
        )
    ]
    print(f"[GRID] Built {len(grid)} parameter combinations")
    return grid


# ─────────────────────────────────────────────────────────────
# 2.  PARAMETERISED MAPPING FUNCTIONS
# ─────────────────────────────────────────────────────────────

OCTT_TIME_SCALE = 60.0
OCTT_T_MIN_REF = 0.0
OCTT_T_MAX_REF = 10.0


def compute_t01(traveltime_h: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalise travel time to [0, 1] — shared across all combos."""
    t = np.asarray(traveltime_h, dtype=float) * OCTT_TIME_SCALE
    denom = OCTT_T_MAX_REF - OCTT_T_MIN_REF
    return np.clip((t - OCTT_T_MIN_REF) / denom, 0.0, 1.0)


def mapping_pipeline(
    t01: NDArray[np.float64],
    params: MappingParams,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Run the three-stage mapping for one parameter combination.

    Returns (att, je, octt) arrays — all same shape as t01.
    """
    # Stage 1: ATT = b_att · t01²
    att = params.b_att * (t01 ** 2)

    # Stage 2: JE = a_je · ATT
    je = params.a_je * att

    # Stage 3: OCTT = 1 / (1 + exp(a_octt · (JE + b_octt)))
    arg = params.a_octt * (je + params.b_octt)
    octt = 1.0 / (1.0 + np.exp(arg))

    return att, je, octt


def compute_att_bu_people(
    octt: NDArray[np.float64],
    total_demand: float | NDArray[np.float64] = 1.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Downstream: OCTT → AMC → mode split → ATT_people, BU_people.

    Uses the same constants as the existing octt_mapping.py so that
    the downstream pipeline stays consistent.
    """
    # Import existing constants to stay in sync
    AMC_A, AMC_B, AMC_OCTT_SCALE = 1.0, 0.1, 100.0
    CT_LOGIT_K, CT_LOGIT_SHIFT = 0.01, 500.0
    BUS_SHARE = 0.1

    dem = np.asarray(total_demand, dtype=float)

    amc = AMC_A + AMC_B * ((octt * AMC_OCTT_SCALE) ** 2)
    ct_share = 1.0 / (1.0 + np.exp(CT_LOGIT_K * (amc - CT_LOGIT_SHIFT)))

    ct = dem * ct_share
    bu = BUS_SHARE * dem - BUS_SHARE * ct
    att_people = dem - bu - ct

    return att_people, bu


# ─────────────────────────────────────────────────────────────
# 3.  SINGLE-COMBO EVALUATOR (summary metrics)
# ─────────────────────────────────────────────────────────────

@dataclass
class ComboResult:
    """Summary metrics for one parameter combination."""
    params: MappingParams

    # Deltas (UE − AoN)
    delta_octt_mean: float
    delta_octt_median: float
    delta_octt_std: float

    delta_att_mean: float
    delta_att_median: float

    delta_bu_mean: float
    delta_bu_median: float

    # Absolute UE values
    octt_ue_mean: float
    octt_ue_p05: float
    octt_ue_p95: float

    att_ue_mean: float
    bu_ue_mean: float


def evaluate_combo(
    t01_aon: NDArray[np.float64],
    t01_ue: NDArray[np.float64],
    params: MappingParams,
) -> ComboResult:
    """Evaluate one parameter combination on pre-computed t01 arrays."""
    att_aon, je_aon, octt_aon = mapping_pipeline(t01_aon, params)
    att_ue, je_ue, octt_ue = mapping_pipeline(t01_ue, params)

    att_ppl_aon, bu_aon = compute_att_bu_people(octt_aon)
    att_ppl_ue, bu_ue = compute_att_bu_people(octt_ue)

    d_octt = octt_ue - octt_aon
    d_att = att_ppl_ue - att_ppl_aon
    d_bu = bu_ue - bu_aon

    return ComboResult(
        params=params,
        delta_octt_mean=float(np.mean(d_octt)),
        delta_octt_median=float(np.median(d_octt)),
        delta_octt_std=float(np.std(d_octt)),
        delta_att_mean=float(np.mean(d_att)),
        delta_att_median=float(np.median(d_att)),
        delta_bu_mean=float(np.mean(d_bu)),
        delta_bu_median=float(np.median(d_bu)),
        octt_ue_mean=float(np.mean(octt_ue)),
        octt_ue_p05=float(np.percentile(octt_ue, 5)),
        octt_ue_p95=float(np.percentile(octt_ue, 95)),
        att_ue_mean=float(np.mean(att_ppl_ue)),
        bu_ue_mean=float(np.mean(bu_ue)),
    )


# ─────────────────────────────────────────────────────────────
# 4.  FULL SWEEP ENGINE
# ─────────────────────────────────────────────────────────────

def run_sweep(
    tt_aon_h: NDArray[np.float64],
    tt_ue_h: NDArray[np.float64],
    grid: List[MappingParams] | None = None,
) -> pd.DataFrame:
    """
    Run all parameter combinations and return a results DataFrame.

    Parameters
    ----------
    tt_aon_h : edge-level BPR travel times (hours) from AoN flow
    tt_ue_h  : edge-level BPR travel times (hours) from UE flow
    grid     : parameter grid (built automatically if None)

    Returns
    -------
    DataFrame with one row per combination and all summary metrics.
    """
    if grid is None:
        grid = build_parameter_grid()

    # Pre-compute t01 once — shared across all combos
    t01_aon = compute_t01(tt_aon_h)
    t01_ue = compute_t01(tt_ue_h)

    rows = []
    n = len(grid)
    for i, params in enumerate(grid):
        if i % 2000 == 0 or i == n - 1:
            print(f"[SWEEP] {i + 1}/{n} combinations evaluated")

        result = evaluate_combo(t01_aon, t01_ue, params)

        rows.append({
            "b_att": params.b_att,
            "a_je": params.a_je,
            "a_octt": params.a_octt,
            "b_octt": params.b_octt,
            "tag": params.tag,
            # deltas
            "delta_octt_mean": result.delta_octt_mean,
            "delta_octt_median": result.delta_octt_median,
            "delta_octt_std": result.delta_octt_std,
            "delta_att_mean": result.delta_att_mean,
            "delta_att_median": result.delta_att_median,
            "delta_bu_mean": result.delta_bu_mean,
            "delta_bu_median": result.delta_bu_median,
            # absolutes
            "octt_ue_mean": result.octt_ue_mean,
            "octt_ue_p05": result.octt_ue_p05,
            "octt_ue_p95": result.octt_ue_p95,
            "att_ue_mean": result.att_ue_mean,
            "bu_ue_mean": result.bu_ue_mean,
        })

    df = pd.DataFrame(rows)
    print(f"[SWEEP] Complete. {len(df)} rows.")
    return df


# ─────────────────────────────────────────────────────────────
# 5.  SAVE / LOAD RESULTS
# ─────────────────────────────────────────────────────────────

SWEEP_RESULTS_CSV = "sweep_results.csv"
SWEEP_RESULTS_PARQUET = "sweep_results.parquet"


def save_sweep_results(df: pd.DataFrame, outputs_dir: Path) -> Path:
    """Save to CSV (and Parquet if pyarrow is available)."""
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = outputs_dir / SWEEP_RESULTS_CSV
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] CSV:     {csv_path}")

    try:
        parquet_path = outputs_dir / SWEEP_RESULTS_PARQUET
        df.to_parquet(parquet_path, index=False)
        print(f"[SAVE] Parquet: {parquet_path}")
    except ImportError:
        print("[SAVE] Parquet skipped (install pyarrow for faster reload)")

    return csv_path


def load_sweep_results(outputs_dir: Path) -> pd.DataFrame:
    """Load previously saved sweep results (Parquet preferred, CSV fallback)."""
    parquet_path = outputs_dir / SWEEP_RESULTS_PARQUET
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            pass
    return pd.read_csv(outputs_dir / SWEEP_RESULTS_CSV)


# ─────────────────────────────────────────────────────────────
# 6.  PLOT GENERATION FOR SELECTED COMBOS
#     (reuses existing project plot functions)
# ─────────────────────────────────────────────────────────────

def generate_plots_for_combo(
    *,
    params: MappingParams,
    tt_aon_h: NDArray[np.float64],
    tt_ue_h: NDArray[np.float64],
    graph,                          # Digraph
    nodes_csv: Path,
    plots_dir: Path,
    plot_lsoa: bool = False,
    lsoa_polygons=None,
    edge_lsoa_map=None,
    lsoa_name_filter=None,
):
    """
    Generate the same spatial plots used in ta_ue.py for one combo.

    Produces:
      - ΔOCTT edge map
      - ΔATT edge map
      - ΔBU edge map
      - (optionally) LSOA-aggregated versions of each
    """
    # Lazy imports — these depend on the project visualisation module
    from no2d_code.visualisation.fw_flow_plotter import (
        plot_edge_value_comparison,
        plot_lsoa_value_comparison,
    )
    from no2d_code.visualisation.lsoa_metric_plotter import plot_lsoa_value_state

    tag = params.tag
    combo_dir = plots_dir / "sweep" / tag
    combo_dir.mkdir(parents=True, exist_ok=True)

    t01_aon = compute_t01(tt_aon_h)
    t01_ue = compute_t01(tt_ue_h)

    att_aon, _, octt_aon = mapping_pipeline(t01_aon, params)
    att_ue, _, octt_ue = mapping_pipeline(t01_ue, params)

    att_ppl_aon, bu_aon = compute_att_bu_people(octt_aon)
    att_ppl_ue, bu_ue = compute_att_bu_people(octt_ue)

    # ── OCTT ──
    plot_edge_value_comparison(
        graph=graph,
        value_init=octt_aon,
        value_final=octt_ue,
        nodes=nodes_csv,
        out_path=combo_dir / "delta_octt.pdf",
        title_init="AoN OCTT",
        title_final="UE OCTT",
        cbar_label="OCTT",
        delta_label="ΔOCTT",
        delta_only=True,
        cbar_numbers=False,
    )

    # ── ATT people ──
    plot_edge_value_comparison(
        graph=graph,
        value_init=att_ppl_aon,
        value_final=att_ppl_ue,
        nodes=nodes_csv,
        out_path=combo_dir / "delta_att.pdf",
        title_init="AoN ATT",
        title_final="UE ATT",
        cbar_label="ATT",
        delta_label="ΔATT",
        delta_only=True,
        cbar_numbers=False,
    )

    # ── BU ──
    plot_edge_value_comparison(
        graph=graph,
        value_init=bu_aon,
        value_final=bu_ue,
        nodes=nodes_csv,
        out_path=combo_dir / "delta_bu.pdf",
        title_init="AoN BU",
        title_final="UE BU",
        cbar_label="BU",
        delta_label="ΔBU",
        delta_only=True,
        cbar_numbers=False,
    )

    # ── LSOA plots (optional) ──
    if plot_lsoa and lsoa_polygons is not None and edge_lsoa_map is not None:
        for val, val_aon, label in [
            (octt_ue, octt_aon, "octt"),
            (att_ppl_ue, att_ppl_aon, "att"),
            (bu_ue, bu_aon, "bu"),
        ]:
            plot_lsoa_value_comparison(
                graph=graph,
                value_init=val_aon,
                value_final=val,
                nodes=nodes_csv,
                lsoa_polygons=lsoa_polygons,
                edge_lsoa_map=edge_lsoa_map,
                out_path=combo_dir / f"lsoa_delta_{label}.png",
                lsoa_name_filter=lsoa_name_filter,
                cbar_label=f"Change in {label.upper()}",
            )

            plot_lsoa_value_state(
                value=val,
                lsoa_polygons=lsoa_polygons,
                edge_lsoa_map=edge_lsoa_map,
                out_path=combo_dir / f"lsoa_{label}_ue.png",
                title=f"{label.upper()} after UE",
                cbar_label=label.upper(),
                agg="mean",
                lsoa_name_filter=lsoa_name_filter,
            )

    print(f"[PLOT] Saved plots for {tag} → {combo_dir}")


def select_interesting_combos(
    df: pd.DataFrame,
    metric: str = "delta_octt_mean",
    n_top: int = 5,
    n_bottom: int = 5,
) -> List[MappingParams]:
    """
    Pick combos that produce the largest and smallest change
    in a given metric — good candidates for spatial plot generation.
    """
    df_sorted = df.sort_values(metric)
    indices = (
        list(df_sorted.head(n_bottom).index) +
        list(df_sorted.tail(n_top).index)
    )
    return [
        MappingParams(
            b_att=df.loc[i, "b_att"],
            a_je=df.loc[i, "a_je"],
            a_octt=df.loc[i, "a_octt"],
            b_octt=df.loc[i, "b_octt"],
        )
        for i in indices
    ]


# ─────────────────────────────────────────────────────────────
# 7.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_recombination_sweep(
    *,
    parent_directory: str | None = None,
    tol: float = 100.0,
    eps: float = 1e-6,
    plot_top_n: int = 5,
    plot_lsoa: bool = False,
    lsoa_name_filter: Optional[List[str]] = None,
):
    """
    Full pipeline: load cached UE → sweep parameters → save → plot.

    Steps
    -----
    1. Load graph + cached UE flows (from ta_ue.py run)
    2. Compute BPR travel times for AoN and UE
    3. Build parameter grid (18,150 combos)
    4. Run sweep — vectorised NumPy, ~30 sec on laptop
    5. Save results (CSV + Parquet)
    6. Select most-interesting combos and generate spatial plots
    """
    from no2d_code.solver.IO_operations import (
        load_edges,
        load_filtered_od_and_demand,
        load_ue_cache_pickle,
        has_ue_cache_pickle,
    )
    from no2d_code.solver.digraph import Digraph
    from no2d_code.solver.bpr import bpr_flow
    from no2d_code.core import filepath_configs as fc

    if parent_directory is None:
        parent_directory = str(fc.DATA_DIR)

    parent_dir = Path(parent_directory)
    outputs_dir = parent_dir / "outputs"
    plots_dir = parent_dir / "plots"
    nodes_csv = Path(fc.input_path(parent_directory, fc.NODES_CSV))

    # ── 1. Load graph ──
    print("[LOAD] Loading graph and demand...")
    edges = load_edges(parent_directory)
    graph = Digraph.from_edges(edges, eps)
    origin_destination, demand = load_filtered_od_and_demand(parent_directory, tol)

    # ── 2. Load cached UE flows ──
    tag = "DAY"
    if not has_ue_cache_pickle(parent_directory, tag):
        raise RuntimeError(
            f"No UE cache found for tag='{tag}'. "
            "Run ta_ue.py first to produce the UE solution."
        )

    ue_flow, ue_flow_best, result, meta = load_ue_cache_pickle(parent_directory, tag)
    print(f"[LOAD] Loaded UE cache: {meta}")

    # ── 3. Compute BPR travel times (fixed) ──
    time = graph.free_flow_travel_h
    capacity = graph.capacity
    params_bpr = graph.bpr_params

    flow_zero = np.zeros(graph.u.size, dtype=float)
    tt_aon_h = bpr_flow(time, flow_zero, capacity, params_bpr)

    # For AoN: recompute from initial all-or-nothing flow
    # (using free-flow as proxy — or load the actual AoN flow if saved)
    tt_ue_h = bpr_flow(time, np.asarray(ue_flow), capacity, params_bpr)

    print(f"[TT] AoN travel time: mean={np.mean(tt_aon_h):.6f}h")
    print(f"[TT] UE  travel time: mean={np.mean(tt_ue_h):.6f}h")

    # ── 4. Run the sweep ──
    grid = build_parameter_grid()
    df = run_sweep(tt_aon_h, tt_ue_h, grid)

    # ── 5. Save results ──
    save_sweep_results(df, outputs_dir)

    # ── 6. Generate spatial plots for interesting combos ──
    interesting = select_interesting_combos(df, n_top=plot_top_n, n_bottom=plot_top_n)
    print(f"[PLOT] Generating spatial plots for {len(interesting)} combos...")

    for combo in interesting:
        generate_plots_for_combo(
            params=combo,
            tt_aon_h=tt_aon_h,
            tt_ue_h=tt_ue_h,
            graph=graph,
            nodes_csv=nodes_csv,
            plots_dir=plots_dir,
            plot_lsoa=plot_lsoa,
            lsoa_name_filter=lsoa_name_filter,
        )

    print("[DONE] Recombination sweep complete.")


if __name__ == "__main__":
    run_recombination_sweep(
        plot_top_n=5,
        plot_lsoa=False,
        lsoa_name_filter=["Barnsley", "Doncaster", "Rotherham", "Sheffield"],
    )