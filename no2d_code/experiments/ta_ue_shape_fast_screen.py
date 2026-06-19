from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from no2d_code.solver.IO_operations import load_edges, load_filtered_od_and_demand
from no2d_code.solver.bpr import bpr_flow
from no2d_code.core.ta_ue_multimodal_extension import (
    MultimodalConfig,
    _make_car_graph,
    _build_run_config,
    _run_single_ue,
    _compute_od_mean_edge_cost,
)
from no2d_code.core import filepath_configs as fc

from no2d_code.experiments.ta_ue_shape_hypothesis import (
    CaseForm,
    fit_case_form,
    SURVEY_MEANS,
)


@dataclass(frozen=True)
class CaseSpec:
    form: CaseForm
    flip: bool = False
    renorm: str = "unit"          # "unit" = score/100 (control); "range" = min-max to [0,1]


@dataclass
class ScenarioSpec:
    case1: CaseSpec
    case2: CaseSpec
    case3: CaseSpec
    case4: CaseSpec
    case5: CaseSpec
    bus_share: float = 0.10
    t_ref_max: float = 10.0       # congestion normalisation window (operating-point lever)
    tt_scale: float = 60.0

    def specs(self) -> List[CaseSpec]:
        return [self.case1, self.case2, self.case3, self.case4, self.case5]


@dataclass
class ScreenConfig:
    scenarios: Dict[str, ScenarioSpec]
    demand_fractions: Tuple[float, ...] = (0.15, 0.30, 0.45, 0.60, 0.80, 1.00)
    fw_step_limit: int = 10
    map_iters: int = 60           # scalar-map iterations (free; run long to see oscillation)
    s0: float = 0.60              # initial car share
    survey_means: Dict[int, Dict[str, List[float]]] = field(default_factory=lambda: SURVEY_MEANS)


def _build_case(spec: CaseSpec, means: Dict[int, Dict[str, List[float]]], case_id: int
                ) -> Callable[[float], float]:
    f = fit_case_form(spec.form, means[case_id]["x"], means[case_id]["y"])
    grid = np.linspace(0.0, 1.0, 201)
    raw = np.clip(f(grid) / 100.0, 0.0, 1.0)
    rmin, rmax = float(raw.min()), float(raw.max())
    span = max(rmax - rmin, 1e-9)

    def g(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        val = float(np.clip(f(u) / 100.0, 0.0, 1.0))
        if spec.renorm == "range":
            val = (val - rmin) / span        # -> [0, 1]
            if spec.flip:
                val = 1.0 - val
        else:  # "unit"
            if spec.flip:
                val = rmin + rmax - val       # reflect within realised band
        return float(np.clip(val, 0.0, 1.0))

    return g


class Chain:
    def __init__(self, spec: ScenarioSpec, means):
        self.spec = spec
        self.g = {i: _build_case(s, means, i) for i, s in zip(range(1, 6), spec.specs())}

    def car_share(self, tbar_h: float) -> float:
        denom = self.spec.t_ref_max
        x = float(np.clip((tbar_h * self.spec.tt_scale) / denom, 0.0, 1.0))
        att = self.g[1](x)
        je = self.g[2](att)
        octt = self.g[3](je)
        amc = self.g[4](octt)
        return self.g[5](amc)

    def loop_gain(self, tbar_at, h: float = 1e-4) -> Dict:
        s0 = 0.55
        f0 = self.car_share(tbar_at(s0))
        f1 = self.car_share(tbar_at(s0 + h))
        dshare_dfrac = (f1 - f0) / h          # behavioural sensitivity h' (renormalised)
        gain = dshare_dfrac                    # frac == car-share fraction here
        return {"h_prime": dshare_dfrac, "loop_sign": int(np.sign(gain)) or 0}


def build_response_surface(parent_directory: str, scfg: ScreenConfig
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cfg = MultimodalConfig(step_limit=scfg.fw_step_limit)
    edges_car = load_edges(parent_directory)
    od, demand_total = load_filtered_od_and_demand(parent_directory, cfg.tol)
    run_cfg = _build_run_config(parent_directory, cfg)
    total = float(demand_total.sum())

    fracs, tbars, tmeds = [], [], []
    for frac in scfg.demand_fractions:
        graph = _make_car_graph(edges_car, cfg)
        flows, _ = _run_single_ue(
            graph=graph,
            origin_destination=od,
            demand=demand_total * frac,
            run_cfg=run_cfg,
            parent_directory=parent_directory,
            tag=f"respsurf_frac{frac:.2f}",
            use_cache=True,
            overwrite_cache=False,
        )
        tt = bpr_flow(graph.free_flow_travel_h, flows, graph.capacity, graph.bpr_params)
        od_tt = _compute_od_mean_edge_cost(graph=graph, origin_destination=od, edge_cost=tt)
        fracs.append(float(frac))
        tbars.append(float(np.mean(od_tt)))
        tmeds.append(float(np.median(tt)))
        print(f"  [surface] frac={frac:.2f}  OD-mean tt={tbars[-1]:.5f} h  "
              f"edge-median tt={tmeds[-1]:.5f} h")

    return np.array(fracs), np.array(tbars), np.array(tmeds), total


def report_kappa(fracs, tbars):
    lo, hi = float(tbars.min()), float(tbars.max())
    rel = (hi - lo) / max(hi, 1e-12)
    print(f"\n  KAPPA GATE — OD-mean travel time vs car-demand fraction:")
    for fr, tb in zip(fracs, tbars):
        print(f"    frac={fr:.2f}  tbar={tb:.5f} h")
    print(f"    spread: {lo:.5f} .. {hi:.5f} h  (relative {rel * 100:.2f}%)")
    if rel < 0.02:
        print("    => kappa ~ 0: congestion barely responds to demand on this network.\n"
              "       No shape/sign config can oscillate here; this is the finding.")
    else:
        print("    => congestion responds; oscillation reachable if h' is lifted.")
    return rel


def run_scalar_map(chain: Chain, fracs, tbars, s0: float, n: int) -> List[float]:
    traj = [float(s0)]
    s = float(s0)
    for _ in range(n):
        tbar = float(np.interp(s, fracs, tbars))   # car-share fraction -> OD-mean tt
        s = chain.car_share(tbar)
        traj.append(s)
    return traj


def classify(traj: List[float], tol: float = 1e-4) -> Dict:
    s = np.asarray(traj, float)
    d = np.diff(s)
    nz = d[np.abs(d) > tol]
    sign_changes = int(np.sum(np.sign(nz[1:]) * np.sign(nz[:-1]) < 0)) if nz.size > 1 else 0
    moved = float(np.max(s) - np.min(s))
    final = s[-1]
    turns = [i for i in range(1, len(s) - 1) if (s[i] - s[i - 1]) * (s[i + 1] - s[i]) < 0]
    amps = [abs(s[i] - final) for i in turns]
    decay = float(amps[-1] / amps[0]) if len(amps) >= 2 and amps[0] > 0 else float("nan")

    if moved < 1e-3:
        label = "frozen"
    elif sign_changes <= 1:
        label = "monotone"
    elif np.isnan(decay) or decay < 0.85:
        label = "damped_oscillation"
    elif decay <= 1.15:
        label = "sustained_oscillation"
    else:
        label = "divergent_oscillation"
    return {"label": label, "sign_changes": sign_changes, "moved": moved,
            "decay": decay, "final": final}


def run_screen(parent_directory: str = str(fc.DATA_DIR), scfg: ScreenConfig | None = None):
    if scfg is None:
        scfg = default_screen_config()

    print("=== Building demand -> congestion response surface (cached FW) ===")
    fracs, tbars, tmeds, total = build_response_surface(parent_directory, scfg)
    rel = report_kappa(fracs, tbars)

    print(f"\n{'=' * 92}")
    print("  SCALAR-MAP SCREEN  (free; confirm winners with the full nested-FW module)")
    print(f"{'=' * 92}")
    print(f"{'Scenario':<24}{'h prime':>10}{'sign':>6}{'observed':>22}{'moved%':>9}{'final%':>9}")
    print("-" * 92)

    rows = []
    for name, spec in scfg.scenarios.items():
        chain = Chain(spec, scfg.survey_means)
        gain = chain.loop_gain(lambda s: float(np.interp(s, fracs, tbars)))
        traj = run_scalar_map(chain, fracs, tbars, scfg.s0, scfg.map_iters)
        diag = classify(traj)
        rows.append({"scenario": name, "h_prime": gain["h_prime"],
                     "sign": gain["loop_sign"], **diag, "traj": traj})
        sym = {1: "+", -1: "-", 0: "0"}[gain["loop_sign"]]
        print(f"{name:<24}{gain['h_prime']:>10.4f}{sym:>6}{diag['label']:>22}"
              f"{diag['moved'] * 100:>8.2f}%{diag['final'] * 100:>8.2f}%")
    print(f"{'=' * 92}\n")

    out_dir = os.path.join(parent_directory, "outputs", "shape_fast_screen")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "traj"} for r in rows]).to_csv(
        os.path.join(out_dir, "screen_summary.csv"), index=False)
    for r in rows:
        pd.DataFrame({"iter": range(len(r["traj"])), "car_share": r["traj"]}).to_csv(
            os.path.join(out_dir, f"traj_{r['scenario']}.csv"), index=False)
    print(f"  saved summary + trajectories to {out_dir}")
    return rows


def default_screen_config() -> ScreenConfig:
    INV, POLY, LIN, POW = (CaseForm.INVERSE_SIGMOID, CaseForm.POLYNOMIAL,
                           CaseForm.LINEAR, CaseForm.POWER)

    def spec(c1, c2, c3, c4, c5, t_ref_max=10.0):
        return ScenarioSpec(c1, c2, c3, c4, c5, t_ref_max=t_ref_max)

    scenarios = {
        # control: reproduces the frozen full-FW run (unit norm everywhere)
        "recommended_unit": spec(
            CaseSpec(POLY), CaseSpec(LIN), CaseSpec(INV),
            CaseSpec(LIN), CaseSpec(INV, flip=True)),

        # lever 1: range-renormalise every case -> recovers chain gain
        "recommended_range": spec(
            CaseSpec(POLY, renorm="range"), CaseSpec(LIN, renorm="range"),
            CaseSpec(INV, renorm="range"), CaseSpec(LIN, renorm="range"),
            CaseSpec(INV, flip=True, renorm="range")),

        # lever 1 + 2: range renorm + move operating point onto the steep region
        "recommended_range_steepX": spec(
            CaseSpec(POLY, renorm="range"), CaseSpec(LIN, renorm="range"),
            CaseSpec(INV, renorm="range"), CaseSpec(LIN, renorm="range"),
            CaseSpec(INV, flip=True, renorm="range"), t_ref_max=2.0),

        # steeper forms too (Case 1 power, Case 3 polynomial), range + operating point
        "range_steepforms_steepX": spec(
            CaseSpec(POW, renorm="range"), CaseSpec(LIN, renorm="range"),
            CaseSpec(POLY, renorm="range"), CaseSpec(LIN, renorm="range"),
            CaseSpec(INV, flip=True, renorm="range"), t_ref_max=2.0),

        # positive-loop control under range renorm (should NOT oscillate even at high gain)
        "current_fit_range": spec(
            CaseSpec(POLY, renorm="range"), CaseSpec(LIN, renorm="range"),
            CaseSpec(INV, renorm="range"), CaseSpec(POLY, renorm="range"),
            CaseSpec(INV, renorm="range")),
    }
    return ScreenConfig(scenarios=scenarios)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fast scalar-map screen + kappa diagnostic")
    p.add_argument("--data-dir", default=str(fc.DATA_DIR))
    p.add_argument("--map-iters", type=int, default=None)
    p.add_argument("--s0", type=float, default=None)
    args = p.parse_args()
    cfg = default_screen_config()
    if args.map_iters is not None:
        cfg.map_iters = args.map_iters
    if args.s0 is not None:
        cfg.s0 = args.s0
    run_screen(parent_directory=args.data_dir, scfg=cfg)