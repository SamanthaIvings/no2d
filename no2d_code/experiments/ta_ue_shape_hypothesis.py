from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from no2d_code.core import filepath_configs as fc
from no2d_code.core.ta_ue_multimodal_extension import (
    MultimodalConfig,
    _make_car_graph,
    _build_run_config,
    _run_single_ue,
    _compute_od_mean_edge_cost,
)
from no2d_code.solver.IO_operations import load_edges, load_filtered_od_and_demand
from no2d_code.solver.bpr import bpr_flow

mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
})


SURVEY_MEANS: Dict[int, Dict[str, List[float]]] = {
    1: {"x": [0.2, 0.4, 0.6, 0.8, 1.0], "y": [69.40, 82.60, 90.10, 93.80, 96.50]},
    2: {"x": [0.2, 1.0],                "y": [57.10, 80.70]},
    3: {"x": [0.2, 0.4, 0.6, 0.8, 1.0], "y": [78.70, 71.80, 60.30, 43.00, 25.30]},
    4: {"x": [0.2, 0.4, 0.6, 0.8, 1.0], "y": [65.50, 56.00, 59.80, 56.50, 60.10]},
    5: {"x": [0.2, 0.5, 0.7, 1.0],      "y": [49.20, 56.00, 69.20, 76.70]},
    6: {"x": [0.2, 1.0],                "y": [41.40, 46.80]},
}


class CaseForm(Enum):
    LINEAR = "linear"               # y = a x + b
    POWER = "power"                 # y = a x^b
    EXPONENTIAL = "exponential"     # y = a e^(b x)
    POLYNOMIAL = "polynomial"       # y = a x^2 + b x + c
    LOGARITHMIC = "logarithmic"     # y = a ln(x) + b
    INVERSE_SIGMOID = "inverse_sigmoid"  # y = L / (1 + e^(k (x - x0)))


def fit_case_form(form: CaseForm, x, y, maxfev: int = 20000) -> Callable[[np.ndarray], np.ndarray]:
    """Fit one candidate form to (x, y) survey means; return f(x) in survey units (0-100)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if form is CaseForm.POLYNOMIAL:
        a, b, c = np.polyfit(x, y, 2)
        return lambda u: a * u * u + b * u + c

    if form is CaseForm.LINEAR:
        a, b = np.polyfit(x, y, 1)
        return lambda u: a * u + b

    if form is CaseForm.POWER:
        def fp(u, a, b):
            return a * np.power(np.clip(u, 1e-9, None), b)
        (a, b), _ = curve_fit(fp, x, y, p0=[float(np.max(y)), 0.3], maxfev=maxfev)
        return lambda u: a * np.power(np.clip(u, 1e-9, None), b)

    if form is CaseForm.EXPONENTIAL:
        def fe(u, a, b):
            return a * np.exp(b * u)
        (a, b), _ = curve_fit(fe, x, y, p0=[float(y[0]) or 1.0, 0.0], maxfev=maxfev)
        return lambda u: a * np.exp(b * u)

    if form is CaseForm.LOGARITHMIC:
        def fl(u, a, b):
            return a * np.log(np.clip(u, 1e-9, None)) + b
        (a, b), _ = curve_fit(fl, x, y, p0=[1.0, float(np.mean(y))], maxfev=maxfev)
        return lambda u: a * np.log(np.clip(u, 1e-9, None)) + b

    if form is CaseForm.INVERSE_SIGMOID:
        def fs(u, L, k, x0):
            return L / (1.0 + np.exp(np.clip(k * (u - x0), -50, 50)))
        trend = float(y[-1] - y[0])
        k0 = 4.0 if trend < 0 else -4.0      # k>0 decreasing, k<0 increasing
        p0 = [float(np.max(y)) * 1.1 or 1.0, k0, float(np.mean(x))]
        (L, k, x0), _ = curve_fit(fs, x, y, p0=p0, maxfev=maxfev)
        return lambda u: L / (1.0 + np.exp(np.clip(k * (u - x0), -50, 50)))

    raise ValueError(f"Unknown form {form}")


@dataclass(frozen=True)
class CaseSpec:
    """One survey case: which fitted form, and whether to reverse its direction.

    `flip` reflects the normalised map about its own realised range over [0, 1]
    (parameter-free: range stays the same, monotonic direction reverses). For an
    inverse sigmoid this is identical to constraining the sign of k — exactly the
    Case 5 fix the survey report recommends.
    """
    form: CaseForm
    flip: bool = False


@dataclass
class ShapeConfig:
    """The behavioural chain for one hypothesis (Cases 1-5 + framework constants)."""
    case1: CaseSpec
    case2: CaseSpec
    case3: CaseSpec
    case4: CaseSpec
    case5: CaseSpec
    bus_share: float = 0.10          # mirrors legacy BUS_SHARE: bus = bus_share * (non-car)
    # congestion normalisation (mirrors legacy octt_mapping: t01 = clip((t*scale - tmin)/(tmax - tmin)))
    tt_scale: float = 60.0
    t_ref_min: float = 0.0
    t_ref_max: float = 10.0

    def specs(self) -> List[CaseSpec]:
        return [self.case1, self.case2, self.case3, self.case4, self.case5]


@dataclass
class ExperimentConfig:
    """Top-level experiment controls + the scenarios to compare."""
    scenarios: Dict[str, ShapeConfig]
    max_outer: int = 30
    tol: float = 0.002
    converge_patience: int = 3        # require this many consecutive sub-tol deltas
    timeout_per_scenario: float = 3600.0
    survey_means: Dict[int, Dict[str, List[float]]] = field(
        default_factory=lambda: SURVEY_MEANS
    )
    mm_cfg: MultimodalConfig | None = None


class BehaviouralChain:
    """Composes the five survey cases into the maps the outer loop needs.

    Each case is a normalised map g_i: [0, 1] -> [0, 1], g_i(u) = clip(f_i(u)/100).
    `flip` reflects within the realised range. The chain is:
        congestion x --g1--> ATT --g2--> JE --g3--> OCTT   (octt_from_traveltime)
        OCTT --g4--> AMC --g5--> car share                 (mode split)
    """

    def __init__(self, cfg: ShapeConfig, means: Dict[int, Dict[str, List[float]]]):
        self.cfg = cfg
        self._g: Dict[int, Callable] = {}
        for case_id, spec in zip(range(1, 6), cfg.specs()):
            self._g[case_id] = self._build_normalised(case_id, spec, means)

    @staticmethod
    def _build_normalised(case_id, spec, means) -> Callable[[np.ndarray], np.ndarray]:
        try:
            f = fit_case_form(spec.form, means[case_id]["x"], means[case_id]["y"])
        except Exception as exc:  # robust fallback so a bad fit never aborts a run
            print(f"  [warn] case {case_id} {spec.form.value} fit failed ({exc}); "
                  f"falling back to polynomial")
            f = fit_case_form(CaseForm.POLYNOMIAL, means[case_id]["x"], means[case_id]["y"])

        grid = np.linspace(0.0, 1.0, 201)
        realised = np.clip(f(grid) / 100.0, 0.0, 1.0)
        rmin, rmax = float(realised.min()), float(realised.max())

        def g(u):
            u = np.clip(np.asarray(u, float), 0.0, 1.0)
            val = np.clip(f(u) / 100.0, 0.0, 1.0)
            if spec.flip:
                val = rmin + rmax - val
            return np.clip(val, 0.0, 1.0)

        return g

    # ---- maps used by the loop -------------------------------------------------

    def octt_from_traveltime(self, tt_edge: np.ndarray) -> np.ndarray:
        """Edge travel times (hours) -> edge OCTT in [0, 1] via Cases 1-3."""
        t = np.asarray(tt_edge, float) * self.cfg.tt_scale
        denom = self.cfg.t_ref_max - self.cfg.t_ref_min
        x = np.clip((t - self.cfg.t_ref_min) / denom, 0.0, 1.0)
        att = self._g[1](x)
        je = self._g[2](att)
        octt = self._g[3](je)
        return octt

    def car_share_from_octt(self, octt: np.ndarray) -> np.ndarray:
        """OCTT in [0, 1] -> car share in [0, 1] via Cases 4-5."""
        amc = self._g[4](octt)
        return self._g[5](amc)

    def mode_split(self, demand_total: np.ndarray, od_octt: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-OD demand split. Bus keeps the framework residual structure."""
        dem = np.asarray(demand_total, float)
        s = self.car_share_from_octt(od_octt)
        ct = s * dem
        non_car = (1.0 - s) * dem
        bu = self.cfg.bus_share * non_car
        att = (1.0 - self.cfg.bus_share) * non_car
        return ct, bu, att

    # ---- FW-free loop-gain prediction -----------------------------------------

    def predict_loop_gain(self, x_star: float, h: float = 1e-4) -> Dict:
        """Local slope sign of each case at the operating point and the composite
        behavioural gain h'(x*). The full loop gain is h'(x*) * kappa, where
        kappa > 0 is the (FW-dependent) congestion coupling, so the SIGN here is
        the loop sign; the MAGNITUDE is indicative of damping."""
        signs: List[int] = []
        slopes: List[float] = []
        u = float(np.clip(x_star, 1e-3, 1 - 1e-3))
        for case_id in range(1, 6):
            g = self._g[case_id]
            d = (float(g(u + h)) - float(g(u - h))) / (2 * h)
            slopes.append(d)
            signs.append(int(np.sign(d)) if abs(d) > 1e-9 else 0)
            u = float(g(u))  # advance operating point along the chain
        hprime = float(np.prod(slopes))
        loop_sign = int(np.sign(hprime))
        return {
            "operating_x": float(x_star),
            "case_signs": signs,
            "case_slopes": slopes,
            "h_prime": hprime,
            "loop_sign": loop_sign,
            "regime": _regime_from_gain(loop_sign, abs(hprime)),
        }


def _regime_from_gain(loop_sign: int, mag: float) -> str:
    if loop_sign >= 0:
        return "positive feedback -> monotone (cannot oscillate)"
    if mag < 1.0:
        return "negative feedback, |h'|<1 -> damped oscillation"
    return "negative feedback, |h'|>=1 -> sustained/growing oscillation"


def classify_trajectory(car_pct: List[float], tol: float) -> Dict:
    s = np.asarray(car_pct, float)
    if len(s) < 3:
        return {"label": "too_short", "sign_changes": 0, "decay": float("nan")}

    deltas = np.diff(s)
    nz = deltas[np.abs(deltas) > tol * 0.25]
    sign_changes = int(np.sum(np.sign(nz[1:]) * np.sign(nz[:-1]) < 0)) if nz.size > 1 else 0

    # amplitudes of successive turning points about the final value
    final = s[-1]
    turns = [i for i in range(1, len(s) - 1)
             if (s[i] - s[i - 1]) * (s[i + 1] - s[i]) < 0]
    amps = [abs(s[i] - final) for i in turns]
    decay = float(amps[-1] / amps[0]) if len(amps) >= 2 and amps[0] > 0 else float("nan")

    if sign_changes <= 1:
        label = "monotone"
    elif np.isnan(decay) or decay < 0.85:
        label = "damped_oscillation"
    elif decay <= 1.15:
        label = "sustained_oscillation"
    else:
        label = "divergent_oscillation"

    return {"label": label, "sign_changes": sign_changes, "decay": decay}


def run_shape_experiment(
    parent_directory: str = str(fc.DATA_DIR),
    exp_cfg: ExperimentConfig | None = None,
):
    if exp_cfg is None:
        exp_cfg = default_experiment_config()
    cfg = exp_cfg.mm_cfg or MultimodalConfig()

    plots_root = os.path.join(parent_directory, "plots", "shape_hypothesis")
    outputs_root = os.path.join(parent_directory, "outputs", "shape_hypothesis")
    os.makedirs(plots_root, exist_ok=True)
    os.makedirs(outputs_root, exist_ok=True)

    edges_car = load_edges(parent_directory)
    origin_destination, demand_total = load_filtered_od_and_demand(parent_directory, cfg.tol)
    run_cfg = _build_run_config(parent_directory, cfg)

    print("=== Stage 1: car UE on full demand (shared across scenarios) ===")
    graph_s1 = _make_car_graph(edges_car, cfg)
    flows_car_s1, _ = _run_single_ue(
        graph=graph_s1,
        origin_destination=origin_destination,
        demand=demand_total,
        run_cfg=run_cfg,
        parent_directory=parent_directory,
        tag="DAY_car_stage1_shared",
        use_cache=True,
        overwrite_cache=False,
    )
    tt_s1 = bpr_flow(graph_s1.free_flow_travel_h, flows_car_s1, graph_s1.capacity, graph_s1.bpr_params)

    # representative operating congestion for the FW-free prediction
    t01_s1 = np.clip(
        (tt_s1 * exp_cfg.scenarios[next(iter(exp_cfg.scenarios))].tt_scale) / 10.0, 0.0, 1.0
    )
    x_star = float(np.median(t01_s1))

    all_histories: Dict[str, pd.DataFrame] = {}
    predictions: Dict[str, Dict] = {}
    aborted = False

    for sc_name, shape_cfg in exp_cfg.scenarios.items():
        if aborted:
            break

        sc_outputs = os.path.join(outputs_root, sc_name)
        os.makedirs(sc_outputs, exist_ok=True)
        csv_path = os.path.join(sc_outputs, f"history_{sc_name}.csv")

        chain = BehaviouralChain(shape_cfg, exp_cfg.survey_means)
        pred = chain.predict_loop_gain(x_star)
        predictions[sc_name] = pred

        print(f"\n{'#' * 74}")
        print(f"#  SCENARIO: {sc_name}")
        print(f"{'#' * 74}")
        _print_prediction(shape_cfg, pred)

        history = []
        tt_current = tt_s1.copy()
        t_start = time.time()
        below = 0

        try:
            for outer in range(exp_cfg.max_outer):
                elapsed = time.time() - t_start
                if elapsed > exp_cfg.timeout_per_scenario:
                    print(f"  [{sc_name}] TIMEOUT after {elapsed:.0f}s at iter {outer}")
                    break

                octt_edge = chain.octt_from_traveltime(tt_current)
                od_octt = _compute_od_mean_edge_cost(
                    graph=graph_s1,
                    origin_destination=origin_destination,
                    edge_cost=octt_edge,
                )
                ct_car, bu_people, att_people = chain.mode_split(demand_total, od_octt)

                total = float(demand_total.sum())
                car_sum = float(ct_car.sum())
                car_pct = car_sum / total
                bus_pct = float(bu_people.sum()) / total
                att_pct = float(att_people.sum()) / total

                row = {
                    "outer_iter": outer,
                    "car_demand": car_sum,
                    "bus_demand": float(bu_people.sum()),
                    "att_demand": float(att_people.sum()),
                    "car_pct": car_pct,
                    "bus_pct": bus_pct,
                    "att_pct": att_pct,
                    "octt_median": float(np.median(octt_edge)),
                    "octt_p05": float(np.percentile(octt_edge, 5)),
                    "octt_p95": float(np.percentile(octt_edge, 95)),
                    "elapsed_s": elapsed,
                }
                history.append(row)
                pd.DataFrame(history).to_csv(csv_path, index=False)

                delta_str = ""
                if outer > 0:
                    delta = car_pct - history[-2]["car_pct"]
                    delta_str = f"  d_car={delta:+.5f}"
                print(f"  [{sc_name}] iter={outer}  car={car_pct:.4f}  bus={bus_pct:.4f}  "
                      f"att={att_pct:.4f}  OCTT_med={row['octt_median']:.4f}  "
                      f"t={elapsed:.0f}s{delta_str}")

                # oscillation-aware convergence: need `converge_patience` consecutive
                # sub-tol deltas, so a single turning point doesn't trigger early.
                if outer > 0 and abs(car_pct - history[-2]["car_pct"]) < exp_cfg.tol:
                    below += 1
                    if below >= exp_cfg.converge_patience:
                        print(f"  [{sc_name}] CONVERGED at iter {outer} "
                              f"({below} consecutive sub-tol deltas, {elapsed:.0f}s)")
                        break
                else:
                    below = 0

                # spill back into the network: re-assign car demand only
                if car_sum < 1e-9:
                    tt_current = graph_s1.free_flow_travel_h.copy()
                    continue

                graph_k = _make_car_graph(edges_car, cfg)
                flows_k, _ = _run_single_ue(
                    graph=graph_k,
                    origin_destination=origin_destination,
                    demand=ct_car,
                    run_cfg=run_cfg,
                    parent_directory=parent_directory,
                    tag=f"shapeEQ_{sc_name}_outer{outer}",
                    use_cache=False,
                    overwrite_cache=False,
                )
                tt_current = bpr_flow(graph_k.free_flow_travel_h, flows_k,
                                      graph_k.capacity, graph_k.bpr_params)
            else:
                print(f"  [{sc_name}] max iters ({exp_cfg.max_outer}) reached "
                      f"({time.time() - t_start:.0f}s)")

        except KeyboardInterrupt:
            print(f"\n  [{sc_name}] INTERRUPTED — saving partial results")
            pd.DataFrame(history).to_csv(csv_path, index=False)
            aborted = True

        if history:
            df = pd.DataFrame(history)
            df.to_csv(csv_path, index=False)
            all_histories[sc_name] = df
            diag = classify_trajectory(df["car_pct"].tolist(), exp_cfg.tol)
            print(f"  [{sc_name}] observed: {diag['label']}  "
                  f"(sign changes={diag['sign_changes']}, decay={diag['decay']:.3f})")

    plot_all(all_histories, plots_root)
    _print_summary(all_histories, predictions, exp_cfg.tol)

    if aborted:
        print("\n  Run interrupted; partial CSVs saved under", outputs_root)


def _print_prediction(shape_cfg: ShapeConfig, pred: Dict):
    names = ["C1 cong->ATT", "C2 ATT->JE", "C3 JE->OCTT", "C4 OCTT->AMC", "C5 AMC->car"]
    sym = {1: "+", -1: "-", 0: "0"}
    print("  FW-free prediction at operating x* = {:.3f}:".format(pred["operating_x"]))
    for nm, spec, sgn, slope in zip(names, shape_cfg.specs(),
                                    pred["case_signs"], pred["case_slopes"]):
        flip = " (flipped)" if spec.flip else ""
        print(f"    {nm:<14} {spec.form.value:<16}{flip:<10} "
              f"slope={slope:+.4f}  sign={sym[sgn]}")
    pattern = " * ".join(sym[s] for s in pred["case_signs"])
    print(f"    loop-product sign: {pattern} = {sym[pred['loop_sign']]}   "
          f"|h'(x*)|={abs(pred['h_prime']):.4f}")
    print(f"    => {pred['regime']}")


def _print_summary(histories: Dict[str, pd.DataFrame], predictions: Dict[str, Dict], tol: float):
    if not histories:
        print("\n  No results to summarise.\n")
        return
    print(f"\n{'=' * 96}")
    print("  SHAPE HYPOTHESIS — PREDICTED vs OBSERVED")
    print(f"{'=' * 96}")
    print(f"{'Scenario':<18}{'Pred loop':>10}{'|h prime|':>10}{'Observed':>22}"
          f"{'Iters':>7}{'Car %':>9}")
    print("-" * 96)
    sym = {1: "+", -1: "-", 0: "0"}
    for name, df in histories.items():
        diag = classify_trajectory(df["car_pct"].tolist(), tol)
        pr = predictions.get(name, {})
        last = df.iloc[-1]
        print(f"{name:<18}{sym.get(pr.get('loop_sign', 0)):>10}"
              f"{abs(pr.get('h_prime', float('nan'))):>10.4f}"
              f"{diag['label']:>22}{int(last['outer_iter']) + 1:>7}"
              f"{last['car_pct'] * 100:>8.2f}%")
    print(f"{'=' * 96}\n")


def plot_all(histories: Dict[str, pd.DataFrame], plots_root: str):
    if not histories:
        return
    for sc_name, df in histories.items():
        sc_plots = os.path.join(plots_root, sc_name)
        os.makedirs(sc_plots, exist_ok=True)
        _plot_convergence(df, sc_name, sc_plots)
    _plot_combined(histories, plots_root)


def _plot_convergence(df: pd.DataFrame, sc_name: str, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    iters = df["outer_iter"].values

    ax1.plot(iters, df["car_pct"] * 100, "o-", color="#2E5090", label="Car", lw=2)
    ax1.plot(iters, df["bus_pct"] * 100, "s-", color="#C04040", label="Bus", lw=1.5)
    ax1.plot(iters, df["att_pct"] * 100, "^-", color="#2E8B57", label="Active", lw=2)
    ax1.set_xlabel("Outer iteration")
    ax1.set_ylabel("Mode share (%)")
    ax1.set_title(f"{sc_name} — mode share")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(iters, df["octt_p05"], df["octt_p95"], alpha=0.2, color="#2E5090")
    ax2.plot(iters, df["octt_median"], "o-", color="#2E5090", lw=2, label="OCTT median")
    ax2.set_xlabel("Outer iteration")
    ax2.set_ylabel("OCTT")
    ax2.set_title(f"{sc_name} — OCTT per iter")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    png = os.path.join(out_dir, f"convergence_{sc_name}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{sc_name}] plot saved: {png}")


def _plot_combined(histories: Dict[str, pd.DataFrame], out_dir: str):
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(histories), 1)))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for (sc_name, df), c in zip(histories.items(), cmap):
        ax.plot(df["outer_iter"], df["car_pct"] * 100, "o-", color=c, lw=2,
                markersize=5, label=sc_name)
    ax.set_xlabel("Outer iteration", fontsize=12)
    ax.set_ylabel("Car mode share (%)", fontsize=12)
    ax.set_title("Shape hypothesis — car share trajectory by scenario",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(out_dir, "all_scenarios_car_trajectory.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined plot: {png}")


def default_experiment_config() -> ExperimentConfig:
    """The scenarios that test the sign/shape hypothesis.

    current_fit       reproduces the survey best fit: C3 dec, C4 weak poly,
                      C5 anomalous INCREASING -> net positive loop -> monotone.
    c5_sign_fix       only flips C5 to decreasing (the keystone the report itself
                      recommends) -> net negative loop, minimal change.
    recommended       C5 flipped + C4 swapped to a monotonic-decreasing form
                      -> clean negative loop, no fragile non-monotonic link.
    recommended_steep adds the magnitude lever: C1 power (steeper at low
                      congestion) and C3 polynomial (steeper mid-range) push |h'|
                      toward 1 for longer oscillation.
    """
    INV = CaseForm.INVERSE_SIGMOID
    POLY = CaseForm.POLYNOMIAL
    LIN = CaseForm.LINEAR
    POW = CaseForm.POWER

    scenarios = {
        "current_fit": ShapeConfig(
            case1=CaseSpec(POLY),
            case2=CaseSpec(LIN),
            case3=CaseSpec(INV),
            case4=CaseSpec(POLY),
            case5=CaseSpec(INV),                 # anomalous increasing, as fitted
        ),
        "c5_sign_fix": ShapeConfig(
            case1=CaseSpec(POLY),
            case2=CaseSpec(LIN),
            case3=CaseSpec(INV),
            case4=CaseSpec(POLY),
            case5=CaseSpec(INV, flip=True),      # constrain C5 to decreasing
        ),
        "recommended": ShapeConfig(
            case1=CaseSpec(POLY),
            case2=CaseSpec(LIN),
            case3=CaseSpec(INV),
            case4=CaseSpec(LIN),                 # monotonic-decreasing swap
            case5=CaseSpec(INV, flip=True),
        ),
        "recommended_steep": ShapeConfig(
            case1=CaseSpec(POW),                 # steeper at low congestion
            case2=CaseSpec(LIN),
            case3=CaseSpec(POLY),                # steeper mid-range than sigmoid tail
            case4=CaseSpec(LIN),
            case5=CaseSpec(INV, flip=True),
        ),
    }
    return ExperimentConfig(scenarios=scenarios, max_outer=10, tol=0.002,
                            converge_patience=6)


def main():
    parser = argparse.ArgumentParser(
        description="Shape/sign hypothesis test for the iterative mode-choice equilibrium"
    )
    parser.add_argument("--data-dir", default=str(fc.DATA_DIR))
    parser.add_argument("--max-outer", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()

    exp_cfg = default_experiment_config()
    if args.max_outer is not None:
        exp_cfg.max_outer = args.max_outer
    if args.tol is not None:
        exp_cfg.tol = args.tol
    if args.patience is not None:
        exp_cfg.converge_patience = args.patience
    if args.timeout is not None:
        exp_cfg.timeout_per_scenario = args.timeout

    run_shape_experiment(parent_directory=args.data_dir, exp_cfg=exp_cfg)


if __name__ == "__main__":
    main()