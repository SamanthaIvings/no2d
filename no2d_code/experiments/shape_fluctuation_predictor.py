from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from no2d_code.experiments.ta_ue_shape_hypothesis import (
    CaseForm,
    fit_case_form,
    SURVEY_MEANS,
)


@dataclass(frozen=True)
class Variant:
    form: CaseForm
    flip: bool = False
    renorm: str = "range"          # "range" (min-max to [0,1]) or "unit" (score/100)

    def tag(self) -> str:
        f = {"range": "r", "unit": "u"}[self.renorm]
        return f"{self.form.value[:6]}{'-' if self.flip else '+'}{f}"


INV, POLY, LIN, POW, EXP, LOG = (
    CaseForm.INVERSE_SIGMOID, CaseForm.POLYNOMIAL, CaseForm.LINEAR,
    CaseForm.POWER, CaseForm.EXPONENTIAL, CaseForm.LOGARITHMIC,
)

# Defaults keep the search bounded (~6k combos, < 1 s) and behaviourally sane:
# Cases 1-2 stay increasing (robust); Cases 3-5 (which set the loop sign) vary.
DEFAULT_CANDIDATES: Dict[int, List[Variant]] = {
    1: [Variant(POLY, False, r) for r in ("range", "unit")]
       + [Variant(POW, False, r) for r in ("range", "unit")],
    2: [Variant(LIN, False, r) for r in ("range", "unit")],
    3: [Variant(form, False, r) for form in (INV, POLY) for r in ("range", "unit")],
    4: [Variant(form, flip, r)
        for form in (LIN, POW, EXP, POLY) for flip in (False, True) for r in ("range", "unit")],
    5: [Variant(form, flip, r)
        for form in (INV, POLY, POW) for flip in (False, True) for r in ("range", "unit")],
}


@lru_cache(maxsize=None)
def _fit(case_id: int, form: CaseForm):
    m = SURVEY_MEANS[case_id]
    f = fit_case_form(form, m["x"], m["y"])
    grid = np.linspace(0.0, 1.0, 201)
    raw = np.clip(f(grid) / 100.0, 0.0, 1.0)
    return f, float(raw.min()), float(raw.max())


def build_map(case_id: int, v: Variant) -> Callable[[float], float]:
    f, rmin, rmax = _fit(case_id, v.form)
    span = max(rmax - rmin, 1e-9)

    def g(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        val = float(np.clip(f(u) / 100.0, 0.0, 1.0))
        if v.renorm == "range":
            val = (val - rmin) / span
            if v.flip:
                val = 1.0 - val
        else:
            if v.flip:
                val = rmin + rmax - val
        return float(np.clip(val, 0.0, 1.0))

    return g


def chain_gain(maps: List[Callable], x_star: float, h: float = 1e-4
               ) -> Tuple[List[int], List[float], float]:
    """Per-case local slope sign/value at the operating point, and their product."""
    signs, slopes = [], []
    u = float(np.clip(x_star, 1e-3, 1 - 1e-3))
    for g in maps:
        d = (g(u + h) - g(u - h)) / (2 * h)
        slopes.append(d)
        signs.append(int(np.sign(d)) if abs(d) > 1e-9 else 0)
        u = g(u)
    return signs, slopes, float(np.prod(slopes))


def regime(G: float) -> str:
    if G >= 0:
        return "monotone (no fluctuation)"
    if G > -1.0:
        return "damped oscillation"
    if G > -1.05:
        return "near-sustained oscillation"
    return "growing oscillation / limit cycle"


def visible_overshoots(G: float, eps: float = 1e-3) -> float:
    """Linear-theory estimate of how many alternations before decaying below eps."""
    if G >= 0:
        return 0.0
    if G <= -1.0:
        return float("inf")
    return float(np.log(eps) / np.log(abs(G)))


def evaluate(config: Dict[int, Variant], x_eval: float, x_grid: np.ndarray) -> Dict:
    maps = [build_map(i, config[i]) for i in range(1, 6)]
    signs, slopes, Bp = chain_gain(maps, x_eval)

    # robustness: fraction of the operating-range grid where the loop stays negative,
    # and the operating point that maximises |B'| among negative-loop points
    neg_count, best_absB, best_x, best_signedB = 0, 0.0, x_eval, Bp
    for x in x_grid:
        _, _, b = chain_gain(maps, float(x))
        if b < 0:
            neg_count += 1
            if abs(b) > best_absB:
                best_absB, best_x, best_signedB = abs(b), float(x), b
    frac_negative = neg_count / len(x_grid)

    return {
        "signature": " ".join(f"C{i}:{config[i].tag()}" for i in range(1, 6)),
        "case_signs": "".join({1: "+", -1: "-", 0: "0"}[s] for s in signs),
        "loop_sign": int(np.sign(Bp)) or 0,
        "B_prime": Bp,
        "abs_B": abs(Bp),
        "frac_negative_over_range": frac_negative,
        "best_absB_over_range": best_absB,
        "x_at_best": best_x,
        "kappa_threshold": (1.0 / abs(Bp)) if abs(Bp) > 1e-12 else float("inf"),
    }


def enumerate_and_rank(candidates: Dict[int, List[Variant]], x_eval: float,
                       x_grid: np.ndarray, kappa: float | None) -> pd.DataFrame:
    keys = [1, 2, 3, 4, 5]
    combos = itertools.product(*(candidates[k] for k in keys))
    n_total = int(np.prod([len(candidates[k]) for k in keys]))
    print(f"[enumerate] evaluating {n_total} shape combinations (no FW) ...")

    rows = []
    for combo in combos:
        config = dict(zip(keys, combo))
        r = evaluate(config, x_eval, x_grid)
        if kappa is not None:
            G = kappa * r["B_prime"]
            r["G_at_kappa"] = G
            r["regime"] = regime(G)
            r["approx_overshoots"] = visible_overshoots(G)
        rows.append(r)

    df = pd.DataFrame(rows)
    # fluctuation-capable first (negative loop at the operating point), then by gain
    df["can_fluctuate"] = df["loop_sign"] < 0
    sort_cols = (["can_fluctuate", "abs_B"] if kappa is None
                 else ["can_fluctuate", "G_at_kappa"])
    ascending = ([False, False] if kappa is None else [False, True])  # most negative G first
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser(
        description="Predict which survey shape combinations give mode fluctuation (no FW)")
    p.add_argument("--x-eval", type=float, default=0.30,
                   help="operating congestion x* to evaluate the loop at (move via renorm)")
    p.add_argument("--x-min", type=float, default=0.05)
    p.add_argument("--x-max", type=float, default=0.95)
    p.add_argument("--kappa", type=float, default=None,
                   help="measured congestion coupling; if given, predicts regime + overshoots")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--out", default="shape_fluctuation_ranking.csv")
    args = p.parse_args()

    x_grid = np.linspace(args.x_min, args.x_max, 19)
    df = enumerate_and_rank(DEFAULT_CANDIDATES, args.x_eval, x_grid, args.kappa)

    capable = df[df["can_fluctuate"]]
    print(f"\n{'=' * 100}")
    print(f"  SHAPES PREDICTED TO FLUCTUATE  (negative loop at x*={args.x_eval}): "
          f"{len(capable)} of {len(df)} combinations")
    print(f"{'=' * 100}")

    cols = ["signature", "case_signs", "abs_B", "kappa_threshold",
            "frac_negative_over_range", "best_absB_over_range", "x_at_best"]
    if args.kappa is not None:
        cols = ["signature", "case_signs", "G_at_kappa", "regime", "approx_overshoots"]

    head = capable.head(args.top) if len(capable) else df.head(args.top)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(head[cols].to_string(index=False))

    print(f"\n  Read: case_signs is the sign of each case slope (C1..C5); an ODD number of '-' "
          f"\n  among C3,C4,C5 gives the negative loop. |B'| larger => lower kappa needed to "
          f"\n  reach sustained fluctuation (kappa_threshold = 1/|B'|).")
    if args.kappa is None:
        print("  Pass --kappa <measured value from the response surface> for regime + overshoot "
              "predictions.")

    df.to_csv(args.out, index=False)
    print(f"\n  Full ranking ({len(df)} combos) saved to {args.out}")


if __name__ == "__main__":
    main()