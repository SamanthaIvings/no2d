"""
octt_spread_filter.py
=====================
Filter OCTT parameter combinations by how well the curve *spreads* across
the [0, 1] OCTT range within a realistic BPR travel-time window.

Unlike the strict beauty filter (which demands both sigmoid tails),
this filter accepts partial sigmoids — curves that show meaningful
variation even if they don't reach full saturation at both ends.

Criteria:
  1. SPREAD        — OCTT(0) - OCTT(x_max) >= threshold  (default 0.3)
  2. MONOTONICITY  — curve is non-increasing over [0, x_max]
  3. CURVATURE     — curve is not a straight line; has visible S-bend
                     measured as max|d²OCTT/dx²| > threshold
  4. UPPER START   — OCTT(0) >= threshold  (default 0.70)
                     curve starts reasonably high
  5. DESCENT       — OCTT(x_max) < OCTT(0) - spread_min
                     (redundant with #1, but explicit)
  6. NO PLATEAU    — the curve is not flat for more than 60% of the
                     x range; measured as fraction of x where
                     |dOCTT/dx| < 0.05

Outputs (saved to <sweep_dir>/):
  octt_spread_filtered.csv               — passing combinations + metrics
  octt_spread_bush.png                    — all passing curves coloured by x_50
  octt_spread_bush_color_by_<param>.png   — 4 parameter-coloured bush plots
  octt_spread_grid.png                    — sample grid of 25 curves
  octt_spread_report.png                  — diagnostic scatter

Usage:
  python octt_spread_filter.py --x-max 0.5
  python octt_spread_filter.py --x-max 0.5 --spread 0.4
  python octt_spread_filter.py --x-max 0.7 --spread 0.5 --upper-start 0.85
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from itertools import product
from typing import List, Optional

import numpy as np
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd


# -- Style --------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -- Parameter grids ----------------------------------------------------------
B_ATT  = np.round(np.arange(0.1, 1.05, 0.1), 1)
A_JE   = np.round(np.arange(0.6, 2.05, 0.1), 1)
A_OCTT = np.arange(1, 11)
B_OCTT = np.round(np.arange(0.0, -1.05, -0.1), 1)
TOTAL  = len(B_ATT) * len(A_JE) * len(A_OCTT) * len(B_OCTT)
N_PTS  = 500

# -- OCTT chain ---------------------------------------------------------------
def octt_curve(x, b_att, a_je, a_octt, b_octt):
    return 1.0 / (1.0 + np.exp(a_octt * (a_je * b_att * x**2 + b_octt)))

def find_crossing(x, y, level):
    diff = y - level
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        return None
    i = sc[0]
    x0, x1 = x[i], x[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    if abs(y1 - y0) < 1e-15:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


# -- Metrics ------------------------------------------------------------------
@dataclass
class SpreadMetrics:
    b_att: float
    a_je: float
    a_octt: float
    b_octt: float

    octt_start: float       # OCTT(0)
    octt_end: float          # OCTT(x_max)
    spread: float            # octt_start - octt_end
    x_50: Optional[float]    # where OCTT = 0.5 (may be None for partial sigmoids)
    max_gradient: float      # max |dOCTT/dx|
    max_curvature: float     # max |d²OCTT/dx²|
    flat_fraction: float     # fraction of x where |dOCTT/dx| < 0.05
    is_monotonic: bool

    pass_spread: bool
    pass_monotonic: bool
    pass_curvature: bool
    pass_upper_start: bool
    pass_no_plateau: bool
    pass_all: bool


# -- Evaluate -----------------------------------------------------------------
def evaluate(
    x: np.ndarray,
    b_att: float, a_je: float, a_octt: float, b_octt: float,
    x_max: float,
    *,
    min_spread: float = 0.3,
    min_upper: float = 0.70,
    min_curvature: float = 0.5,
    max_flat_frac: float = 0.60,
) -> SpreadMetrics:
    y = octt_curve(x, b_att, a_je, a_octt, b_octt)

    octt_start = float(y[0])
    octt_end = float(y[-1])
    spread = octt_start - octt_end

    x_50 = find_crossing(x, y, 0.5)

    dx = np.diff(x)
    dy = np.diff(y)
    grad = dy / dx
    max_grad = float(np.max(np.abs(grad)))

    # Curvature (second derivative)
    d2y = np.diff(grad) / dx[:-1]
    max_curv = float(np.max(np.abs(d2y))) if len(d2y) > 0 else 0.0

    # Monotonicity
    is_mono = bool(np.all(dy <= 1e-10))

    # Flat fraction
    flat_mask = np.abs(grad) < 0.05
    flat_frac = float(np.sum(flat_mask)) / len(grad)

    # Verdicts
    p_spread = spread >= min_spread
    p_mono = is_mono
    p_curv = max_curv >= min_curvature
    p_upper = octt_start >= min_upper
    p_flat = flat_frac <= max_flat_frac

    return SpreadMetrics(
        b_att=b_att, a_je=a_je, a_octt=a_octt, b_octt=b_octt,
        octt_start=octt_start, octt_end=octt_end, spread=spread,
        x_50=x_50, max_gradient=max_grad, max_curvature=max_curv,
        flat_fraction=flat_frac, is_monotonic=is_mono,
        pass_spread=p_spread, pass_monotonic=p_mono,
        pass_curvature=p_curv, pass_upper_start=p_upper,
        pass_no_plateau=p_flat, pass_all=all([p_spread, p_mono, p_curv, p_upper, p_flat]),
    )


# -- Filter -------------------------------------------------------------------
def run_filter(x_max, **kw):
    x = np.linspace(0.0, x_max, N_PTS)
    passed, failed = [], []
    for ba, aj, ao, bo in product(B_ATT, A_JE, A_OCTT, B_OCTT):
        m = evaluate(x, ba, aj, float(ao), bo, x_max, **kw)
        (passed if m.pass_all else failed).append(m)
    return passed, failed


# -- Plot helpers -------------------------------------------------------------
PARAM_CFGS = {
    "b_att":  {"cmap": "RdYlGn",  "label": "$b_{ATT}$"},
    "a_je":   {"cmap": "cool",    "label": "$a_{JE}$"},
    "a_octt": {"cmap": "plasma",  "label": "$a_{OCTT}$"},
    "b_octt": {"cmap": "RdYlBu",  "label": "$b_{OCTT}$"},
}


def plot_bush_x50(passed, x_max, sweep_dir, dpi=200):
    x = np.linspace(0, x_max, 300)
    n = len(passed)
    alpha = np.clip(30.0 / max(n, 1), 0.04, 0.6)
    lw = np.clip(8.0 / (n ** 0.3), 0.3, 1.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#fafafa")

    spreads = [m.spread for m in passed]
    norm = Normalize(vmin=min(spreads), vmax=max(spreads))
    cmap = plt.get_cmap("Spectral_r")

    for m in passed:
        y = octt_curve(x, m.b_att, m.a_je, m.a_octt, m.b_octt)
        ax.plot(x, y, color=cmap(norm(m.spread)), lw=lw, alpha=alpha, rasterized=True)

    ax.set_xlim(0, x_max); ax.set_ylim(0, 1)
    ax.set_xlabel("$x$ (normalised travel time)")
    ax.set_ylabel("OCTT")
    ax.set_title(f"Spread-filtered OCTT  ({n} / {TOTAL:,} pass, $x_{{max}}$ = {x_max})", fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, label="spread")
    ax.grid(True, lw=0.3, alpha=0.4); fig.tight_layout()
    fig.savefig(os.path.join(sweep_dir, "octt_spread_bush.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_bush_by_param(passed, x_max, sweep_dir, dpi=200):
    if not passed:
        return
    x = np.linspace(0, x_max, 300)
    n = len(passed)
    alpha = np.clip(30.0 / max(n, 1), 0.04, 0.5)
    lw = np.clip(8.0 / (n ** 0.3), 0.3, 1.5)

    for pname, cfg in PARAM_CFGS.items():
        vals = np.array([getattr(m, pname) for m in passed])
        cmap = plt.get_cmap(cfg["cmap"])
        norm = Normalize(vmin=float(vals.min()), vmax=float(vals.max()))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#fafafa")
        for m in passed:
            y = octt_curve(x, m.b_att, m.a_je, m.a_octt, m.b_octt)
            ax.plot(x, y, color=cmap(norm(getattr(m, pname))),
                    lw=lw, alpha=alpha, rasterized=True)
        ax.set_xlim(0, x_max); ax.set_ylim(0, 1)
        ax.set_xlabel("$x$ (normalised travel time)")
        ax.set_ylabel("OCTT")
        ax.set_title(f"Spread-filtered  ({n} / {TOTAL:,}, by {cfg['label']}, "
                     f"$x_{{max}}$ = {x_max})", fontsize=12)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, label=cfg["label"])
        ax.grid(True, lw=0.3, alpha=0.4); fig.tight_layout()
        fig.savefig(os.path.join(sweep_dir, f"octt_spread_bush_color_by_{pname}.png"),
                    dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_diagnostic(passed, failed, x_max, sweep_dir, dpi=200):
    all_m = passed + failed
    is_pass = np.array([m.pass_all for m in all_m])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1: spread vs gradient
    ax = axes[0, 0]
    sp = np.array([m.spread for m in all_m])
    mg = np.array([m.max_gradient for m in all_m])
    ax.scatter(sp[~is_pass], mg[~is_pass], s=2, alpha=0.15, c="#cccccc", label="rejected")
    ax.scatter(sp[is_pass], mg[is_pass], s=6, alpha=0.5, c="#2563eb", label="passed")
    ax.set_xlabel("Spread"); ax.set_ylabel("Max |dOCTT/dx|")
    ax.legend(fontsize=8); ax.set_title("Spread vs Gradient")

    # 2: spread vs flat fraction
    ax = axes[0, 1]
    ff = np.array([m.flat_fraction for m in all_m])
    ax.scatter(sp[~is_pass], ff[~is_pass], s=2, alpha=0.15, c="#cccccc", label="rejected")
    ax.scatter(sp[is_pass], ff[is_pass], s=6, alpha=0.5, c="#2563eb", label="passed")
    ax.axhline(0.60, color="red", ls="--", lw=0.8, label="plateau limit")
    ax.set_xlabel("Spread"); ax.set_ylabel("Flat fraction")
    ax.legend(fontsize=8); ax.set_title("Spread vs Flatness")

    # 3: OCTT(0) vs OCTT(x_max)
    ax = axes[1, 0]
    y0 = np.array([m.octt_start for m in all_m])
    y1 = np.array([m.octt_end for m in all_m])
    ax.scatter(y0[~is_pass], y1[~is_pass], s=2, alpha=0.1, c="#cccccc", label="rejected")
    ax.scatter(y0[is_pass], y1[is_pass], s=6, alpha=0.5, c="#2563eb", label="passed")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
    ax.set_xlabel("OCTT(0)"); ax.set_ylabel(f"OCTT({x_max})")
    ax.legend(fontsize=8); ax.set_title("Start vs End values")

    # 4: rejection breakdown
    ax = axes[1, 1]
    criteria = ["spread", "monotonic", "curvature", "upper_start", "no_plateau"]
    counts = [sum(1 for m in all_m if not getattr(m, f"pass_{c}")) for c in criteria]
    bars = ax.barh(criteria, counts, color="#ef4444", alpha=0.7)
    ax.set_xlabel("Rejected count"); ax.set_title("Rejection breakdown")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}", va="center", fontsize=9, color="#666")

    fig.suptitle(f"Spread filter diagnostics — {len(passed)} passed / {TOTAL:,} total  "
                 f"($x_{{max}}$ = {x_max})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(sweep_dir, "octt_spread_report.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(passed, x_max, sweep_dir, n=25, dpi=200):
    if not passed:
        return
    sorted_p = sorted(passed, key=lambda m: m.spread)
    if len(sorted_p) > n:
        indices = np.linspace(0, len(sorted_p) - 1, n, dtype=int)
        sample = [sorted_p[i] for i in indices]
    else:
        sample = sorted_p

    ncols = 5
    nrows = (len(sample) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows),
                             sharex=True, sharey=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    x = np.linspace(0, x_max, 300)
    cmap = plt.get_cmap("viridis")

    for idx, m in enumerate(sample):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        y = octt_curve(x, m.b_att, m.a_je, m.a_octt, m.b_octt)
        colour = cmap(idx / max(len(sample) - 1, 1))
        ax.plot(x, y, color=colour, lw=1.8)
        ax.fill_between(x, y, alpha=0.08, color=colour)

        if m.x_50 is not None:
            ax.axvline(m.x_50, color="#ccc", lw=0.6, ls="--")
            ax.plot(m.x_50, 0.5, "o", color=colour, ms=4, zorder=5)

        ax.set_xlim(0, x_max); ax.set_ylim(0, 1)
        ax.grid(True, lw=0.3, alpha=0.3)

        ax.text(0.97, 0.03,
                f"$b_{{ATT}}$={m.b_att}\n$a_{{JE}}$={m.a_je}\n"
                f"$a_{{OCTT}}$={m.a_octt:.0f}\n$b_{{OCTT}}$={m.b_octt}",
                transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
                color="#555", bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                       ec="#ddd", alpha=0.85))
        ax.text(0.03, 0.03, f"spr={m.spread:.2f}\nflat={m.flat_fraction:.2f}",
                transform=ax.transAxes, fontsize=6, ha="left", va="bottom", color="#888")

    for idx in range(len(sample), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"Spread-filtered sample  ({len(passed)} total, showing {len(sample)}, "
                 f"$x_{{max}}$ = {x_max})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(sweep_dir, "octt_spread_grid.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Spread-filter OCTT parameter combinations."
    )
    parser.add_argument("--x-max", type=float, default=0.5)
    parser.add_argument("--spread", type=float, default=0.3,
                        help="Min OCTT value range (default: 0.3)")
    parser.add_argument("--upper-start", type=float, default=0.70,
                        help="Min OCTT(0) (default: 0.70)")
    parser.add_argument("--min-curvature", type=float, default=0.5,
                        help="Min max|d²OCTT/dx²| (default: 0.5)")
    parser.add_argument("--max-flat", type=float, default=0.60,
                        help="Max fraction of flat x range (default: 0.60)")
    parser.add_argument("--sweep-dir", type=str, default="sweep")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.sweep_dir, exist_ok=True)

    print(f"Running spread filter on {TOTAL:,} combinations ...")
    print(f"  x_max = {args.x_max}  (= {args.x_max * 10:.1f} min travel time)")
    print(f"  min spread = {args.spread}")
    print(f"  upper start >= {args.upper_start}")
    print(f"  min curvature = {args.min_curvature}")
    print(f"  max flat fraction = {args.max_flat}")
    print()

    t0 = time.time()
    passed, failed = run_filter(
        args.x_max,
        min_spread=args.spread,
        min_upper=args.upper_start,
        min_curvature=args.min_curvature,
        max_flat_frac=args.max_flat,
    )
    print(f"Filter complete in {time.time()-t0:.1f}s")
    print(f"  PASSED: {len(passed)} / {TOTAL:,}  ({100*len(passed)/TOTAL:.1f}%)")
    print()

    if passed:
        rows = [{
            "b_att": m.b_att, "a_je": m.a_je, "a_octt": m.a_octt, "b_octt": m.b_octt,
            "octt_start": round(m.octt_start, 4), "octt_end": round(m.octt_end, 4),
            "spread": round(m.spread, 4),
            "x_50": round(m.x_50, 4) if m.x_50 else None,
            "max_gradient": round(m.max_gradient, 2),
            "max_curvature": round(m.max_curvature, 2),
            "flat_fraction": round(m.flat_fraction, 3),
        } for m in passed]
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.sweep_dir, "octt_spread_filtered.csv")
        df.to_csv(csv_path, index=False)
        print(f"  CSV -> {csv_path}")
        print(f"\n  Passing parameter ranges:")
        print(f"    b_att  : {df['b_att'].min()} .. {df['b_att'].max()}")
        print(f"    a_je   : {df['a_je'].min()} .. {df['a_je'].max()}")
        print(f"    a_octt : {df['a_octt'].min():.0f} .. {df['a_octt'].max():.0f}")
        print(f"    b_octt : {df['b_octt'].max()} .. {df['b_octt'].min()}")
        print(f"    spread : {df['spread'].min():.3f} .. {df['spread'].max():.3f}")
    else:
        print("  No combinations passed.")

    print("\nGenerating plots ...")
    t1 = time.time()
    plot_diagnostic(passed, failed, args.x_max, args.sweep_dir, args.dpi)
    print(f"  diagnostic         ({time.time()-t1:.1f}s)")

    if passed:
        t1 = time.time()
        plot_bush_x50(passed, args.x_max, args.sweep_dir, args.dpi)
        print(f"  bush (spread)      ({time.time()-t1:.1f}s)")

        t1 = time.time()
        plot_bush_by_param(passed, args.x_max, args.sweep_dir, args.dpi)
        print(f"  bush x4 (params)   ({time.time()-t1:.1f}s)")

        t1 = time.time()
        plot_sample_grid(passed, args.x_max, args.sweep_dir, dpi=args.dpi)
        print(f"  sample grid        ({time.time()-t1:.1f}s)")

    print(f"\nAll outputs saved to {args.sweep_dir}/")


if __name__ == "__main__":
    main()