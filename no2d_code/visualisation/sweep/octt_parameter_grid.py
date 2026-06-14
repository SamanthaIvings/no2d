"""
octt_grid_aoctt_boctt.py
========================
For a chosen (b_att, a_je) pair, generate a 5x5 subplot grid showing
individual OCTT curves for every combination of:

  rows:    a_octt in {3, 4, 5, 6, 7}       (top to bottom)
  columns: b_octt in {-0.6, -0.7, -0.8, -0.9, -1.0}  (left to right)

Each subplot shows a single, clean curve so the shape of the sigmoid
is visible without the bush overlap.  A bold "productive regime" overview
in one figure.

Usage:
  python octt_grid_aoctt_boctt.py                          # defaults: b_att=0.5, a_je=1.3
  python octt_grid_aoctt_boctt.py --b-att 0.8 --a-je 1.0
  python octt_grid_aoctt_boctt.py --b-att 0.5 --a-je 1.3 --sweep-dir outputs/sweep
  python octt_grid_aoctt_boctt.py --all                    # one grid per (b_att, a_je) combo
"""

from __future__ import annotations

import argparse
import os
import time
from itertools import product

import numpy as np
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# -- Plotting style -----------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": False,  # set True if LaTeX + type1ec.sty available
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# -- Productive-regime grids --------------------------------------------------
A_OCTT_GRID = np.array([3, 4, 5, 6, 7])                          # rows
B_OCTT_GRID = np.round(np.arange(-0.6, -1.05, -0.1), 1)          # columns

# Full grids for --all mode
B_ATT_ALL = np.round(np.arange(0.1, 1.05, 0.1), 1)
A_JE_ALL  = np.round(np.arange(0.6, 2.05, 0.1), 1)

N_PTS = 200
X_MAX = 1.0


# -- OCTT chain ---------------------------------------------------------------
def octt_chain(x, b_att, a_je, a_octt, b_octt):
    return 1.0 / (1.0 + np.exp(a_octt * (a_je * b_att * x**2 + b_octt)))


# -- Build one 5x5 grid figure ------------------------------------------------
def make_grid_figure(b_att: float, a_je: float) -> mpl.figure.Figure:
    """
    5x5 grid: rows = a_octt (3..7), columns = b_octt (-0.6..-1.0).
    Each subplot shows the single OCTT curve for that (b_att, a_je, a_octt, b_octt).
    """
    n_rows = len(A_OCTT_GRID)
    n_cols = len(B_OCTT_GRID)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 11),
        sharex=True, sharey=True,
    )

    x = np.linspace(0.0, X_MAX, N_PTS)

    # Colour by combined "intensity" = a_octt * |b_octt|
    cmap = plt.get_cmap("viridis")
    max_intensity = float(A_OCTT_GRID.max()) * float(np.abs(B_OCTT_GRID).max())
    min_intensity = float(A_OCTT_GRID.min()) * float(np.abs(B_OCTT_GRID).min())
    norm = Normalize(vmin=min_intensity, vmax=max_intensity)

    for i, a_octt in enumerate(A_OCTT_GRID):
        for j, b_octt in enumerate(B_OCTT_GRID):
            ax = axes[i, j]
            y = octt_chain(x, b_att, a_je, float(a_octt), b_octt)

            intensity = float(a_octt) * abs(float(b_octt))
            color = cmap(norm(intensity))

            ax.plot(x, y, color=color, lw=1.8)
            ax.fill_between(x, y, alpha=0.08, color=color)

            # OCTT at x=0 and x=1 annotations
            y0 = float(y[0])
            y1 = float(y[-1])
            ax.annotate(f"{y0:.2f}", xy=(0.02, y0), fontsize=7,
                        color="#666666", va="bottom")
            ax.annotate(f"{y1:.2f}", xy=(0.95, y1), fontsize=7,
                        color="#666666", va="top", ha="right")

            # Midpoint marker (where OCTT = 0.5)
            cross = np.where(np.diff(np.sign(y - 0.5)))[0]
            if len(cross) > 0:
                x_mid = x[cross[0]]
                ax.axvline(x_mid, color="#cccccc", lw=0.6, ls="--")
                ax.plot(x_mid, 0.5, "o", color=color, ms=4, zorder=5)
                ax.annotate(f"x={x_mid:.2f}", xy=(x_mid, 0.48),
                            fontsize=6, color="#999999", ha="center", va="top")

            ax.set_xlim(0, X_MAX)
            ax.set_ylim(0, 1)
            ax.grid(True, lw=0.3, alpha=0.3)

            # Subplot label
            ax.text(0.97, 0.03,
                    f"$a_{{OCTT}}$={a_octt}\n$b_{{OCTT}}$={b_octt}",
                    transform=ax.transAxes, fontsize=7,
                    ha="right", va="bottom", color="#555555",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#dddddd", alpha=0.8))

    # Row labels (left side)
    for i, a_octt in enumerate(A_OCTT_GRID):
        axes[i, 0].set_ylabel(f"$a_{{OCTT}}$ = {a_octt}\n\nOCTT", fontsize=9)

    # Column labels (top)
    for j, b_octt in enumerate(B_OCTT_GRID):
        axes[0, j].set_title(f"$b_{{OCTT}}$ = {b_octt}", fontsize=10, pad=8)

    # Bottom x-labels
    for j in range(n_cols):
        axes[-1, j].set_xlabel("$x$", fontsize=9)

    fig.suptitle(
        f"OCTT sigmoid grid — $b_{{ATT}}$ = {b_att},  $a_{{JE}}$ = {a_je}"
        f"    (productive regime: 25 panels)",
        fontsize=13, y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="5x5 grid of OCTT curves for fixed (b_att, a_je)."
    )
    parser.add_argument("--b-att", type=float, default=0.5,
                        help="b_att value (default: 0.5)")
    parser.add_argument("--a-je", type=float, default=1.3,
                        help="a_je value (default: 1.3)")
    parser.add_argument("--sweep-dir", type=str, default="sweep")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fmt", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--all", action="store_true",
                        help="Generate one grid for every (b_att, a_je) combination")
    args = parser.parse_args()

    os.makedirs(args.sweep_dir, exist_ok=True)

    if args.all:
        combos = list(product(B_ATT_ALL, A_JE_ALL))
        print(f"Generating {len(combos)} grid figures (all b_att x a_je) ...")
        for idx, (ba, aj) in enumerate(combos):
            t0 = time.time()
            fig = make_grid_figure(ba, aj)
            name = f"octt_grid_batt{ba:.1f}_aje{aj:.1f}.{args.fmt}"
            out = os.path.join(args.sweep_dir, name)
            fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(combos)}] {time.time()-t0:.1f}s -> {name}")
        print(f"\n{len(combos)} grid figures saved to {args.sweep_dir}/")
    else:
        t0 = time.time()
        ba, aj = args.b_att, args.a_je
        print(f"Generating grid for b_att={ba}, a_je={aj} ...")
        fig = make_grid_figure(ba, aj)
        name = f"octt_grid_batt{ba:.1f}_aje{aj:.1f}.{args.fmt}"
        out = os.path.join(args.sweep_dir, name)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  done in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()