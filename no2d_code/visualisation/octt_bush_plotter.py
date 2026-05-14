"""
octt_bush_plot.py
=================
Generate "bush" plots of the OCTT formula across all parameter combinations
from the parameter_combinator_experiment grid (pink-sticker specs).

Chain:
  (1) ATT  = b_att * x^2                              b_att  in {0.1, 0.2, ..., 1.0}
  (2) JE   = a_je  * ATT                              a_je   in {0.6, 0.7, ..., 2.0}
  (3) OCTT = 1 / (1 + exp(a_octt * (JE + b_octt)))    a_octt in {1, 2, ..., 10}
                                                       b_octt in {0.0, -0.1, ..., -1.0}

Total combinations: 10 x 15 x 10 x 11 = 16,500 curves  (b_att=0 excluded).

Outputs (saved to <sweep_dir>/):
  octt_bush_color_by_b_att.png
  octt_bush_color_by_a_je.png
  octt_bush_color_by_a_octt.png
  octt_bush_color_by_b_octt.png

Usage:
  python octt_bush_plot.py                      # default: saves to ./sweep/
  python octt_bush_plot.py --sweep-dir /my/path # custom output directory
  python octt_bush_plot.py --dpi 600            # publication quality
  python octt_bush_plot.py --fmt pdf            # PDF output instead of PNG
"""

from __future__ import annotations

import argparse
import os
import time
from itertools import product

import numpy as np
import matplotlib as mpl

mpl.use("Agg")  # headless backend -- remove if running interactively

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# -- Plotting style (matches project conventions) -----------------------------
# Set text.usetex = True on systems with full LaTeX + type1ec.sty installed.
_USETEX = False

mpl.rcParams.update({
    "text.usetex": _USETEX,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# -- Parameter grids (pink-sticker specs, b_att=0 excluded) -------------------
B_ATT  = np.round(np.arange(0.1, 1.05, 0.1), 1)   # 10 values
A_JE   = np.round(np.arange(0.6, 2.05, 0.1), 1)    # 15 values
A_OCTT = np.arange(1, 11)                            # 10 values
B_OCTT = np.round(np.arange(0.0, -1.05, -0.1), 1)   # 11 values

TOTAL = len(B_ATT) * len(A_JE) * len(A_OCTT) * len(B_OCTT)  # 16,500


# -- Colour-map and label config per parameter --------------------------------
CONFIGS = {
    "b_att":  {"cmap": "RdYlGn",  "vmin": 0.1, "vmax": 1.0,  "label": "$b_{ATT}$"},
    "a_je":   {"cmap": "cool",    "vmin": 0.6, "vmax": 2.0,  "label": "$a_{JE}$"},
    "a_octt": {"cmap": "plasma",  "vmin": 1.0, "vmax": 10.0, "label": "$a_{OCTT}$"},
    "b_octt": {"cmap": "RdYlBu",  "vmin": -1.0, "vmax": 0.0, "label": "$b_{OCTT}$"},
}


# -- OCTT chain (vectorised) -------------------------------------------------
def octt_chain(x: np.ndarray, b_att: float, a_je: float,
               a_octt: float, b_octt: float) -> np.ndarray:
    """Evaluate ATT -> JE -> OCTT at array of x values."""
    return 1.0 / (1.0 + np.exp(a_octt * (a_je * b_att * x**2 + b_octt)))


# -- Single bush figure ------------------------------------------------------
def make_bush_figure(
    color_by: str,
    n_pts: int = 100,
    x_max: float = 1.0,
    alpha: float = 0.08,
    linewidth: float = 0.3,
) -> mpl.figure.Figure:
    """
    Build one bush figure with all 16,500 curves coloured by *color_by*.
    Each curve is rasterized for fast PDF/PNG export.
    """
    cfg  = CONFIGS[color_by]
    cmap = plt.get_cmap(cfg["cmap"])
    norm = Normalize(vmin=cfg["vmin"], vmax=cfg["vmax"])

    x = np.linspace(0.0, x_max, n_pts)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("#fafafa")

    for ba, aj, ao, bo in product(B_ATT, A_JE, A_OCTT, B_OCTT):
        y = octt_chain(x, ba, aj, float(ao), bo)

        if   color_by == "b_att":  cv = ba
        elif color_by == "a_je":   cv = aj
        elif color_by == "a_octt": cv = float(ao)
        else:                      cv = bo

        ax.plot(x, y, color=cmap(norm(cv)), lw=linewidth, alpha=alpha,
                rasterized=True)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$x$ (normalised travel time)")
    ax.set_ylabel("OCTT")
    ax.set_title(
        f"OCTT parameter bush  ({TOTAL:,} curves, coloured by {cfg['label']})",
        fontsize=12,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, label=cfg["label"])

    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    return fig


# -- Main ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OCTT bush plots for all parameter combinations."
    )
    parser.add_argument("--sweep-dir", type=str, default="sweep",
                        help="Output directory (default: ./sweep/)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Output resolution (default: 200; use 600 for publication)")
    parser.add_argument("--fmt", type=str, default="png", choices=["png", "pdf"],
                        help="Output format (default: png)")
    parser.add_argument("--alpha", type=float, default=0.08,
                        help="Curve opacity (default: 0.08)")
    args = parser.parse_args()

    os.makedirs(args.sweep_dir, exist_ok=True)

    for param_name in ["b_att", "a_je", "a_octt", "b_octt"]:
        t0 = time.time()
        print(f"[{param_name}] rendering {TOTAL:,} curves ...")

        fig = make_bush_figure(param_name, alpha=args.alpha)
        out_path = os.path.join(
            args.sweep_dir, f"octt_bush_color_by_{param_name}.{args.fmt}"
        )
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"  done in {time.time() - t0:.1f}s -> {out_path}")

    print(f"\nAll 4 figures saved to {args.sweep_dir}/")


if __name__ == "__main__":
    main()