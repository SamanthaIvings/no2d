"""
octt_bush_productive.py
=======================
Generate bush plots restricted to the "productive" parameter regime where
the OCTT mapping produces meaningful edge-level differentiation:

  b_octt in {-0.6, -0.7, -0.8, -0.9, -1.0}   (5 values)
  a_octt in {3, 4, 5, 6, 7}                    (5 values)
  b_att  in {0.1, 0.2, ..., 1.0}               (10 values, full grid)
  a_je   in {0.6, 0.7, ..., 2.0}               (15 values, full grid)

Total: 10 x 15 x 5 x 5 = 3,750 curves (23% of full 16,500 grid).

Usage:
  python octt_bush_productive.py
  python octt_bush_productive.py --sweep-dir outputs/sweep --dpi 600
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


# -- Plotting style (project conventions) -------------------------------------
mpl.rcParams.update({
    "text.usetex": False,  # set True if LaTeX + type1ec.sty available
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# -- Parameter grids ----------------------------------------------------------
# Full grids (unchanged)
B_ATT = np.round(np.arange(0.1, 1.05, 0.1), 1)    # 10 values
A_JE  = np.round(np.arange(0.6, 2.05, 0.1), 1)     # 15 values

# Productive regime only
A_OCTT = np.arange(3, 8)                             # {3, 4, 5, 6, 7}
B_OCTT = np.round(np.arange(-0.6, -1.05, -0.1), 1)  # {-0.6, -0.7, -0.8, -0.9, -1.0}

TOTAL = len(B_ATT) * len(A_JE) * len(A_OCTT) * len(B_OCTT)  # 3,750


# -- Config -------------------------------------------------------------------
CONFIGS = {
    "b_att":  {"cmap": "RdYlGn",  "vmin": 0.1,  "vmax": 1.0,  "label": "$b_{ATT}$"},
    "a_je":   {"cmap": "cool",    "vmin": 0.6,  "vmax": 2.0,  "label": "$a_{JE}$"},
    "a_octt": {"cmap": "plasma",  "vmin": 3.0,  "vmax": 7.0,  "label": "$a_{OCTT}$"},
    "b_octt": {"cmap": "RdYlBu",  "vmin": -1.0, "vmax": -0.6, "label": "$b_{OCTT}$"},
}


# -- OCTT chain ---------------------------------------------------------------
def octt_chain(x, b_att, a_je, a_octt, b_octt):
    return 1.0 / (1.0 + np.exp(a_octt * (a_je * b_att * x**2 + b_octt)))


# -- Figure builder -----------------------------------------------------------
def make_bush_figure(color_by, n_pts=100, x_max=1.0, alpha=0.12, linewidth=0.4):
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
        f"OCTT productive regime  ({TOTAL:,} curves, coloured by {cfg['label']})",
        fontsize=12,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, label=cfg["label"])
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    return fig


# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="OCTT bush plots — productive regime only (3,750 curves)."
    )
    parser.add_argument("--sweep-dir", type=str, default="sweep")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fmt", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--alpha", type=float, default=0.12)
    args = parser.parse_args()

    os.makedirs(args.sweep_dir, exist_ok=True)

    for param_name in ["b_att", "a_je", "a_octt", "b_octt"]:
        t0 = time.time()
        print(f"[{param_name}] rendering {TOTAL:,} curves (productive regime) ...")
        fig = make_bush_figure(param_name, alpha=args.alpha)
        out = os.path.join(args.sweep_dir,
                           f"octt_bush_productive_color_by_{param_name}.{args.fmt}")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  done in {time.time() - t0:.1f}s -> {out}")

    print(f"\nAll 4 productive-regime figures saved to {args.sweep_dir}/")


if __name__ == "__main__":
    main()