from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from no2d_code.core import filepath_configs as fc

# ── Colour palette ────────────────────────────────────────────────────────
C_OLD = "#2b6cb0"
C_NEW = "#c53030"
C_OLD_LIGHT = "#bee3f8"
C_NEW_LIGHT = "#fed7d7"
C_FILL = "#fbb6b6"
C_GRID = "#e2e8f0"
C_BG = "#fafafa"
C_TEXT = "#1a202c"
C_ANNOT = "#718096"

XLABEL = "Normalised travel time  x ∈ [0, 1]"


# ── Curve data ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Curves:
    x01: np.ndarray
    att_o: np.ndarray
    je_o: np.ndarray
    octt_o: np.ndarray
    att_n: np.ndarray
    je_n: np.ndarray
    octt_n: np.ndarray
    x01_inflect_old: float
    x01_inflect_new: float


def compute_curves() -> Curves:
    """Evaluate the old and survey-calibrated mapping chains on normalised x."""
    x01 = np.linspace(0, 1, 600)

    # Old pipeline on normalised x
    att_o = x01 ** 2
    je_o = att_o  # JE_POWER = 1
    # Old sigmoid: arg = -2.5 + 0.2*t², where t = x01 * 10 (minutes)
    t_min_old = x01 * 10.0
    arg_old = -2.5 + 0.2 * t_min_old ** 2
    octt_o = 1.0 / (1.0 + np.exp(np.clip(arg_old, -500, 500)))

    # New (survey stickers) on normalised x
    att_n = 1.0 * x01 ** 2       # sticker #1: b = 1.0
    je_n = 0.9 * att_n           # sticker #2: a = 0.9
    octt_n = 1.0 / (1.0 + np.exp(5.0 * (je_n - 0.8)))  # sticker #3

    # inflection points in normalised x
    # Old: arg=0 → 0.2*t²=2.5 → t=3.54min → x01 = 3.54/10 = 0.354
    x01_inflect_old = float(np.sqrt(12.5) / 10.0)   # 0.354
    # New: 5*(JE-0.8)=0 → JE=0.8 → 0.9*x01²=0.8 → x01 = sqrt(0.8/0.9) = 0.943
    x01_inflect_new = float(np.sqrt(0.8 / 0.9))     # 0.943

    return Curves(
        x01=x01, att_o=att_o, je_o=je_o, octt_o=octt_o,
        att_n=att_n, je_n=je_n, octt_n=octt_n,
        x01_inflect_old=x01_inflect_old, x01_inflect_new=x01_inflect_new,
    )


# ── Helper ────────────────────────────────────────────────────────────────
def _style_ax(ax, title, xlabel=XLABEL, ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold", color=C_TEXT, pad=10)
    ax.set_xlabel(xlabel, fontsize=9.5, color=C_TEXT)
    ax.set_ylabel(ylabel, fontsize=9.5, color=C_TEXT)
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(labelsize=8.5, colors=C_TEXT)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e0")
        spine.set_linewidth(0.6)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Three-panel side by side
# ══════════════════════════════════════════════════════════════════════════
def figure_three_panel(c: Curves, out_dir: Path) -> Path:
    x01 = c.x01
    att_o, att_n = c.att_o, c.att_n
    je_o, je_n = c.je_o, c.je_n
    octt_o, octt_n = c.octt_o, c.octt_n
    x01_inflect_old, x01_inflect_new = c.x01_inflect_old, c.x01_inflect_new

    fig1, axes1 = plt.subplots(1, 3, figsize=(17, 5.2))
    fig1.patch.set_facecolor("white")
    fig1.suptitle("Old Pipeline  vs  Survey-Calibrated Stickers",
                  fontsize=14, fontweight="bold", color=C_TEXT, y=1.01)

    # Panel 1: ATT
    ax = axes1[0]
    ax.plot(x01, att_o, color=C_OLD, linewidth=2.4, label="Old:  ATT = x²  (b = 1.0)")
    ax.plot(x01, att_n, color=C_NEW, linewidth=2.4, linestyle="--",
            label="New: ATT = 1.0·x²  (b = 1.0)")
    ax.text(0.5, 0.13, "IDENTICAL", fontsize=16, color="#a0aec0",
            ha="center", fontstyle="italic", alpha=0.5, fontweight="bold")
    _style_ax(ax, "Step 1:  ATT from BPR", ylabel="ATT index  [0, 1]")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)

    # Panel 2: JE
    ax = axes1[1]
    ax.plot(x01, je_o, color=C_OLD, linewidth=2.4, label="Old:  JE = ATT¹·⁰  (a = 1.0)")
    ax.plot(x01, je_n, color=C_NEW, linewidth=2.4, linestyle="--",
            label="New: JE = 0.9·ATT  (a = 0.9)")
    ax.fill_between(x01, je_n, je_o, alpha=0.15, color=C_NEW)
    ax.annotate("10 % reduction", xy=(0.72, 0.42), fontsize=11, color=C_NEW,
                fontstyle="italic", ha="center", fontweight="bold")
    _style_ax(ax, "Step 2:  JE from ATT", ylabel="JE index  [0, 1]")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)

    # Panel 3: OCTT
    ax = axes1[2]
    ax.plot(x01, octt_o, color=C_OLD, linewidth=2.4,
            label=f"Old:  sigmoid of t²  [inflection x = {x01_inflect_old:.2f}]")
    ax.plot(x01, octt_n, color=C_NEW, linewidth=2.4, linestyle="--",
            label=f"New: sigmoid of JE  [inflection x = {x01_inflect_new:.2f}]")
    ax.fill_between(x01, octt_o, octt_n, alpha=0.13, color=C_NEW)

    ax.axvline(x01_inflect_old, color=C_OLD, linewidth=0.9, linestyle=":", alpha=0.5)
    ax.axvline(x01_inflect_new, color=C_NEW, linewidth=0.9, linestyle=":", alpha=0.5)
    ax.plot(x01_inflect_old, 0.5, "o", color=C_OLD, markersize=6, zorder=5)
    ax.plot(x01_inflect_new, 0.5, "o", color=C_NEW, markersize=6, zorder=5)

    ax.annotate(f"Old\nx = {x01_inflect_old:.2f}",
                xy=(x01_inflect_old, 0.5), xytext=(0.1, 0.28),
                fontsize=8, color=C_OLD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_OLD, lw=1))
    ax.annotate(f"New\nx = {x01_inflect_new:.2f}",
                xy=(x01_inflect_new, 0.5), xytext=(0.75, 0.68),
                fontsize=8, color=C_NEW, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_NEW, lw=1))

    _style_ax(ax, "Step 3:  OCTT  (the big change)", ylabel="OCTT  [0, 1]")
    ax.legend(fontsize=7.2, loc="lower left", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)

    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = out_dir / "pipeline_comparison_3panel.png"
    fig1.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {p1}")
    return p1


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — OCTT detail with structural annotations
# ══════════════════════════════════════════════════════════════════════════
def figure_octt_detail(c: Curves, out_dir: Path) -> Path:
    x01 = c.x01
    octt_o, octt_n = c.octt_o, c.octt_n
    x01_inflect_old, x01_inflect_new = c.x01_inflect_old, c.x01_inflect_new

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    fig2.patch.set_facecolor("white")

    ax2.plot(x01, octt_o, color=C_OLD, linewidth=2.8, label="Old pipeline", zorder=3)
    ax2.plot(x01, octt_n, color=C_NEW, linewidth=2.8, linestyle="--",
             label="Survey-calibrated (stickers)", zorder=4)
    ax2.fill_between(x01, octt_o, octt_n, alpha=0.15, color=C_NEW, zorder=2,
                     label="Difference (new − old)")

    ax2.axvline(x01_inflect_old, color=C_OLD, linewidth=1, linestyle=":", alpha=0.4)
    ax2.axvline(x01_inflect_new, color=C_NEW, linewidth=1, linestyle=":", alpha=0.4)
    ax2.axhline(0.5, color="#a0aec0", linewidth=0.7, linestyle=":", alpha=0.4)

    ax2.plot(x01_inflect_old, 0.5, "o", color=C_OLD, markersize=8, zorder=5,
             markeredgecolor="white", markeredgewidth=1.5)
    ax2.plot(x01_inflect_new, 0.5, "o", color=C_NEW, markersize=8, zorder=5,
             markeredgecolor="white", markeredgewidth=1.5)

    # annotate inflections
    ax2.annotate(f"Old inflection\nx = {x01_inflect_old:.2f}\narg = −2.5 + 0.2·t²",
                 xy=(x01_inflect_old, 0.5), xytext=(0.05, 0.22),
                 fontsize=9, color=C_OLD, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_OLD, lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", fc=C_OLD_LIGHT, ec=C_OLD, alpha=0.8))

    ax2.annotate(f"New inflection\nx = {x01_inflect_new:.2f}\narg = 5·(JE − 0.8)",
                 xy=(x01_inflect_new, 0.5), xytext=(0.62, 0.18),
                 fontsize=9, color=C_NEW, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_NEW, lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", fc=C_NEW_LIGHT, ec=C_NEW, alpha=0.8))

    # annotate structural difference
    ax2.text(0.55, 0.88,
             "Old: sigmoid driven by raw t² (minutes)\n"
             "New: sigmoid driven by JE (normalised index)\n\n"
             "Steepness:  K = 10 → a = 5  (halved)\n"
             f"Inflection:  x = {x01_inflect_old:.2f} → {x01_inflect_new:.2f}",
             fontsize=8.5, color=C_TEXT, va="top",
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cbd5e0", alpha=0.95))

    _style_ax(ax2, "OCTT:  Old Pipeline vs Survey Calibration (Detail)",
              ylabel="OCTT  [0, 1]")
    ax2.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.03, 1.05)

    p2 = out_dir / "pipeline_octt_detail.png"
    fig2.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {p2}")
    return p2


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Full chain overview (4 panels: t01, ATT, JE, OCTT)
# ══════════════════════════════════════════════════════════════════════════
def figure_full_chain(c: Curves, out_dir: Path) -> Path:
    x01 = c.x01
    att_o = c.att_o
    je_o, je_n = c.je_o, c.je_n
    octt_o, octt_n = c.octt_o, c.octt_n
    x01_inflect_old, x01_inflect_new = c.x01_inflect_old, c.x01_inflect_new

    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9.5))
    fig3.patch.set_facecolor("white")
    fig3.suptitle("Full Mapping Chain:  BPR travel time → ATT → JE → OCTT",
                  fontsize=13.5, fontweight="bold", color=C_TEXT, y=0.98)

    # a) Normalisation
    ax = axes3[0, 0]
    ax.plot(x01, x01, color=C_OLD, linewidth=2.2, label="x = normalised travel time")
    _style_ax(ax, "a)  Normalised travel time x ∈ [0, 1]", ylabel="x")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(0, 1)

    # b) ATT
    ax = axes3[0, 1]
    ax.plot(x01, att_o, color=C_OLD, linewidth=2.2, label="Old & New:  ATT = 1.0·x²")
    _style_ax(ax, "b)  ATT = b·x²   [b = 1.0 — identical]", ylabel="ATT")
    ax.text(0.5, 0.13, "No change", fontsize=14, color="#a0aec0",
            ha="center", fontstyle="italic", alpha=0.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)

    # c) JE
    ax = axes3[1, 0]
    ax.plot(x01, je_o, color=C_OLD, linewidth=2.2, label="Old:  JE = ATT  (a = 1.0)")
    ax.plot(x01, je_n, color=C_NEW, linewidth=2.2, linestyle="--",
            label="New: JE = 0.9·ATT  (a = 0.9)")
    ax.fill_between(x01, je_n, je_o, alpha=0.12, color=C_NEW)

    # mark the difference at x=1
    ax.annotate(f"Δ = {je_n[-1] - je_o[-1]:+.2f}",
                xy=(1.0, (je_o[-1] + je_n[-1]) / 2),
                xytext=(0.82, 0.7), fontsize=9, color=C_NEW, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_NEW, lw=1))

    _style_ax(ax, "c)  JE = a·ATT   [a: 1.0 → 0.9]", ylabel="JE")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)

    # d) OCTT
    ax = axes3[1, 1]
    ax.plot(x01, octt_o, color=C_OLD, linewidth=2.2, label="Old:  sigmoid of t² cost")
    ax.plot(x01, octt_n, color=C_NEW, linewidth=2.2, linestyle="--",
            label="New: sigmoid of JE (a=5, b=−0.8)")
    ax.fill_between(x01, octt_o, octt_n, alpha=0.12, color=C_NEW)

    ax.axvline(x01_inflect_old, color=C_OLD, linewidth=0.9, linestyle=":", alpha=0.5)
    ax.axvline(x01_inflect_new, color=C_NEW, linewidth=0.9, linestyle=":", alpha=0.5)
    ax.plot(x01_inflect_old, 0.5, "o", color=C_OLD, ms=6, zorder=5)
    ax.plot(x01_inflect_new, 0.5, "o", color=C_NEW, ms=6, zorder=5)

    ax.annotate(f"{x01_inflect_old:.2f}", xy=(x01_inflect_old, 0.5),
                xytext=(x01_inflect_old, 0.38),
                fontsize=8, color=C_OLD, ha="center", fontweight="bold")
    ax.annotate(f"{x01_inflect_new:.2f}", xy=(x01_inflect_new, 0.5),
                xytext=(x01_inflect_new, 0.38),
                fontsize=8, color=C_NEW, ha="center", fontweight="bold")

    _style_ax(ax, f"d)  OCTT sigmoid   [K: 10 → 5,  inflection: {x01_inflect_old:.2f} → {x01_inflect_new:.2f}]",
              ylabel="OCTT")
    ax.legend(fontsize=8.5, loc="center left")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)

    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    p3 = out_dir / "pipeline_full_chain.png"
    fig3.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {p3}")
    return p3


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Signed difference (new − old) for each variable
# ══════════════════════════════════════════════════════════════════════════
def figure_signed_difference(c: Curves, out_dir: Path) -> Path:
    x01 = c.x01
    att_o, att_n = c.att_o, c.att_n
    je_o, je_n = c.je_o, c.je_n
    octt_o, octt_n = c.octt_o, c.octt_n

    fig4, axes4 = plt.subplots(1, 3, figsize=(16, 4.5))
    fig4.patch.set_facecolor("white")
    fig4.suptitle("Signed Difference:  New (survey) − Old (octt_mapping.py)",
                  fontsize=13, fontweight="bold", color=C_TEXT, y=1.02)

    for ax, (label, d_old, d_new) in zip(axes4, [
        ("Δ ATT", att_o, att_n),
        ("Δ JE", je_o, je_n),
        ("Δ OCTT", octt_o, octt_n),
    ]):
        diff = d_new - d_old
        ax.fill_between(x01, 0, diff,
                        where=diff >= 0, color=C_NEW_LIGHT, alpha=0.7)
        ax.fill_between(x01, 0, diff,
                        where=diff < 0, color=C_OLD_LIGHT, alpha=0.7)
        ax.plot(x01, diff, color=C_TEXT, linewidth=1.8)
        ax.axhline(0, color="#718096", linewidth=0.8, linestyle="-")

        _style_ax(ax, label, ylabel="New − Old")
        ax.set_xlim(0, 1)

        # annotate peak
        idx_peak = np.argmax(np.abs(diff))
        peak_val = diff[idx_peak]
        peak_x = x01[idx_peak]
        ax.annotate(f"peak: {peak_val:+.3f}\nat x = {peak_x:.2f}",
                    xy=(peak_x, peak_val),
                    xytext=(peak_x + 0.12, peak_val * 0.7),
                    fontsize=8, color=C_NEW if peak_val > 0 else C_OLD,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->",
                                    color=C_NEW if peak_val > 0 else C_OLD, lw=1))

    fig4.tight_layout(rect=[0, 0, 1, 0.95])
    p4 = out_dir / "pipeline_difference.png"
    fig4.savefig(p4, dpi=180, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved: {p4}")
    return p4


# ── Summary table to stdout ──────────────────────────────────────────────
def print_summary() -> None:
    print()
    print("=" * 72)
    print("  Parameter Comparison Summary")
    print("=" * 72)
    print(f"{'Step':<22s}  {'Old pipeline':<24s}  {'Survey stickers':<20s}")
    print("-" * 72)
    print(f"{'1. ATT = b·x²':<22s}  {'b = 1.0 (ATT_POWER=2)':<24s}  {'b = 1.0':<20s}  SAME")
    print(f"{'2. JE = a·ATT':<22s}  {'a = 1.0 (JE_POWER=1)':<24s}  {'a = 0.9':<20s}  −10%")
    print(f"{'3. OCTT sigmoid':<22s}  {'K=10, input=t² cost':<24s}  {'a=5, input=JE':<20s}  DIFFERENT")
    print(f"{'   steepness':<22s}  {'K = 10':<24s}  {'a = 5':<20s}  halved")
    print(f"{'   inflection':<22s}  {'x = 0.35':<24s}  {'x = 0.94':<20s}  +0.59")
    print(f"{'   sigmoid input':<22s}  {'f(t²) (raw minutes)':<24s}  {'JE (normalised)':<20s}  structural")
    print("=" * 72)


def main(out_dir: Path | str | None = None) -> None:
    import warnings
    warnings.filterwarnings("ignore")

    if out_dir is None:
        out_dir = fc.data_path("plots", "function_shape_comparison")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = compute_curves()
    figure_three_panel(curves, out_dir)
    figure_octt_detail(curves, out_dir)
    figure_full_chain(curves, out_dir)
    figure_signed_difference(curves, out_dir)
    print_summary()


if __name__ == "__main__":
    main()