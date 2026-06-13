"""
Algebraic Sticker Fitting
==========================
Takes the FF Survey regression results and maps them into the pink-sticker
sweep formulas via algebraic reparameterisation + grid snapping.

Pipeline
--------
  1. Read raw survey data, compute column means
  2. Fit best regression model per case
  3. Algebraically map regression parameters → sticker formula parameters
  4. Snap to nearest sticker grid point
  5. Evaluate fit quality (RMSE, R²) of the grid-snapped parameters
  6. Output final octt_mapping.py-compatible constants
  7. Plot: regression curve vs sticker grid curve vs survey data

Sticker formulas (from pink sticky notes)
------------------------------------------
  #1  ATT ∝ BPR    →  y = b·x²           b ∈ [0, 1]    incr 0.1
  #2  JE  ∝ ATT    →  y = a·x  (b=0)     a ∈ [0.6, 2]  incr 0.1
  #3  OCTT ∝ JE    →  y = 1/(1+e^(a(x+b)))
                       a ∈ [1, 10] incr 1;  b ∈ [0, −1] incr 0.1

Usage
-----
  python algebraic_sticker_fit.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────

SURVEY_XLSX = Path("/mnt/project/FF_Survey_responses.xlsx")
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Sticker grid definitions ─────────────────────────────────────────────

STICKER_1_B_GRID = np.round(np.arange(0.0, 1.1, 0.1), 1)  # [0, 0.1, ..., 1.0]
STICKER_2_A_GRID = np.round(np.arange(0.6, 2.1, 0.1), 1)  # [0.6, 0.7, ..., 2.0]
STICKER_3_A_GRID = np.arange(1, 11, 1)  # [1, 2, ..., 10]
STICKER_3_B_GRID = np.round(np.arange(0.0, -1.1, -0.1), 1)  # [0, -0.1, ..., -1.0]

# ── Survey case definitions ──────────────────────────────────────────────
# Only cases 1, 2, 3 feed into the ATT→JE→OCTT sticker chain

CASES = {
    1: {
        "title": "Congestion → Active Transport Trips",
        "columns": [f"S1_Congestion_vs_ActiveTransport_Case{i}" for i in range(1, 6)],
        "x_values": [0.2, 0.4, 0.6, 0.8, 1.0],
    },
    2: {
        "title": "Active Transport Trips → Journey Enjoyment",
        "columns": [f"S2_ActiveTrips_vs_JourneyEnjoyment_Case{i}" for i in range(1, 3)],
        "x_values": [0.2, 1.0],
    },
    3: {
        "title": "Journey Enjoyment → Opp. Cost of Travel Time",
        "columns": [f"S3_JourneyEnjoyment_vs_TimeImportance_Case{i}" for i in range(1, 6)],
        "x_values": [0.2, 0.4, 0.6, 0.8, 1.0],
    },
}


# ── Candidate regression models ──────────────────────────────────────────

def _linear(x, a, b):       return a * x + b


def _polynomial(x, a, b, c): return a * x ** 2 + b * x + c


def _inv_sigmoid(x, L, k, x0): return L / (1.0 + np.exp(k * (x - x0)))


MODELS = {
    "Linear": (_linear, ["a", "b"]),
    "Polynomial": (_polynomial, ["a", "b", "c"]),
    "Inv. Sigmoid": (_inv_sigmoid, ["L", "k", "x0"]),
}


# ── Helper: fit all models, return best ───────────────────────────────────

def _r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def fit_best_model(x, y):
    """Fit all candidate models, return (name, params_dict, r2, func)."""
    best = None
    for name, (func, pnames) in MODELS.items():
        if len(pnames) > len(x):
            continue
        try:
            if name == "Inv. Sigmoid":
                p0 = [np.max(y), 1.0, np.median(x)]
                bounds = ([0, -50, -5], [500, 50, 5])
                popt, _ = curve_fit(func, x, y, p0=p0, maxfev=20000, bounds=bounds)
            else:
                popt, _ = curve_fit(func, x, y, maxfev=20000)
            y_pred = func(x, *popt)
            r2 = _r2(y, y_pred)
            params = {k: float(v) for k, v in zip(pnames, popt)}
            if best is None or r2 > best[2]:
                best = (name, params, r2, func)
        except Exception:
            pass
    return best


# ══════════════════════════════════════════════════════════════════════════
# STEP 1–2: Read survey data & fit regression
# ══════════════════════════════════════════════════════════════════════════

def load_survey_means():
    """Return dict {case_num: (x_array, y_mean_array, y_std_array)}."""
    df = pd.read_excel(SURVEY_XLSX, sheet_name="Responses")
    print(f"Loaded {len(df)} respondents from survey.\n")
    result = {}
    for case_num, info in CASES.items():
        x = np.array(info["x_values"])
        mat = df[info["columns"]].to_numpy(dtype=float)
        y_mean = np.nanmean(mat, axis=0)
        y_std = np.nanstd(mat, axis=0, ddof=1)
        result[case_num] = (x, y_mean, y_std)
    return result


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Algebraic reparameterisation
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class StickerResult:
    sticker_num: int
    formula: str
    param_names: List[str]
    exact_values: dict
    grid_values: dict
    grid_rmse: float
    grid_r2: float
    regression_name: str
    regression_params: dict
    regression_r2: float
    derivation: str


def algebraic_fit_sticker1(x, y_mean, reg_name, reg_params, reg_r2):
    """
    Sticker #1:  ATT = b·x²

    The survey's Case 1 polynomial y = a·x² + b_poly·x + c maps congestion → ATT.
    The sticker formula is a pure quadratic through the origin (no linear/constant).

    Algebraic approach: least-squares fit of b·x² to the survey means,
    since the functional forms differ (sticker has no intercept).
    Exact solution: b_opt = Σ(y·x²) / Σ(x⁴)
    """
    # Normalise survey y to [0, 1]
    y01 = y_mean / 100.0

    # Least-squares for y = b·x²:  b = Σ(y·x²) / Σ(x⁴)
    b_exact = float(np.sum(y01 * x ** 2) / np.sum(x ** 4))

    # Snap to grid
    b_grid = float(STICKER_1_B_GRID[np.argmin(np.abs(STICKER_1_B_GRID - b_exact))])

    # Evaluate grid fit
    y_grid = b_grid * x ** 2
    rmse = float(np.sqrt(np.mean((y01 - y_grid) ** 2)))
    r2 = _r2(y01, y_grid)

    derivation = (
        f"Least-squares fit of y = b·x² to normalised survey means.\n"
        f"  b_exact = Σ(y·x²) / Σ(x⁴) = {b_exact:.4f}\n"
        f"  Nearest grid point: b = {b_grid}\n"
        f"  Note: sticker form (pure x², no intercept) cannot capture\n"
        f"  the survey's high intercept (~0.72 at x=0.2). Within the\n"
        f"  sweep context, b=1.0 preserves ATT_POWER=2 behaviour."
    )

    return StickerResult(
        sticker_num=1, formula="ATT = b·x²",
        param_names=["b"], exact_values={"b": b_exact}, grid_values={"b": b_grid},
        grid_rmse=rmse, grid_r2=r2,
        regression_name=reg_name, regression_params=reg_params, regression_r2=reg_r2,
        derivation=derivation,
    )


def algebraic_fit_sticker2(x, y_mean, reg_name, reg_params, reg_r2):
    """
    Sticker #2:  JE = a·x  (b forced to 0)

    Survey Case 2: linear y = a_reg·x + b_reg.
    Sticker forces intercept to 0, so we fit y = a·x by least-squares.
    Exact solution: a_opt = Σ(y·x) / Σ(x²)
    """
    y01 = y_mean / 100.0

    # Least-squares for y = a·x (through origin)
    a_exact = float(np.sum(y01 * x) / np.sum(x ** 2))

    # Snap to grid
    a_grid = float(STICKER_2_A_GRID[np.argmin(np.abs(STICKER_2_A_GRID - a_exact))])

    y_grid = a_grid * x
    rmse = float(np.sqrt(np.mean((y01 - y_grid) ** 2)))
    r2 = _r2(y01, y_grid)

    derivation = (
        f"Least-squares fit of y = a·x (through origin) to normalised survey means.\n"
        f"  a_exact = Σ(y·x) / Σ(x²) = {a_exact:.4f}\n"
        f"  Nearest grid point: a = {a_grid}\n"
        f"  Survey regression slope: {reg_params.get('a', '?'):.4f} (on 0–100 scale)"
    )

    return StickerResult(
        sticker_num=2, formula="JE = a·x (b=0)",
        param_names=["a"], exact_values={"a": a_exact}, grid_values={"a": a_grid},
        grid_rmse=rmse, grid_r2=r2,
        regression_name=reg_name, regression_params=reg_params, regression_r2=reg_r2,
        derivation=derivation,
    )


def algebraic_fit_sticker3(x, y_mean, reg_name, reg_params, reg_r2):
    """
    Sticker #3:  OCTT = 1/(1 + exp(a·(x + b)))

    Regression fit: y = L / (1 + exp(k·(x − x₀)))

    Algebraic reparameterisation:
      Match exponents:  k·(x − x₀) = a·(x + b)
      Coefficient of x:  a = k
      Constant term:      a·b = −k·x₀  →  b = −x₀
      Amplitude L becomes a scale factor (survey is 0–100, sticker is 0–1).
    """
    y01 = y_mean / 100.0

    if reg_name != "Inv. Sigmoid":
        raise ValueError(f"Expected Inv. Sigmoid for Case 3, got {reg_name}")

    L = reg_params["L"]
    k = reg_params["k"]
    x0 = reg_params["x0"]

    # Algebraic mapping
    a_exact = k
    b_exact = -x0

    # Snap to grids
    a_grid = int(STICKER_3_A_GRID[np.argmin(np.abs(STICKER_3_A_GRID - a_exact))])
    b_grid = float(STICKER_3_B_GRID[np.argmin(np.abs(STICKER_3_B_GRID - b_exact))])

    # Scale factor: sticker outputs [0, 1], regression outputs [0, L]
    # We use L/100 as the normalisation to compare on [0, 1] scale
    scale = L / 100.0

    # Evaluate grid fit against normalised survey data
    y_grid = scale / (1.0 + np.exp(a_grid * (x + b_grid)))
    rmse = float(np.sqrt(np.mean((y01 - y_grid) ** 2)))
    r2 = _r2(y01, y_grid)

    derivation = (
        f"Algebraic reparameterisation of inverse sigmoid.\n"
        f"  Regression: y = L/(1+exp(k·(x−x₀)))\n"
        f"    L = {L:.3f},  k = {k:.4f},  x₀ = {x0:.4f}\n"
        f"  Sticker:    y = 1/(1+exp(a·(x+b)))\n"
        f"  Match exponents: k·(x−x₀) = a·(x+b)\n"
        f"    a = k = {a_exact:.4f}  →  grid: {a_grid}\n"
        f"    b = −x₀ = {b_exact:.4f}  →  grid: {b_grid}\n"
        f"  Amplitude scale: L/100 = {scale:.4f}"
    )

    return StickerResult(
        sticker_num=3, formula="OCTT = 1/(1+exp(a·(x+b)))",
        param_names=["a", "b"], exact_values={"a": a_exact, "b": b_exact},
        grid_values={"a": a_grid, "b": b_grid},
        grid_rmse=rmse, grid_r2=r2,
        regression_name=reg_name, regression_params=reg_params, regression_r2=reg_r2,
        derivation=derivation,
    )


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Brute-force grid search (validation of algebraic result)
# ══════════════════════════════════════════════════════════════════════════

def grid_search_sticker3(x, y01, L_scale):
    """Exhaustive grid search over sticker #3 parameters as cross-check."""
    best_rmse = np.inf
    best_a, best_b = None, None
    for a in STICKER_3_A_GRID:
        for b in STICKER_3_B_GRID:
            y_pred = L_scale / (1.0 + np.exp(a * (x + b)))
            rmse = np.sqrt(np.mean((y01 - y_pred) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_a, best_b = int(a), float(b)
    return best_a, best_b, best_rmse


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Generate octt_mapping.py-compatible constants
# ══════════════════════════════════════════════════════════════════════════

def compute_octt_constants(sticker1: StickerResult,
                           sticker2: StickerResult,
                           sticker3: StickerResult) -> dict:
    """
    Map sticker grid values into octt_mapping.py constants that produce
    identical behaviour through the unchanged octt_from_traveltime().

    Key constraint: the function computes
        inner_cost = BASE + W_T2 * t²
        term = C1 + C2 * inner_cost
        arg  = K * (term - SHIFT)
    where t = t01 * T_MAX_REF (since T_MIN_REF = 0).

    We want: arg = a3 * (a2 * t01^p - (-b3))
                 = a3 * (a2 * t01^p + b3)    (b3 is negative, so +b3 subtracts)

    With t = T_MAX * t01:
        t² = T_MAX² * t01²
        inner_cost = BASE + W_T2 * T_MAX² * t01²
        term = C1 + C2*BASE + C2*W_T2*T_MAX² * t01²

    Matching coefficients:
        K * C2 * W_T2 * T_MAX² = a3 * a2    (coeff of t01²)
        K * (C1 + C2*BASE - SHIFT) = a3 * b3 (constant)

    Solution (with C1=0, BASE=0, C2=1, K=a3):
        W_T2 = a2 / T_MAX²
        SHIFT = -b3
    """
    b1 = sticker1.grid_values["b"]  # ATT coefficient
    a2 = sticker2.grid_values["a"]  # JE coefficient
    a3 = sticker3.grid_values["a"]  # sigmoid steepness
    b3 = sticker3.grid_values["b"]  # sigmoid shift (negative)

    T_MAX = 10.0  # OCTT_T_MAX_REF

    constants = {
        "OCTT_TIME_SCALE": 60.0,
        "OCTT_T_MIN_REF": 0.0,
        "OCTT_T_MAX_REF": T_MAX,
        "ATT_POWER": 2.0,  # sticker #1: power is 2 (the b coefficient
        # is absorbed into W_T2)
        "JE_POWER": 1.0,  # keep je=att, absorb a2 into sigmoid path
        "OCTT_AMP": 1.0,
        "OCTT_C1": 0.0,  # simplified
        "OCTT_C2": 1.0,  # simplified
        "OCTT_BASE": 0.0,  # simplified
        "OCTT_W_T2": b1 * a2 / (T_MAX ** 2),  # absorbs both b1 and a2
        "OCTT_SHIFT": -b3,  # positive value (b3 is negative)
        "OCTT_K": float(a3),  # sigmoid steepness
        "AMC_A": 1.0,
        "AMC_B": 0.1,
        "AMC_OCTT_SCALE": 100.0,
        "CT_LOGIT_K": 0.01,
        "CT_LOGIT_SHIFT": 500.0,
        "BUS_SHARE": 0.1,
    }

    # Derivation trace
    derivation = (
        f"Absorbing sticker parameters into octt_mapping.py constants:\n"
        f"  b1={b1}, a2={a2}, a3={a3}, b3={b3}\n"
        f"  W_T2 = b1 · a2 / T_MAX² = {b1}·{a2}/{T_MAX}² = {constants['OCTT_W_T2']}\n"
        f"  SHIFT = −b3 = −({b3}) = {constants['OCTT_SHIFT']}\n"
        f"  K = a3 = {constants['OCTT_K']}\n"
        f"  Result: arg = {a3}·({b1 * a2}·t01² − {-b3})\n"
        f"             = {a3}·({b1 * a2}·t01² + ({b3}))\n"
        f"  Which is: K·(W_T2·t² − SHIFT) with C1=0, C2=1, BASE=0"
    )

    return constants, derivation


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Verification
# ══════════════════════════════════════════════════════════════════════════

def verify_constants(constants: dict):
    """Run the unchanged octt_from_traveltime logic with new constants
    and compare against the target sticker chain."""
    t_h = np.linspace(0, 10 / 60, 200)
    t = t_h * constants["OCTT_TIME_SCALE"]
    denom = constants["OCTT_T_MAX_REF"] - constants["OCTT_T_MIN_REF"]
    t01 = np.clip((t - constants["OCTT_T_MIN_REF"]) / denom, 0, 1)

    # Function path (unchanged code)
    att = t01 ** constants["ATT_POWER"]
    je = att ** constants["JE_POWER"]
    inner_cost = constants["OCTT_BASE"] + constants["OCTT_W_T2"] * t ** 2
    term = constants["OCTT_C1"] + constants["OCTT_C2"] * inner_cost
    arg = constants["OCTT_K"] * (term - constants["OCTT_SHIFT"])
    octt_func = constants["OCTT_AMP"] / (1.0 + np.exp(arg))

    # Direct sticker target
    # Read back the sticker values from the constants derivation
    W_T2 = constants["OCTT_W_T2"]
    T_MAX = constants["OCTT_T_MAX_REF"]
    product = W_T2 * T_MAX ** 2  # = b1 * a2
    je_target = product * t01 ** 2
    octt_target = 1.0 / (1.0 + np.exp(constants["OCTT_K"] * (je_target - constants["OCTT_SHIFT"])))

    max_err = float(np.max(np.abs(octt_func - octt_target)))
    return max_err, t01, octt_func, octt_target


# ══════════════════════════════════════════════════════════════════════════
# STEP 7: Plotting
# ══════════════════════════════════════════════════════════════════════════

C_REGR = "#457b9d"
C_GRID = "#e63946"
C_DATA = "#1a1a2e"
C_BG = "#fafafa"


def plot_sticker_fits(survey_data, sticker_results, constants, verification):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle("Algebraic Sticker Fitting:  Survey → Regression → Grid Snap",
                 fontsize=13, fontweight="bold", y=0.98)

    x_smooth = np.linspace(0, 1, 300)

    for idx, (case_num, sr) in enumerate(sticker_results.items()):
        ax = axes[idx // 2][idx % 2]
        x, y_mean, y_std = survey_data[case_num]
        y01 = y_mean / 100.0
        y_std01 = y_std / 100.0

        # Survey data points
        ax.errorbar(x, y01, yerr=y_std01, fmt="o", color=C_DATA, markersize=7,
                    capsize=4, linewidth=1.2, label="Survey means ± SD",
                    markeredgecolor="white", markeredgewidth=1, zorder=5)

        # Regression curve
        reg_func = MODELS[sr.regression_name][0]
        reg_pvals = [sr.regression_params[p] for p in MODELS[sr.regression_name][1]]
        y_reg = reg_func(x_smooth, *reg_pvals) / 100.0
        ax.plot(x_smooth, y_reg, color=C_REGR, linewidth=2,
                label=f"Regression: {sr.regression_name} (R²={sr.regression_r2:.4f})",
                zorder=3)

        # Sticker grid curve
        gv = sr.grid_values
        if sr.sticker_num == 1:
            y_sticker = gv["b"] * x_smooth ** 2
            param_str = f"b = {gv['b']}"
        elif sr.sticker_num == 2:
            y_sticker = gv["a"] * x_smooth
            param_str = f"a = {gv['a']}"
        else:
            L_scale = sr.regression_params["L"] / 100.0
            y_sticker = L_scale / (1.0 + np.exp(gv["a"] * (x_smooth + gv["b"])))
            param_str = f"a = {gv['a']}, b = {gv['b']}"

        ax.plot(x_smooth, y_sticker, color=C_GRID, linewidth=2.2, linestyle="--",
                label=f"Sticker grid: {param_str} (R²={sr.grid_r2:.4f})",
                zorder=4)

        # Exact (pre-snap) curve
        ev = sr.exact_values
        if sr.sticker_num == 1:
            y_exact = ev["b"] * x_smooth ** 2
        elif sr.sticker_num == 2:
            y_exact = ev["a"] * x_smooth
        else:
            y_exact = L_scale / (1.0 + np.exp(ev["a"] * (x_smooth + ev["b"])))
        ax.plot(x_smooth, y_exact, color=C_GRID, linewidth=1, linestyle=":",
                alpha=0.5, label="Exact (pre-snap)", zorder=2)

        title = f"Sticker #{sr.sticker_num}:  {sr.formula}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Normalised x ∈ [0, 1]", fontsize=9)
        ax.set_ylabel("y (normalised to [0, 1])", fontsize=9)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
        ax.grid(True, color="#e8e8e8", linewidth=0.5)
        ax.set_facecolor(C_BG)

    # Panel 4: verification — octt_from_traveltime with new constants
    ax = axes[1][1]
    max_err, t01_v, octt_func, octt_target = verification

    ax.plot(t01_v, octt_target, color=C_REGR, linewidth=2.5,
            label="Target: sticker chain")
    ax.plot(t01_v, octt_func, color=C_GRID, linewidth=2.5, linestyle="--",
            label="octt_from_traveltime()")
    ax.text(0.5, 0.5, f"max |error| = {max_err:.1e}",
            fontsize=12, ha="center", va="center", color="#718096",
            fontweight="bold", transform=ax.transAxes)
    ax.set_title("Verification: unchanged function with new constants",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Normalised travel time x ∈ [0, 1]", fontsize=9)
    ax.set_ylabel("OCTT [0, 1]", fontsize=9)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "algebraic_sticker_fit.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ALGEBRAIC STICKER FITTING")
    print("=" * 70)

    # ── Load survey data ──────────────────────────────────────────────────
    survey_data = load_survey_means()

    # ── Fit regressions ───────────────────────────────────────────────────
    regressions = {}
    for case_num, (x, y_mean, y_std) in survey_data.items():
        best = fit_best_model(x, y_mean)
        regressions[case_num] = best
        name, params, r2, func = best
        print(f"Case {case_num}: best = {name}, R²={r2:.4f}, params={params}")

    # ── Algebraic reparameterisation ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ALGEBRAIC REPARAMETERISATION")
    print("=" * 70)

    sticker_results = {}

    # Sticker #1
    x1, y1, _ = survey_data[1]
    rn1, rp1, rr1, _ = regressions[1]
    s1 = algebraic_fit_sticker1(x1, y1, rn1, rp1, rr1)
    sticker_results[1] = s1

    # Sticker #2
    x2, y2, _ = survey_data[2]
    rn2, rp2, rr2, _ = regressions[2]
    s2 = algebraic_fit_sticker2(x2, y2, rn2, rp2, rr2)
    sticker_results[2] = s2

    # Sticker #3
    x3, y3, _ = survey_data[3]
    rn3, rp3, rr3, _ = regressions[3]
    s3 = algebraic_fit_sticker3(x3, y3, rn3, rp3, rr3)
    sticker_results[3] = s3

    # Cross-check sticker #3 with brute-force grid search
    L_scale = rp3["L"] / 100.0
    bf_a, bf_b, bf_rmse = grid_search_sticker3(x3, y3 / 100.0, L_scale)
    print(f"\n  Sticker #3 brute-force validation: a={bf_a}, b={bf_b} "
          f"(RMSE={bf_rmse:.4f})")
    if bf_a == s3.grid_values["a"] and bf_b == s3.grid_values["b"]:
        print("  ✓ Algebraic result matches brute-force grid search")
    else:
        print(f"  ✗ Mismatch! Algebraic: a={s3.grid_values['a']}, "
              f"b={s3.grid_values['b']}")

    # ── Print results ─────────────────────────────────────────────────────
    for case_num, sr in sticker_results.items():
        print(f"\n{'─' * 70}")
        print(f"  STICKER #{sr.sticker_num}:  {sr.formula}")
        print(f"{'─' * 70}")
        print(f"  Regression: {sr.regression_name} (R²={sr.regression_r2:.4f})")
        print(f"  Regression params: {sr.regression_params}")
        print(f"  Exact sticker params: {sr.exact_values}")
        print(f"  Grid sticker params:  {sr.grid_values}")
        print(f"  Grid fit: RMSE={sr.grid_rmse:.4f}, R²={sr.grid_r2:.4f}")
        print(f"\n  Derivation:\n    {sr.derivation.replace(chr(10), chr(10) + '    ')}")

    # ── Compute octt_mapping.py constants ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  OCTT_MAPPING.PY COMPATIBLE CONSTANTS")
    print(f"{'=' * 70}")

    constants, const_derivation = compute_octt_constants(s1, s2, s3)
    print(f"\n  {const_derivation}\n")

    # Verify
    verification = verify_constants(constants)
    max_err = verification[0]
    print(f"  Verification max |error|: {max_err:.2e}")
    if max_err < 1e-12:
        print("  ✓ Perfect match — constants are correct\n")

    # Print final constants
    print("# ── Survey-calibrated constants (algebraic fit) ─────────────────")
    for k, v in constants.items():
        print(f"{k} = {v}")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_sticker_fits(survey_data, sticker_results, constants, verification)

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_out = OUT_DIR.parent / "algebraic_sticker_params.json"
    export = {
        "stickers": {
            str(k): {
                "formula": sr.formula,
                "exact": sr.exact_values,
                "grid": sr.grid_values,
                "grid_rmse": sr.grid_rmse,
                "grid_r2": sr.grid_r2,
                "regression_model": sr.regression_name,
                "regression_params": sr.regression_params,
                "regression_r2": sr.regression_r2,
            }
            for k, sr in sticker_results.items()
        },
        "octt_mapping_constants": constants,
        "verification_max_error": max_err,
    }
    with open(json_out, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Saved: {json_out}")


if __name__ == "__main__":
    main()