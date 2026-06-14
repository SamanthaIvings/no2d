from __future__ import annotations
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import warnings
import os

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "pdf.fonttype": 42,
})

CASES = [
    dict(n=1, cols=[f"S1_Congestion_vs_ActiveTransport_Case{i}" for i in range(1, 6)],
         x=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), model="poly", xlabel="Congestion", ylabel="Active Transport Trips"),
    dict(n=2, cols=[f"S2_ActiveTrips_vs_JourneyEnjoyment_Case{i}" for i in range(1, 3)],
         x=np.array([0.2, 1.0]), model="lin", xlabel="Active Transport Trips", ylabel="Journey Enjoyment"),
    dict(n=3, cols=[f"S3_JourneyEnjoyment_vs_TimeImportance_Case{i}" for i in range(1, 6)],
         x=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), model="sig_dec", xlabel="Journey Enjoyment", ylabel="Opportunity Cost of Travel Time"),
    dict(n=4, cols=[f"S4_TimeImportance_vs_AltModalChoices_Case{i}" for i in range(1, 6)],
         x=np.array([0.2, 0.4, 0.6, 0.8, 1.0]), model="poly", xlabel="Opportunity Cost of Travel Time", ylabel="Alternative Modal Choices"),
    dict(n=5, cols=[f"S5_AltModalChoices_vs_CarTrips_Case{i}" for i in range(1, 5)],
         x=np.array([0.2, 0.5, 0.7, 1.0]), model="sig_inc", xlabel="Alternative Modal Choices", ylabel="Car Trips"),
    dict(n=6, cols=[f"S6_CarTrips_vs_BusUse_Case{i}" for i in range(1, 3)],
         x=np.array([0.2, 1.0]), model="lin", xlabel="Car Trips", ylabel="Perception of Bus Congestion"),
]

def lin(x, a, b): return a * x + b
def poly(x, a, b, c): return a * x**2 + b * x + c
def sig(x, L, k, x0): return L / (1.0 + np.exp(k * (x - x0)))

def fit(x, y, model):
    y = np.clip(y, 0.01, 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "lin":
            p, _ = curve_fit(lin, x, y, maxfev=20000)
            return lin, p
        if model == "poly":
            p, _ = curve_fit(poly, x, y, maxfev=20000)
            return poly, p
        if model == "sig_dec":
            p, _ = curve_fit(sig, x, y, p0=[max(y)*1.1, 4.0, 0.7], bounds=([0,0.01,0],[200,50,2]), maxfev=20000)
            return sig, p
        p, _ = curve_fit(sig, x, y, p0=[max(y)*1.5, -2.0, 0.6], bounds=([0,-50,0],[300,-0.01,2]), maxfev=20000)
        return sig, p

XLSX = os.path.join(os.path.dirname(__file__), "..", "..", "data", "inputs", "FF_Survey_responses.xlsx")
if not os.path.exists(XLSX):
    XLSX = "FF_Survey_responses.xlsx"

df = pd.read_excel(XLSX, sheet_name="Responses")

COLORS = {"mean": "#2E5090", "plus_sd": "#C04040", "minus_sd": "#2E8B57"}
LABELS = {"mean": "Regression (mean)", "plus_sd": "Mean + 1 SD", "minus_sd": "Mean − 1 SD"}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("FF Survey — Regression vs ±1 SD Functional Form Comparison", fontsize=14, fontweight="bold", y=0.97)

for idx, case in enumerate(CASES):
    ax = axes[idx // 3, idx % 3]
    vals = df[case["cols"]].values.astype(float)
    means = np.nanmean(vals, axis=0)
    sds = np.nanstd(vals, axis=0, ddof=1)
    xf = np.linspace(case["x"].min(), case["x"].max(), 200)

    ax.fill_between(case["x"], means - sds, means + sds, alpha=0.15, color="#888888", label="±1 SD (data)")
    ax.errorbar(case["x"], means, yerr=sds, fmt="o", color="#555555", markersize=5, capsize=3, zorder=5, label="Mean ± SD")

    for tag, shift in [("mean", 0), ("plus_sd", +1), ("minus_sd", -1)]:
        target = np.clip(means + shift * sds, 0.0, 100.0)
        fn, p = fit(case["x"], target, case["model"])
        yf = fn(xf, *p)
        lw = 2.5 if tag == "mean" else 1.5
        ls = "-" if tag == "mean" else "--"
        ax.plot(xf, yf, color=COLORS[tag], lw=lw, ls=ls, label=LABELS[tag])

    ax.set_title(f"Case #{case['n']}", fontsize=11, fontweight="bold")
    ax.set_xlabel(case["xlabel"], fontsize=9)
    ax.set_ylabel(case["ylabel"], fontsize=9)
    ax.set_xlim(case["x"].min() - 0.02, case["x"].max() + 0.02)
    ax.tick_params(labelsize=8)
    if idx == 0:
        ax.legend(fontsize=7, loc="lower right")

fig.tight_layout(rect=[0, 0, 1, 0.94])

out = os.path.join(os.path.dirname(__file__), "..", "..", "data", "plots", "sd_simulations")
os.makedirs(out, exist_ok=True)
out_png = os.path.join(out, "ff_survey_sd_comparison.png")
try:
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")
except Exception:
    out_png = "ff_survey_sd_comparison.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")
plt.close(fig)