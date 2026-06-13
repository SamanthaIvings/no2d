"""Builds OCTT pipeline from raw FF Survey xlsx. Drop into no2d_code/frank_wolfe/."""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# x-values per number of survey cases
X_GRIDS = {2: [0.2, 1.0], 4: [0.2, 0.5, 0.7, 1.0], 5: [0.2, 0.4, 0.6, 0.8, 1.0]}

# (section_prefix, label)
STAGES = [("S1","Cong→ATT"), ("S2","ATT→JE"), ("S3","JE→OCTT"),
          ("S4","OCTT→AMC"), ("S5","AMC→Car"), ("S6","Car→Bus")]

# candidate models: (name, func, p0, min_points)
def _lin(x, a, b):        return a * x + b
def _poly(x, a, b, c):    return a * x**2 + b * x + c
def _sig(x, L, k, x0):    return L / (1.0 + np.exp(k * (x - x0)))

MODELS = [
    ("linear",     _lin,  [1.0, 50.0],      2),
    ("polynomial", _poly, [0.0, 1.0, 50.0], 3),
    ("sigmoid",    _sig,  [80.0, 4.0, 0.5], 4),
]


def _best_fit(x, y):
    ss_tot = np.sum((y - y.mean()) ** 2)
    best = None
    for name, fn, p0, min_pts in MODELS:
        if len(x) < min_pts:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=10_000)
            ss_res = np.sum((y - fn(x, *popt)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
            if best is None or r2 > best[2]:
                best = (name, fn, r2, popt)
        except (RuntimeError, ValueError):
            pass
    if best is None:
        raise RuntimeError(f"No model converged for x={x.tolist()}, y={y.tolist()}")
    return best


class SurveyOCTTPipeline:

    TIME_SCALE = 60.0
    T_MAX_REF = 10.0

    def __init__(self, xlsx_path: str, x_grids=None, sheet_name=0):
        xg = x_grids or X_GRIDS
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

        self._fns = []
        self._ranges = []
        self._info = []

        for prefix, label in STAGES:
            cols = sorted([c for c in df.columns if c.startswith(prefix + "_")],
                          key=lambda c: int(c.rsplit("Case", 1)[1]))
            n = len(cols)
            if n not in xg:
                raise ValueError(f"{prefix}: {n} cases, no x-grid defined")

            x = np.array(xg[n])
            y = np.nanmean(df[cols].dropna(how="all").values.astype(float), axis=0)
            name, fn, r2, params = _best_fit(x, y)

            # normalisation range over [0,1]
            yd = fn(np.linspace(0, 1, 500), *params)
            lo, hi = float(yd.min()), float(yd.max())
            if hi - lo < 1e-6:
                lo, hi = 0.0, 100.0

            self._fns.append(lambda x, _f=fn, _p=params: _f(x, *_p))
            self._ranges.append((lo, hi))
            self._info.append((prefix, label, name, r2, params, x, y, df[cols].shape[0]))

    def _t01(self, tt_h):
        return np.clip(np.asarray(tt_h, float) * self.TIME_SCALE / self.T_MAX_REF, 0, 1)

    def _stage(self, x, i):
        lo, hi = self._ranges[i]
        return np.clip((self._fns[i](x) - lo) / (hi - lo), 0, 1)

    def full_pipeline(self, tt_h, *, debug=False):
        x = self._t01(tt_h)
        out = {"t01": x}
        for i, key in enumerate(["att", "je", "octt", "amc", "car_trips", "bus_use"]):
            x = self._stage(x, i)
            out[key] = x
            if debug:
                p, lbl, nm, r2, *_ = self._info[i]
                print(f"  {p} {nm} R²={r2:.4f}  [{x.min():.3f}, {x.max():.3f}]")
        return out

    def octt_from_traveltime(self, tt_h, *, debug=False):
        return self.full_pipeline(tt_h, debug=debug)["octt"]

    def att_bu_from_octt(self, octt, *, total_demand=1.0):
        d = np.asarray(total_demand, float)
        amc = self._stage(np.asarray(octt, float), 3)
        car = self._stage(amc, 4)
        bus = self._stage(car, 5)
        return d * (1 - car) * (1 - bus), d * bus

    def print_report(self):
        for prefix, label, name, r2, params, x, y, n_resp in self._info:
            flag = "★" if r2 >= 0.90 else "▲"
            print(f"{flag} {prefix} {label}  n={n_resp}  {name} R²={r2:.4f}  "
                  f"params={np.round(params, 4).tolist()}")


if __name__ == "__main__":
    import sys
    pipe = SurveyOCTTPipeline(sys.argv[1])
    pipe.print_report()
    tt = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.30])
    s = pipe.full_pipeline(tt)
    print(f"\n{'min':>5} {'t01':>5} {'att':>5} {'je':>5} {'octt':>5} {'amc':>5} {'car':>5} {'bus':>5}")
    for i in range(len(tt)):
        print(f"{tt[i]*60:5.1f} {s['t01'][i]:5.3f} {s['att'][i]:5.3f} {s['je'][i]:5.3f} "
              f"{s['octt'][i]:5.3f} {s['amc'][i]:5.3f} {s['car_trips'][i]:5.3f} {s['bus_use'][i]:5.3f}")