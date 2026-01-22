from __future__ import annotations

import numpy as np


OCTT_TIME_SCALE = 60.0

OCTT_T_MIN_REF = 0.0
OCTT_T_MAX_REF = 10.0

ATT_POWER = 2.0
JE_POWER = 1.0

OCTT_AMP = 1.0

OCTT_C1 = 1.0
OCTT_C2 = 0.1
OCTT_BASE = 100.0
OCTT_W_T2 = 0.2

OCTT_SHIFT = 11.25
OCTT_K = 10.0


def octt_from_traveltime(traveltime_h: np.ndarray, *, debug: bool = False) -> np.ndarray:
    t = np.asarray(traveltime_h, dtype=float) * OCTT_TIME_SCALE

    denom = OCTT_T_MAX_REF - OCTT_T_MIN_REF
    if denom <= 0.0:
        raise ValueError("OCTT_T_MAX_REF must be > OCTT_T_MIN_REF")

    t01 = np.clip((t - OCTT_T_MIN_REF) / denom, 0.0, 1.0)
    att = t01**ATT_POWER
    je = att**JE_POWER

    inner_cost = OCTT_BASE + OCTT_W_T2 * (t**2)
    term = OCTT_C1 + OCTT_C2 * inner_cost
    arg = OCTT_K * (term - OCTT_SHIFT)
    octt = OCTT_AMP / (1.0 + np.exp(arg))

    if debug:
        print("traveltime_h pct:", _p(np.asarray(traveltime_h, dtype=float)))
        print("t_scaled pct:", _p(t))
        print("ATT pct:", _p(att))
        print("JE pct:", _p(je))
        print("term pct:", _p(term))
        print("arg pct:", _p(arg))
        print("OCTT pct:", _p(octt))

    return octt


def _p(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, [0, 1, 5, 50, 95, 99, 100])
