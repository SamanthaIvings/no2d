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

AMC_A = 1.0
AMC_B = 0.1
AMC_OCTT_SCALE = 100.0

CT_LOGIT_K = 0.01
CT_LOGIT_SHIFT = 500.0

BUS_SHARE = 0.1


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
        print("ATT_index pct:", _p(att))
        print("JE pct:", _p(je))
        print("term pct:", _p(term))
        print("arg pct:", _p(arg))
        print("OCTT pct:", _p(octt))

    return octt


def att_index_from_traveltime(traveltime_h: np.ndarray) -> np.ndarray:
    t = np.asarray(traveltime_h, dtype=float) * OCTT_TIME_SCALE

    denom = OCTT_T_MAX_REF - OCTT_T_MIN_REF
    if denom <= 0.0:
        raise ValueError("OCTT_T_MAX_REF must be > OCTT_T_MIN_REF")

    t01 = np.clip((t - OCTT_T_MIN_REF) / denom, 0.0, 1.0)
    return t01**ATT_POWER


def amc_from_octt(octt: np.ndarray) -> np.ndarray:
    o = np.asarray(octt, dtype=float)
    return AMC_A + AMC_B * ((o * AMC_OCTT_SCALE) ** 2)


def ct_share_from_amc(amc: np.ndarray) -> np.ndarray:
    a = np.asarray(amc, dtype=float)
    return 1.0 / (1.0 + np.exp(CT_LOGIT_K * (a - CT_LOGIT_SHIFT)))


def att_bu_from_octt(
    octt: np.ndarray,
    *,
    total_demand: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(octt, dtype=float)
    dem = np.asarray(total_demand, dtype=float)

    amc = amc_from_octt(o)
    ct_share = ct_share_from_amc(amc)

    ct = dem * ct_share
    bu = BUS_SHARE * dem - BUS_SHARE * ct
    att = dem - bu - ct

    return att, bu


def _p(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, [0, 1, 5, 50, 95, 99, 100])
