from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class FWRunConfig:
    eps: float
    stepbreak: int
    txt_name: str
    crit_log_name: str
    crit_bests_name: str


@dataclass
class FWResult:
    flows: NDArray[np.float64]
    flows_best: NDArray[np.float64]
    crit1: float
    crit2: float
    crit1_best: float
    crit2_best: float
    iterations: int
    iter_best: int
    Xa_Gi: NDArray[np.float64]
    crit_log: NDArray[np.float64]
    crit_bests: NDArray[np.float64]
    LBD: float
    LBD_best: float
