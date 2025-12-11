from __future__ import annotations

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from no2d_code.frank_wolfe.filenames import (
    EDGES_CSV,
    NODES_CSV,
    DEMAND_CSV,
    od_list_filename,
    OUT_LOG_TXT,
    ALL_CRIT_CSV,
    BEST_CRIT_CSV,
    UE_FLOW_CSV,
    UE_FLOW_BEST_CSV,
    UE_CRIT_CSV,
    UE_CRIT_BEST_CSV,
    UE_L_CSV,
    UE_L_BEST_CSV,
    UE_LBD_CSV,
    UE_LBD_BEST_CSV,
    input_path,
    output_path,
    outputs_dir,
    log_path,
)
import os
import numpy as np


def load_edges(parent_dir: str) -> pd.DataFrame:
    return pd.read_csv(input_path(parent_dir, EDGES_CSV))


def load_nodes(parent_dir: str) -> pd.DataFrame:
    return pd.read_csv(input_path(parent_dir, NODES_CSV))


def load_od_list(parent_dir: str, tol: float) -> np.ndarray:
    filename = od_list_filename(tol)
    return np.loadtxt(input_path(parent_dir, filename), delimiter=",", skiprows=1)


def load_demand(parent_dir: str) -> np.ndarray:
    return np.loadtxt(input_path(parent_dir, DEMAND_CSV), delimiter=",", skiprows=1)


def init_ue_logs(parent_dir: str, steplimit: int) -> Tuple[str, str, str]:
    txt_name = log_path(parent_dir, OUT_LOG_TXT)
    with open(txt_name, "w", encoding="utf-8") as f:
        f.write(f"File created: {datetime.now().isoformat()}.\n")

    crit_log_name = log_path(parent_dir, ALL_CRIT_CSV)
    np.savetxt(
        crit_log_name,
        np.full((steplimit + 1, 2), np.inf, dtype=float),
        delimiter=",",
    )

    crit_bests_name = log_path(parent_dir, BEST_CRIT_CSV)
    np.savetxt(
        crit_bests_name,
        np.array([[np.inf, np.inf, np.inf]], dtype=float),
        delimiter=",",
    )

    return txt_name, crit_log_name, crit_bests_name


def save_ue_results(
    parent_dir: str,
    UEflows: np.ndarray,
    UEflowsBest: np.ndarray,
    crit1_UE: float,
    crit2_UE: float,
    crit1_UE_Best: float,
    crit2_UE_Best: float,
    L_UE: float,
    iter_UE: int,
    LBD_UE: float,
    LBD_UE_Best: float,
) -> None:
    os.makedirs(outputs_dir(parent_dir), exist_ok=True)

    np.savetxt(
        output_path(parent_dir, UE_FLOW_CSV),
        UEflows,
        delimiter=",",
    )
    np.savetxt(
        output_path(parent_dir, UE_FLOW_BEST_CSV),
        UEflowsBest,
        delimiter=",",
    )

    np.savetxt(
        output_path(parent_dir, UE_CRIT_CSV),
        np.array([crit1_UE, crit2_UE], dtype=float),
        delimiter=",",
    )
    np.savetxt(
        output_path(parent_dir, UE_CRIT_BEST_CSV),
        np.array([crit1_UE_Best, crit2_UE_Best], dtype=float),
        delimiter=",",
    )

    np.savetxt(
        output_path(parent_dir, UE_L_CSV),
        np.array([L_UE], dtype=float),
        delimiter=",",
    )
    np.savetxt(
        output_path(parent_dir, UE_L_BEST_CSV),
        np.array([iter_UE], dtype=float),
        delimiter=",",
    )
    np.savetxt(
        output_path(parent_dir, UE_LBD_CSV),
        np.array([LBD_UE], dtype=float),
        delimiter=",",
    )
    np.savetxt(
        output_path(parent_dir, UE_LBD_BEST_CSV),
        np.array([LBD_UE_Best], dtype=float),
        delimiter=",",
    )
