from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from no2d_code.core.filepath_configs import (
    EDGES_CSV,
    DEMAND_CSV,
    od_list_filename,
    OUT_LOG_TXT,
    ALL_CRIT_CSV,
    BEST_CRIT_CSV,
    input_path,
    output_path,
    outputs_dir,
    log_path,
    UE_RESULTS_FILE,
)
from no2d_code.solver.frank_wolfe_classes import FWResult


def load_edges(parent_dir: str) -> pd.DataFrame:
    return pd.read_csv(input_path(parent_dir, EDGES_CSV))


def load_od_list(parent_dir: str, tol: float) -> np.ndarray:
    filename = od_list_filename(tol)
    return np.loadtxt(input_path(parent_dir, filename), delimiter=",", skiprows=1)


def load_demand(parent_dir: str) -> np.ndarray:
    return np.loadtxt(input_path(parent_dir, DEMAND_CSV), delimiter=",", skiprows=1)


def load_filtered_od_and_demand(parent_dir: str, tol: float) -> tuple[np.ndarray, np.ndarray]:
    OD_list = load_od_list(parent_dir, tol)
    demand = load_demand(parent_dir)

    inds = np.where(OD_list[:, 0] == OD_list[:, 1])[0]
    if inds.size > 0:
        OD_list = np.delete(OD_list, inds, axis=0)
        demand = np.delete(demand, inds, axis=0)

    return OD_list, demand


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
    result: "FWResult"):
    os.makedirs(outputs_dir(parent_dir), exist_ok=True)

    path = output_path(parent_dir, UE_RESULTS_FILE)
    np.savez_compressed(
        path,
        UEflows=np.asarray(UEflows),
        UEflowsBest=np.asarray(UEflowsBest),
        crit=np.array([result.crit1, result.crit2], dtype=np.float64),
        crit_best=np.array([result.crit1_best, result.crit2_best], dtype=np.float64),
        L=np.array([result.iterations], dtype=np.int64),
        L_best=np.array([result.iter_best], dtype=np.int64)
    )


def ue_cache_pickle_path(parent_dir: str, tag: str) -> str:
    os.makedirs(outputs_dir(parent_dir), exist_ok=True)
    return output_path(parent_dir, f"ue_cache_{tag}.pkl")


def save_ue_cache_pickle(
    parent_dir: str,
    tag: str,
    *,
    UEflows_col: np.ndarray,
    UEflowsBest_col: np.ndarray,
    result: "FWResult",
    meta: Optional[dict] = None,
) -> None:
    path = ue_cache_pickle_path(parent_dir, tag)

    payload = {
        "UEflows_col": np.asarray(UEflows_col),
        "UEflowsBest_col": np.asarray(UEflowsBest_col),
        "result": result,
        "meta": {} if meta is None else dict(meta),
        "created_at": datetime.now().isoformat(),
        "tag": tag,
    }

    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_ue_cache_pickle(parent_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray, "FWResult", dict]:
    path = ue_cache_pickle_path(parent_dir, tag)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    UEflows_col = np.asarray(payload["UEflows_col"])
    UEflowsBest_col = np.asarray(payload["UEflowsBest_col"])
    result = payload["result"]
    meta = payload.get("meta", {})
    return UEflows_col, UEflowsBest_col, result, meta


def has_ue_cache_pickle(parent_dir: str, tag: str) -> bool:
    return os.path.exists(ue_cache_pickle_path(parent_dir, tag))
