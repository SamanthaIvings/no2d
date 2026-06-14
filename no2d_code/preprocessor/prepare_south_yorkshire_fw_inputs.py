from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYMCA_DIR = PROJECT_ROOT / "data" / "inputs" / "symca_OD_2019_230523"
OD_KEY_CSV = SYMCA_DIR / "OD_Region_Key.csv"
OD_MATRIX_CSV = SYMCA_DIR / "OD_WD_AM_PEAK_Local_Authority_south_yorkshire_only.csv"

GEO_DIR = PROJECT_ROOT / "data" / "inputs" / "geo"
LSOA_POP_CSV = GEO_DIR / "lsoa_population_density_mid2022.csv"
LSOA_PWC_CSV = GEO_DIR / "LSOA_PopCentroids_EW_2021.csv"
LSOA11_TO_LSOA21_CSV = GEO_DIR / "LSOA_2011_to_LSOA_2021.csv"
LSOA11_TO_MSOA11_CSV = GEO_DIR / "LSOA_to_MSOA_2011.csv"

NETWORK_DIR = PROJECT_ROOT / "data" / "inputs" / "osm_south_yorkshire"
NODES_PKL = NETWORK_DIR / "nodes_df.pkl"
EDGES_CAR_PKL = NETWORK_DIR / "edges_car.pkl"

OUT_DIR = PROJECT_ROOT / "data" / "inputs" / "fw_inputs_sy"
EDGES_CSV = OUT_DIR / "edges.csv"
NODES_CSV = OUT_DIR / "nodes.csv"
DEMAND_CSV = OUT_DIR / "demand.csv"
OD_LIST_CSV = OUT_DIR / "OD_list.csv"
ZONE_INDEX_CSV = OUT_DIR / "zone_index.csv"

RNG_SEED = 7
ZONE_NODE_SAMPLES = 800
MAX_SAMPLES_PER_OD = 10
ZONE_INDEX_BASE = 0


@dataclass(frozen=True)
class ZoneSampler:
    node_ids: np.ndarray


def _read_od_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    idx_col = df.columns[0]
    m = df.set_index(idx_col)
    m.index = m.index.astype(str)
    m.columns = m.columns.astype(str)

    if list(m.columns) != list(m.index):
        m = m.reindex(columns=list(m.index))
        if m.isna().any().any():
            raise ValueError("OD matrix columns do not match index labels.")
    return m


def _to_int_matrix(m: pd.DataFrame) -> np.ndarray:
    a = m.apply(pd.to_numeric, errors="coerce").fillna(0)
    a_int = a.round().astype(np.int64)
    return a_int.to_numpy()


def _read_od_key_msoa_map(path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    required = {"ODRegion", "MSOA11CD"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OD key missing columns: {sorted(missing)}")

    df = df.loc[df["MSOA11CD"].notna()].copy()
    df["MSOA11CD"] = df["MSOA11CD"].astype(str).str.strip()
    df = df.loc[df["MSOA11CD"].ne("") & df["MSOA11CD"].ne("Blank")]

    out: Dict[str, List[str]] = {}
    for od_region, sub in df.groupby("ODRegion"):
        codes = sorted(set(sub["MSOA11CD"].tolist()))
        out[str(od_region)] = codes
    return out


def _read_lsoa11_to_msoa11(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        usecols=["LSOA11CD", "MSOA11CD"],
        low_memory=False,
    )
    df = df.dropna()
    df["LSOA11CD"] = df["LSOA11CD"].astype(str).str.strip()
    df["MSOA11CD"] = df["MSOA11CD"].astype(str).str.strip()
    df = df.loc[df["LSOA11CD"].ne("") & df["MSOA11CD"].ne("")]
    return df.drop_duplicates().reset_index(drop=True)


def _read_lsoa11_to_lsoa21(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        usecols=["LSOA11CD", "LSOA21CD"],
        low_memory=False,
    )
    df = df.dropna()
    df["LSOA11CD"] = df["LSOA11CD"].astype(str).str.strip()
    df["LSOA21CD"] = df["LSOA21CD"].astype(str).str.strip()
    df = df.loc[df["LSOA11CD"].ne("") & df["LSOA21CD"].ne("")]
    return df.drop_duplicates().reset_index(drop=True)


def _read_lsoa21_centroids(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    required = {"LSOA21CD", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Centroids file missing columns: {sorted(missing)}")

    out = df.loc[:, ["LSOA21CD", "x", "y"]].copy()
    out["LSOA21CD"] = out["LSOA21CD"].astype(str).str.strip()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna()
    return out


def _read_lsoa21_population(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        usecols=["LSOA 2021 Code", "Total"],
        low_memory=False,
    )
    out = df.rename(columns={"LSOA 2021 Code": "LSOA21CD"}).copy()
    out["LSOA21CD"] = out["LSOA21CD"].astype(str).str.strip()
    out["Total"] = out["Total"].astype(str).str.replace(",", "", regex=False)
    out["pop"] = pd.to_numeric(out["Total"], errors="coerce")
    out = out.dropna(subset=["LSOA21CD", "pop"])
    out = out.loc[out["LSOA21CD"].ne("") & out["pop"].gt(0)]
    return out.loc[:, ["LSOA21CD", "pop"]]


def _build_msoa11_to_lsoa21(
    lsoa11_to_msoa11: pd.DataFrame,
    lsoa11_to_lsoa21: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    merged = lsoa11_to_msoa11.merge(lsoa11_to_lsoa21, on="LSOA11CD", how="inner")
    merged = merged.loc[:, ["MSOA11CD", "LSOA21CD"]].drop_duplicates()

    out: Dict[str, np.ndarray] = {}
    for msoa, sub in merged.groupby("MSOA11CD"):
        out[str(msoa)] = sub["LSOA21CD"].to_numpy(dtype=str)
    return out


def _build_zone_lsoa21_table(
    zone_labels: List[str],
    od_to_msoa11: Dict[str, List[str]],
    msoa11_to_lsoa21: Dict[str, np.ndarray],
    lsoa21_xy: pd.DataFrame,
    lsoa21_pop: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    pop_xy = lsoa21_xy.merge(lsoa21_pop, on="LSOA21CD", how="inner").drop_duplicates("LSOA21CD")

    out: Dict[str, pd.DataFrame] = {}
    for z in zone_labels:
        msoas = od_to_msoa11.get(z, [])
        if not msoas:
            raise ValueError(f"No MSOA11 codes for OD zone: {z}")

        lsoa_codes: List[str] = []
        for msoa in msoas:
            lsoa_codes.extend(msoa11_to_lsoa21.get(msoa, np.array([], dtype=str)).tolist())

        if not lsoa_codes:
            raise ValueError(f"No LSOA21 codes resolved for OD zone: {z}")

        sub = pop_xy.loc[pop_xy["LSOA21CD"].isin(set(lsoa_codes))].copy()
        if sub.empty:
            raise ValueError(f"No centroid+population records for OD zone: {z}")

        out[z] = sub.reset_index(drop=True)

    return out


def _load_network_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    required = {"node", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"nodes_df.pkl missing columns: {sorted(missing)}")
    return df.loc[:, ["node", "x", "y"]].copy()


def _snap_points_to_nodes(nodes_df: pd.DataFrame, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords = nodes_df.loc[:, ["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    dists, idx = tree.query(np.column_stack([xs, ys]), k=1)
    node_ids = nodes_df["node"].to_numpy(dtype=int)[idx]
    return node_ids, dists


def _build_zone_samplers(
    zone_tables: Dict[str, pd.DataFrame],
    nodes_df: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[Dict[str, ZoneSampler], pd.DataFrame]:
    rows = []
    samplers: Dict[str, ZoneSampler] = {}

    for zone, tab in zone_tables.items():
        w = tab["pop"].to_numpy(dtype=float)
        w_sum = float(w.sum())
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError(f"Non-positive weight sum for zone: {zone}")

        probs = w / w_sum
        idx = rng.choice(np.arange(tab.shape[0]), size=ZONE_NODE_SAMPLES, replace=True, p=probs)
        xs = tab["x"].to_numpy(dtype=float)[idx]
        ys = tab["y"].to_numpy(dtype=float)[idx]

        node_ids, dists = _snap_points_to_nodes(nodes_df, xs, ys)
        samplers[zone] = ZoneSampler(node_ids=node_ids.astype(int))

        rows.append(
            {
                "zone": zone,
                "n_samples": int(ZONE_NODE_SAMPLES),
                "n_unique_nodes": int(np.unique(node_ids).size),
                "snap_dist_mean_m": float(np.mean(dists)),
                "snap_dist_p95_m": float(np.percentile(dists, 95)),
            }
        )

    return samplers, pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def _expand_od_to_lists(
    demand_mat: np.ndarray,
    zone_labels: List[str],
    samplers: Dict[str, ZoneSampler],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    nz = np.argwhere(demand_mat > 0)
    od_rows: List[Tuple[int, int, int, int]] = []
    demands: List[int] = []

    for r, c in nz:
        d = int(demand_mat[r, c])
        if d <= 0:
            continue

        k = min(MAX_SAMPLES_PER_OD, d)
        z_o = zone_labels[r]
        z_d = zone_labels[c]

        o_nodes = samplers[z_o].node_ids
        d_nodes = samplers[z_d].node_ids

        o_pick = rng.choice(o_nodes, size=k, replace=True)
        d_pick = rng.choice(d_nodes, size=k, replace=True)

        base = d // k
        rem = d - base * k

        for i in range(k):
            di = base + (1 if i < rem else 0)
            if di <= 0:
                continue
            od_rows.append((int(r + ZONE_INDEX_BASE), int(c + ZONE_INDEX_BASE), int(o_pick[i]), int(d_pick[i])))
            demands.append(int(di))

    return np.array(od_rows, dtype=int), np.array(demands, dtype=int)


def _export_network_csvs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_pickle(NODES_PKL)
    edges = pd.read_pickle(EDGES_CAR_PKL)

    nodes.loc[:, ["node", "x", "y"]].to_csv(NODES_CSV, index=False)
    edges.to_csv(EDGES_CSV, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _export_network_csvs(OUT_DIR)

    od = _read_od_matrix(OD_MATRIX_CSV)
    zone_labels = od.index.tolist()
    demand_mat = _to_int_matrix(od)

    pd.DataFrame({"zone_index": np.arange(len(zone_labels), dtype=int) + ZONE_INDEX_BASE, "zone": zone_labels}).to_csv(
        ZONE_INDEX_CSV, index=False
    )

    od_to_msoa11 = _read_od_key_msoa_map(OD_KEY_CSV)
    lsoa11_to_msoa11 = _read_lsoa11_to_msoa11(LSOA11_TO_MSOA11_CSV)
    lsoa11_to_lsoa21 = _read_lsoa11_to_lsoa21(LSOA11_TO_LSOA21_CSV)

    msoa11_to_lsoa21 = _build_msoa11_to_lsoa21(lsoa11_to_msoa11, lsoa11_to_lsoa21)

    lsoa21_xy = _read_lsoa21_centroids(LSOA_PWC_CSV)
    lsoa21_pop = _read_lsoa21_population(LSOA_POP_CSV)

    zone_tables = _build_zone_lsoa21_table(zone_labels, od_to_msoa11, msoa11_to_lsoa21, lsoa21_xy, lsoa21_pop)

    nodes_df = _load_network_nodes(NODES_PKL)

    rng = np.random.default_rng(RNG_SEED)
    samplers, snap_report = _build_zone_samplers(zone_tables, nodes_df, rng)
    snap_report.to_csv(OUT_DIR / "snap_report.csv", index=False)

    od_list, demand_list = _expand_od_to_lists(demand_mat, zone_labels, samplers, rng)

    pd.DataFrame(od_list, columns=["o_zone", "d_zone", "o_node", "d_node"]).to_csv(OD_LIST_CSV, index=False)
    pd.DataFrame({"demand": demand_list}).to_csv(DEMAND_CSV, index=False)

    print(f"Zones: {len(zone_labels)}")
    print(f"OD_list rows: {od_list.shape[0]}")
    print(f"Total demand (expanded): {int(demand_list.sum())}")
    print(f"Saved: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
