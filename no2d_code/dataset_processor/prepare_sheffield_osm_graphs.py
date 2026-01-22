from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import osmnx as ox


PLACE = "Sheffield, England, United Kingdom"
OUT_DIR = Path("data/osm_sheffield")
EPSG_UK = 27700


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    G = _download_drive_graph(PLACE)
    G = _project_graph(G, EPSG_UK)
    G, old_node_id_to_new_idx = _relabel_nodes_to_integers(G)

    nodes_df = _extract_nodes_df(G)
    edges_df = _extract_edges_df(G)

    edges_df = _add_infra_flags(edges_df)

    edges_car = edges_df.copy()
    edges_bus = _filter_bus_edges(edges_df)
    edges_bike = _filter_bike_edges(edges_df)

    _atomic_pickle_dump(OUT_DIR / "nodes_df.pkl", nodes_df)
    _atomic_pickle_dump(OUT_DIR / "old_node_id_to_new_idx.pkl", old_node_id_to_new_idx)

    _atomic_pickle_dump(OUT_DIR / "edges_car.pkl", edges_car)
    _atomic_pickle_dump(OUT_DIR / "edges_bus.pkl", edges_bus)
    _atomic_pickle_dump(OUT_DIR / "edges_bike.pkl", edges_bike)

    meta = _build_meta(
        place=PLACE,
        epsg=EPSG_UK,
        n_nodes=int(nodes_df.shape[0]),
        n_edges=int(edges_df.shape[0]),
    )
    _atomic_pickle_dump(OUT_DIR / "meta.pkl", meta)

    print(f"Saved to: {OUT_DIR.resolve()}")
    print(f"Nodes: {nodes_df.shape[0]}")
    print(f"Edges car/bus/bike: {edges_car.shape[0]} / {edges_bus.shape[0]} / {edges_bike.shape[0]}")


def _download_drive_graph(place: str) -> nx.MultiDiGraph:
    ox.settings.log_console = False
    return ox.graph_from_place(place, network_type="drive", simplify=True)


def _project_graph(G: nx.MultiDiGraph, epsg: int) -> nx.MultiDiGraph:
    return ox.project_graph(G, to_crs=f"EPSG:{epsg}")


def _relabel_nodes_to_integers(G: nx.MultiDiGraph) -> Tuple[nx.MultiDiGraph, Dict[int, int]]:
    old_nodes = list(G.nodes())
    mapping = {old: i for i, old in enumerate(old_nodes)}
    return nx.relabel_nodes(G, mapping, copy=True), mapping


def _extract_nodes_df(G: nx.MultiDiGraph) -> pd.DataFrame:
    rows = []
    for n, data in G.nodes(data=True):
        rows.append(
            {
                "node": int(n),
                "x": float(data.get("x", np.nan)),
                "y": float(data.get("y", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("node").reset_index(drop=True)


def _extract_edges_df(G: nx.MultiDiGraph) -> pd.DataFrame:
    gdf = ox.graph_to_gdfs(G, nodes=False, edges=True, fill_edge_geometry=False).reset_index()
    n = len(gdf)

    u = gdf["u"].to_numpy(dtype=int)
    v = gdf["v"].to_numpy(dtype=int)

    length_m = _safe_numeric_array(gdf.get("length"), n=n, default=1.0)

    highway = _col_str(gdf, "highway", n)
    maxspeed = _col_str(gdf, "maxspeed", n)
    lanes = _col_str(gdf, "lanes", n)

    speed_kmh = _infer_speed_kmh(highway=highway, maxspeed=maxspeed)
    capacity = _infer_capacity(lanes=lanes, highway=highway)
    critical_density = _infer_critical_density(highway=highway)

    df = pd.DataFrame(
        {
            "u": u,
            "v": v,
            "length": length_m,
            "speedlim": speed_kmh,
            "capacity": capacity,
            "criticalDensity": critical_density,
            "highway": highway,
            "maxspeed": maxspeed,
            "lanes": lanes,
            "access": _col_str(gdf, "access", n),
            "bicycle": _col_str(gdf, "bicycle", n),
            "bus": _col_str(gdf, "bus", n),
            "psv": _col_str(gdf, "psv", n),
            "motor_vehicle": _col_str(gdf, "motor_vehicle", n),
            "vehicle": _col_str(gdf, "vehicle", n),
            "busway": _col_str(gdf, "busway", n),
            "busway_left": _col_str(gdf, "busway:left", n),
            "busway_right": _col_str(gdf, "busway:right", n),
            "cycleway": _col_str(gdf, "cycleway", n),
            "cycleway_left": _col_str(gdf, "cycleway:left", n),
            "cycleway_right": _col_str(gdf, "cycleway:right", n),
        }
    )
    return df


def _col_str(gdf: pd.DataFrame, col: str, n: int) -> pd.Series:
    if col not in gdf.columns:
        return pd.Series(["nan"] * n, dtype=str)
    s = gdf[col]
    if isinstance(s, pd.Series):
        out = s.astype(str)
    else:
        out = pd.Series(s, dtype=str).astype(str)
    if len(out) != n:
        out = out.reindex(range(n)).fillna("nan").astype(str)
    return out


def _safe_numeric_array(s: Any, *, n: int, default: float) -> np.ndarray:
    if s is None:
        return np.full(n, default, dtype=float)
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if a.size != n:
        a2 = np.full(n, np.nan, dtype=float)
        m = min(n, a.size)
        a2[:m] = a[:m]
        a = a2
    a[~np.isfinite(a)] = default
    return a


def _add_infra_flags(edges: pd.DataFrame) -> pd.DataFrame:
    e = edges.copy()
    hw = e["highway"].astype(str).str.lower()

    cycle_tag = (
        e["cycleway"].astype(str).str.lower().ne("nan")
        | e["cycleway_left"].astype(str).str.lower().ne("nan")
        | e["cycleway_right"].astype(str).str.lower().ne("nan")
    )
    bike_designated = e["bicycle"].astype(str).str.lower().isin(["designated", "yes"])
    e["is_bike_infra"] = (hw.str.contains("cycleway", na=False) | cycle_tag | bike_designated).to_numpy(dtype=bool)

    bus_tag = (
        e["busway"].astype(str).str.lower().ne("nan")
        | e["busway_left"].astype(str).str.lower().ne("nan")
        | e["busway_right"].astype(str).str.lower().ne("nan")
        | hw.str.contains("busway", na=False)
    )
    psv_ok = e["psv"].astype(str).str.lower().isin(["yes", "designated"])
    bus_ok = e["bus"].astype(str).str.lower().isin(["yes", "designated"])
    e["is_bus_priority"] = (bus_tag | psv_ok | bus_ok).to_numpy(dtype=bool)

    return e


def _filter_bike_edges(edges: pd.DataFrame) -> pd.DataFrame:
    e = edges.copy()
    bike = e["bicycle"].astype(str).str.lower()
    access = e["access"].astype(str).str.lower()
    hw = e["highway"].astype(str).str.lower()

    disallowed = (bike == "no") | (access == "no")
    disallowed |= (hw.str.contains("motorway", na=False)) & (bike != "yes")

    return e.loc[~disallowed].reset_index(drop=True)


def _filter_bus_edges(edges: pd.DataFrame) -> pd.DataFrame:
    e = edges.copy()
    bus = e["bus"].astype(str).str.lower()
    psv = e["psv"].astype(str).str.lower()
    access = e["access"].astype(str).str.lower()

    disallowed = (access == "no") | (bus == "no") | (psv == "no")

    return e.loc[~disallowed].reset_index(drop=True)


def _infer_speed_kmh(*, highway: pd.Series, maxspeed: pd.Series) -> np.ndarray:
    hw = highway.astype(str).str.lower()
    ms = maxspeed.astype(str).str.lower()

    parsed = np.array([_parse_maxspeed_kmh(x) for x in ms.to_list()], dtype=float)
    defaults = np.array([_default_speed_for_highway(x) for x in hw.to_list()], dtype=float)

    out = defaults
    mask = np.isfinite(parsed)
    out[mask] = parsed[mask]
    return np.clip(out, 5.0, 130.0)


def _parse_maxspeed_kmh(s: str) -> float:
    if not s or s == "nan":
        return float("nan")
    nums = re.findall(r"\d+", s)
    if not nums:
        return float("nan")
    v = float(nums[0])
    return v * 1.609344 if "mph" in s else v


def _default_speed_for_highway(hw: str) -> float:
    if not hw or hw == "nan":
        return 30.0
    if "motorway" in hw:
        return 112.0
    if "trunk" in hw:
        return 80.0
    if "primary" in hw:
        return 60.0
    if "secondary" in hw:
        return 50.0
    if "tertiary" in hw:
        return 40.0
    if "residential" in hw:
        return 30.0
    if "service" in hw:
        return 20.0
    return 30.0


def _infer_capacity(*, lanes: pd.Series, highway: pd.Series) -> np.ndarray:
    hw = highway.astype(str).str.lower()
    ln = lanes.astype(str).str.lower()

    lanes_num = np.array([_parse_lanes(x) for x in ln.to_list()], dtype=float)
    defaults = np.array([_default_lanes_for_highway(x) for x in hw.to_list()], dtype=float)

    missing = ~np.isfinite(lanes_num)
    lanes_num[missing] = defaults[missing]
    lanes_num = np.clip(lanes_num, 1.0, 6.0)

    cap_per_lane = 1800.0
    return lanes_num * cap_per_lane


def _parse_lanes(s: str) -> float:
    if not s or s == "nan":
        return float("nan")
    nums = re.findall(r"\d+", s)
    if not nums:
        return float("nan")
    return float(nums[0])


def _default_lanes_for_highway(hw: str) -> float:
    if "motorway" in hw or "trunk" in hw:
        return 2.0
    if "primary" in hw or "secondary" in hw:
        return 1.0
    return 1.0


def _infer_critical_density(*, highway: pd.Series) -> np.ndarray:
    hw = highway.astype(str).str.lower()
    out = []
    for x in hw.to_list():
        if "motorway" in x or "trunk" in x:
            out.append(35.0)
        elif "primary" in x or "secondary" in x:
            out.append(45.0)
        else:
            out.append(55.0)
    return np.array(out, dtype=float)


def _atomic_pickle_dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _build_meta(*, place: str, epsg: int, n_nodes: int, n_edges: int) -> Dict[str, Any]:
    return {
        "place": place,
        "epsg": int(epsg),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "osmnx_version": _try_pkg_version("osmnx"),
        "networkx_version": _try_pkg_version("networkx"),
        "numpy_version": _try_pkg_version("numpy"),
        "pandas_version": _try_pkg_version("pandas"),
    }


def _try_pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


if __name__ == "__main__":
    main()
