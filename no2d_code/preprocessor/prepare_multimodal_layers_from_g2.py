from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd

from no2d_code.core import filepath_configs as fc

SEED = 43


@dataclass(frozen=True)
class Paths:
    data_dir: str
    edges_csv: str
    out_dir: str
    edges_car_pkl: str
    edges_bus_pkl: str
    edges_bike_pkl: str


def main() -> None:
    paths = _paths()
    os.makedirs(paths.out_dir, exist_ok=True)

    edges = pd.read_csv(paths.edges_csv)
    edges = _ensure_columns(edges)

    edges_car = edges.copy()

    is_bus_priority = _make_bus_priority_flags(edges)
    is_bike_infra = _make_bike_infra_flags(edges, is_bus_priority=is_bus_priority)

    edges_bus = edges.copy()
    edges_bus["is_bus_priority"] = is_bus_priority

    edges_bike = edges.copy()
    edges_bike["is_bike_infra"] = is_bike_infra
    edges_bike["is_bus_priority"] = is_bus_priority

    _save_pickle(paths.edges_car_pkl, edges_car)
    _save_pickle(paths.edges_bus_pkl, edges_bus)
    _save_pickle(paths.edges_bike_pkl, edges_bike)

    _print_summary(edges, is_bus_priority, is_bike_infra, paths)

    edges_preview = edges.copy()
    edges_preview["is_bus_priority"] = is_bus_priority
    edges_preview["is_bike_infra"] = is_bike_infra

    preview_png = os.path.join(paths.out_dir, "layers_preview.png")
    _plot_layer_preview(edges_preview, preview_png)
    print("Saved preview:", preview_png)


def _paths() -> Paths:
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    edges_csv = fc.input_path(data_dir, fc.EDGES_CSV)

    out_dir = os.path.join(data_dir, "outputs", "osm_sheffield")
    return Paths(
        data_dir=data_dir,
        edges_csv=edges_csv,
        out_dir=out_dir,
        edges_car_pkl=os.path.join(out_dir, "edges_car.pkl"),
        edges_bus_pkl=os.path.join(out_dir, "edges_bus.pkl"),
        edges_bike_pkl=os.path.join(out_dir, "edges_bike.pkl"),
    )


def _ensure_columns(edges: pd.DataFrame) -> pd.DataFrame:
    for c in ("u", "v"):
        if c not in edges.columns:
            raise ValueError(f"edges.csv missing required column: {c}")

    out = edges.copy()

    if "key" not in out.columns:
        out["key"] = 0

    for c in ("highway", "name", "access", "maxspeed"):
        if c not in out.columns:
            out[c] = ""

    for c in ("lanes", "width"):
        if c not in out.columns:
            out[c] = np.nan

    out["highway"] = out["highway"].astype(str).str.lower()
    out["name"] = out["name"].astype(str)
    out["access"] = out["access"].astype(str).str.lower()

    out["lanes"] = pd.to_numeric(out["lanes"], errors="coerce")
    out["width"] = pd.to_numeric(out["width"], errors="coerce")

    return out


def _centre_boost(edges: pd.DataFrame, strength: float, radius_frac: float) -> np.ndarray:
    if "geometry" not in edges.columns:
        return np.ones(edges.shape[0], dtype=float)

    try:
        import geopandas as gpd
    except ImportError:
        return np.ones(edges.shape[0], dtype=float)

    g = gpd.GeoSeries.from_wkt(edges["geometry"], crs="EPSG:4326").to_crs("EPSG:27700")
    mids = g.interpolate(0.5, normalized=True)

    mx = mids.x.to_numpy(dtype=float)
    my = mids.y.to_numpy(dtype=float)

    cx = float(np.median(mx))
    cy = float(np.median(my))

    d = np.sqrt((mx - cx) ** 2 + (my - cy) ** 2)

    r_base = float(np.quantile(d, 0.95))
    r = max(r_base * float(radius_frac), 1.0)

    return 1.0 + float(strength) * np.exp(-((d / r) ** 2))


def _make_bus_priority_flags(edges: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(SEED)

    base_rate: Dict[str, float] = {
        "motorway": 0.00,
        "trunk": 0.03,
        "primary": 0.12,
        "secondary": 0.08,
        "tertiary": 0.05,
        "unclassified": 0.02,
    }

    hwy = edges["highway"].to_numpy(dtype=str)
    lanes = edges["lanes"].to_numpy(dtype=float)
    width = edges["width"].to_numpy(dtype=float)
    access = edges["access"].to_numpy(dtype=str)

    p = np.zeros(edges.shape[0], dtype=float)
    for k, r in base_rate.items():
        p[hwy == k] = r

    lanes_ok = np.isfinite(lanes) & (lanes >= 3.0)
    width_ok = np.isfinite(width) & (width >= 10.0)

    p = p + 0.06 * lanes_ok + 0.04 * width_ok

    restricted = np.isin(access, ["no", "private"])
    p = np.where(restricted, 0.0, p)

    boost = _centre_boost(edges, strength=1.5, radius_frac=0.6)
    p = p * boost

    p = np.clip(p, 0.0, 0.60)
    return (rng.random(edges.shape[0]) < p).astype(bool)


def _make_bike_infra_flags(edges: pd.DataFrame, is_bus_priority: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED + 1)

    base_rate: Dict[str, float] = {
        "motorway": 0.00,
        "trunk": 0.00,
        "primary": 0.03,
        "secondary": 0.07,
        "tertiary": 0.18,
        "unclassified": 0.14,
    }

    hwy = edges["highway"].to_numpy(dtype=str)
    lanes = edges["lanes"].to_numpy(dtype=float)
    width = edges["width"].to_numpy(dtype=float)

    p = np.zeros(edges.shape[0], dtype=float)
    for k, r in base_rate.items():
        p[hwy == k] = r

    width_good = np.isfinite(width) & (width >= 8.0)
    lanes_bad = np.isfinite(lanes) & (lanes >= 4.0)

    p = p + 0.06 * width_good - 0.12 * lanes_bad

    boost = _centre_boost(edges, strength=1.3, radius_frac=0.7)
    p = p * boost

    p = np.clip(p, 0.0, 0.55)

    return (rng.random(edges.shape[0]) < p).astype(bool)



def _save_pickle(path: str, obj: object) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _print_summary(edges: pd.DataFrame, bus: np.ndarray, bike: np.ndarray, paths: Paths) -> None:
    print("Input edges:", paths.edges_csv)
    print("Saved:", paths.edges_car_pkl)
    print("Saved:", paths.edges_bus_pkl)
    print("Saved:", paths.edges_bike_pkl)
    print("Edges:", int(edges.shape[0]))
    print("Bus priority edges:", int(bus.sum()))
    print("Bike infra edges:", int(bike.sum()))


def _plot_layer_preview(edges: pd.DataFrame, out_png: str) -> None:
    if "geometry" not in edges.columns:
        print("No geometry column found; skipping preview plot.")
        return

    g = gpd.GeoDataFrame(
        edges.copy(),
        geometry=gpd.GeoSeries.from_wkt(edges["geometry"], crs="EPSG:4326"),
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    # Base network
    g.plot(
        ax=ax,
        color="#cfcfcf",
        linewidth=0.20,
        alpha=0.35,
        zorder=1,
        label="Car network (base)",
    )

    # Bus priority overlay
    if "is_bike_infra" in g.columns:
        g.loc[g["is_bike_infra"]].plot(
            ax=ax,
            color="#2ca02c",
            linewidth=0.50,
            alpha=0.85,
            zorder=2,
        )

    # Bus last + thicker
    if "is_bus_priority" in g.columns:
        g.loc[g["is_bus_priority"]].plot(
            ax=ax,
            color="#1f77b4",
            linewidth=1.10,
            alpha=0.95,
            zorder=3,
        )

    ax.set_axis_off()
    ax.set_title("G2 layers by mode")

    # Manual legend (geopandas doesn't always populate handles reliably)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="#cfcfcf", lw=2, alpha=0.7, label="Car network (base)"),
    ]
    if "is_bus_priority" in g.columns and bool(g["is_bus_priority"].any()):
        handles.append(Line2D([0], [0], color="#1f77b4", lw=2, label="Bus priority"))
    if "is_bike_infra" in g.columns and bool(g["is_bike_infra"].any()):
        handles.append(Line2D([0], [0], color="#2ca02c", lw=2, label="Bike infra"))

    ax.legend(handles=handles, loc="lower left", frameon=True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    main()
