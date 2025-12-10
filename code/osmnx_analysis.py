from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class MapConfig:
    place: str = "South Yorkshire, England, United Kingdom"
    out_dir: Path = Path("../plots/osm_transport_networks")
    dpi: int = 220
    figsize: Tuple[float, float] = (10, 10)
    style: Dict[str, Dict[str, object]] | None = None


_DEFAULT_STYLE = {
    "cars": {"color": "#4E79A7", "lw": 0.55, "alpha": 0.80, "label": "Car streets (drive)"},
    "bike": {"color": "#59A14F", "lw": 0.95, "alpha": 0.90, "label": "Bike infra (cycleways + lanes)"},
    "bike_on_road": {"color": "#2F7D32", "lw": 1.25, "alpha": 0.95, "label": "Bike lanes on roads"},
    "rail": {"color": "#B07AA1", "lw": 1.05, "alpha": 0.90, "label": "Railways"},
    "boundary": {"color": "#111111", "lw": 1.10, "alpha": 1.00, "label": "Boundary"},
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _extend_useful_tags_way(tags: tuple[str, ...]) -> None:
    current = list(getattr(ox.settings, "useful_tags_way", []))
    for t in tags:
        if t not in current:
            current.append(t)
    ox.settings.useful_tags_way = current


def _empty_lines_gdf(crs: str | None = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _features_from_polygon(boundary_geom: BaseGeometry, tags: dict) -> gpd.GeoDataFrame:
    fn = getattr(ox, "features_from_polygon", None)
    if callable(fn):
        try:
            return fn(boundary_geom, tags=tags)
        except Exception as e:
            if e.__class__.__name__ == "InsufficientResponseError":
                return _empty_lines_gdf()
            raise
    fn2 = getattr(ox, "geometries_from_polygon", None)
    if callable(fn2):
        try:
            return fn2(boundary_geom, tags=tags)
        except Exception as e:
            if e.__class__.__name__ == "InsufficientResponseError":
                return _empty_lines_gdf()
            raise
    raise AttributeError("OSMnx has neither features_from_polygon nor geometries_from_polygon")


def _only_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf.reset_index(drop=True)
    return gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()


def _to_crs_if_not_empty(gdf: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf.to_crs(crs)


def _concat_lines(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    frames = []
    if a is not None and not a.empty:
        frames.append(a[["geometry"]].copy())
    if b is not None and not b.empty:
        frames.append(b[["geometry"]].copy())
    if not frames:
        return _empty_lines_gdf(crs)
    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)
    return _only_lines(out)


def _get_boundary(place: str) -> gpd.GeoDataFrame:
    gdf = ox.geocode_to_gdf(place)
    if gdf.empty:
        raise ValueError(f"Could not geocode boundary for: {place}")
    return gdf[["geometry"]].copy()


def _download_drive_graph(boundary_geom: BaseGeometry):
    return ox.graph_from_polygon(boundary_geom, network_type="drive", simplify=True)


def _graph_edges_gdf(G) -> gpd.GeoDataFrame:
    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True, fill_edge_geometry=True)
    return edges.reset_index()


def _download_rail_layers(boundary_geom: BaseGeometry) -> gpd.GeoDataFrame:
    gdf = _features_from_polygon(
        boundary_geom,
        tags={"railway": ["rail", "light_rail", "subway", "tram", "narrow_gauge"]},
    )
    return _only_lines(gdf)


def _download_dedicated_cycleways(boundary_geom: BaseGeometry) -> gpd.GeoDataFrame:
    gdf = _features_from_polygon(boundary_geom, tags={"highway": ["cycleway"]})
    return _only_lines(gdf)


def _is_truthy_cycleway_value(v: object) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    if isinstance(v, str):
        s = v.strip().lower()
        return s not in {"", "no", "none", "false", "0"}
    return True


def _extract_on_road_bike_lanes(car_edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = [c for c in ("cycleway", "cycleway:left", "cycleway:right", "bicycle") if c in car_edges.columns]
    if not cols:
        return car_edges.iloc[0:0].copy()

    mask = pd.Series(False, index=car_edges.index)
    for c in cols:
        mask |= car_edges[c].map(_is_truthy_cycleway_value)

    return _only_lines(car_edges[mask].copy())


def _plot_boundary(ax, boundary: gpd.GeoDataFrame, st: dict) -> None:
    boundary.boundary.plot(
        ax=ax,
        color=st["boundary"]["color"],
        linewidth=float(st["boundary"]["lw"]),
        alpha=float(st["boundary"]["alpha"]),
    )
    ax.set_axis_off()


def _plot_layer(ax, gdf: gpd.GeoDataFrame, st: dict) -> None:
    if gdf is None or gdf.empty:
        return
    gdf.plot(
        ax=ax,
        color=st["color"],
        linewidth=float(st["lw"]),
        alpha=float(st["alpha"]),
    )


def _add_legend(ax, styles: list[dict]) -> None:
    handles = [Line2D([0], [0], color=s["color"], lw=float(s["lw"]), label=str(s["label"])) for s in styles]
    ax.legend(handles=handles, loc="lower left", frameon=True)


def _save_fig(fig, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_maps(cfg: MapConfig) -> None:
    ox.settings.use_cache = True
    ox.settings.log_console = True
    _extend_useful_tags_way(("cycleway", "cycleway:left", "cycleway:right", "bicycle"))

    _ensure_dir(cfg.out_dir)
    st = cfg.style or _DEFAULT_STYLE

    boundary = _get_boundary(cfg.place)
    boundary_geom = boundary.geometry.iloc[0]

    G_drive = _download_drive_graph(boundary_geom)
    G_drive = ox.project_graph(G_drive)

    car_edges = _graph_edges_gdf(G_drive)
    crs = car_edges.crs
    boundary_p = boundary.to_crs(crs)

    bike_dedicated = _to_crs_if_not_empty(_download_dedicated_cycleways(boundary_geom), crs)
    bike_on_road = _to_crs_if_not_empty(_extract_on_road_bike_lanes(car_edges), crs)
    bike_all = _concat_lines(bike_dedicated, bike_on_road, crs=crs)

    rail_layers = _to_crs_if_not_empty(_download_rail_layers(boundary_geom), crs)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    _plot_boundary(ax, boundary_p, st)
    _plot_layer(ax, car_edges, st["cars"])
    ax.set_title("South Yorkshire — Car network (drive)")
    _add_legend(ax, [st["cars"]])
    _save_fig(fig, cfg.out_dir / "01_cars_drive_network.png", cfg.dpi)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    _plot_boundary(ax, boundary_p, st)
    _plot_layer(ax, bike_all, st["bike"])
    ax.set_title("South Yorkshire — Bike infrastructure (cycleways + lanes)")
    _add_legend(ax, [st["bike"]])
    _save_fig(fig, cfg.out_dir / "02_bike_infrastructure.png", cfg.dpi)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    _plot_boundary(ax, boundary_p, st)
    _plot_layer(ax, car_edges, {**st["cars"], "alpha": 0.25, "lw": 0.45})
    _plot_layer(ax, bike_on_road, st["bike_on_road"])
    ax.set_title("South Yorkshire — Intersection (roads with bike-lane tags)")
    _add_legend(ax, [st["cars"], st["bike_on_road"]])
    _save_fig(fig, cfg.out_dir / "03_intersection_on_road_bikelanes.png", cfg.dpi)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    _plot_boundary(ax, boundary_p, st)
    _plot_layer(ax, rail_layers, st["rail"])
    ax.set_title("South Yorkshire — Railways (OSM)")
    _add_legend(ax, [st["rail"]])
    _save_fig(fig, cfg.out_dir / "04_railways.png", cfg.dpi)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    _plot_boundary(ax, boundary_p, st)
    _plot_layer(ax, car_edges, {**st["cars"], "alpha": 0.30})
    _plot_layer(ax, rail_layers, st["rail"])
    _plot_layer(ax, bike_all, st["bike"])
    ax.set_title("South Yorkshire — All transport layers")
    _add_legend(ax, [st["cars"], st["bike"], st["rail"]])
    _save_fig(fig, cfg.out_dir / "05_all_modes.png", cfg.dpi)


def main() -> None:
    build_maps(MapConfig(style=_DEFAULT_STYLE))


if __name__ == "__main__":
    main()
