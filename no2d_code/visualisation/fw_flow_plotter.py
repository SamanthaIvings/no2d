from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from no2d_code.solver.digraph import Digraph


@dataclass(frozen=True)
class NodeXY:
    x: NDArray[np.float64]
    y: NDArray[np.float64]


NodesLike = Union[pd.DataFrame, str, Path, NodeXY]
LsoaLike = Union[pd.DataFrame, str, Path]
EdgeLsoaMapLike = Union[pd.DataFrame, str, Path]


def _guess_nodes_crs(node_xy: NodeXY) -> str:
    x_max = float(np.nanmax(np.abs(node_xy.x)))
    y_max = float(np.nanmax(np.abs(node_xy.y)))
    if x_max <= 180.0 and y_max <= 90.0:
        return "EPSG:4326"
    return "EPSG:27700"


def _node_xy_from_nodes(nodes: NodesLike, n_nodes: int) -> NodeXY:
    if isinstance(nodes, NodeXY):
        if nodes.x.size < n_nodes or nodes.y.size < n_nodes:
            raise ValueError(f"NodeXY arrays too small: need n_nodes={n_nodes}")
        return NodeXY(x=nodes.x[:n_nodes].astype(float), y=nodes.y[:n_nodes].astype(float))

    if isinstance(nodes, (str, Path)):
        nodes_df = pd.read_csv(str(nodes))
    elif isinstance(nodes, pd.DataFrame):
        nodes_df = nodes
    else:
        raise TypeError(f"Unsupported nodes type: {type(nodes)}")

    if "NodeID_New" in nodes_df.columns:
        id_col = "NodeID_New"
    elif "NodeID" in nodes_df.columns:
        id_col = "NodeID"
    elif "node" in nodes_df.columns:
        id_col = "node"
    else:
        id_col = None

    if "x" not in nodes_df.columns or "y" not in nodes_df.columns:
        raise ValueError("nodes must contain columns 'x' and 'y'")

    x = np.full(n_nodes, np.nan, dtype=float)
    y = np.full(n_nodes, np.nan, dtype=float)

    if id_col is None:
        if len(nodes_df) < n_nodes:
            raise ValueError(f"nodes has {len(nodes_df)} rows but graph has n_nodes={n_nodes}")
        x[:] = nodes_df["x"].to_numpy(dtype=float)[:n_nodes]
        y[:] = nodes_df["y"].to_numpy(dtype=float)[:n_nodes]
        return NodeXY(x=x, y=y)

    ids = nodes_df[id_col].to_numpy(dtype=int)
    xs = nodes_df["x"].to_numpy(dtype=float)
    ys = nodes_df["y"].to_numpy(dtype=float)

    mask = (ids >= 0) & (ids < n_nodes)
    x[ids[mask]] = xs[mask]
    y[ids[mask]] = ys[mask]

    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        bad = int(np.sum(~np.isfinite(x) | ~np.isfinite(y)))
        raise ValueError(
            f"Missing coordinates for {bad} nodes. "
            f"Check that {id_col} matches Digraph node ids 0..{n_nodes - 1}."
        )

    return NodeXY(x=x, y=y)


def _edge_segments(graph: Digraph, node_xy: NodeXY) -> NDArray[np.float64]:
    u = graph.u.astype(int)
    v = graph.v.astype(int)

    x0 = node_xy.x[u]
    y0 = node_xy.y[u]
    x1 = node_xy.x[v]
    y1 = node_xy.y[v]

    seg = np.empty((u.size, 2, 2), dtype=float)
    seg[:, 0, 0] = x0
    seg[:, 0, 1] = y0
    seg[:, 1, 0] = x1
    seg[:, 1, 1] = y1
    return seg


def _edge_midpoints(graph: Digraph, node_xy: NodeXY) -> NDArray[np.float64]:
    u = graph.u.astype(int)
    v = graph.v.astype(int)
    x = (node_xy.x[u] + node_xy.x[v]) * 0.5
    y = (node_xy.y[u] + node_xy.y[v]) * 0.5
    return np.column_stack([x, y])


def _require_geopandas():
    try:
        import geopandas as gpd
        from shapely import wkt
    except ImportError as exc:
        raise ImportError(
            "LSOA plotting requires geopandas and shapely."
        ) from exc
    return gpd, wkt


def _resolve_column(
    columns: pd.Index,
    preferred: Optional[str],
    candidates: tuple[str, ...],
    label: str,
) -> str:
    if preferred:
        if preferred in columns:
            return preferred
        raise ValueError(f"{label} column '{preferred}' not found.")
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"Could not find {label} column. Tried {candidates}.")


def _read_lsoa_polygons(
    lsoa_polygons: LsoaLike,
    *,
    code_col: Optional[str] = None,
    geometry_col: Optional[str] = None,
    name_col: Optional[str] = None,
    name_filter: Optional[list[str]] = None,
) -> "gpd.GeoDataFrame":
    gpd, wkt = _require_geopandas()

    def apply_name_filter(df: pd.DataFrame) -> pd.DataFrame:
        if not name_filter:
            return df
        columns = df.columns
        resolved = _resolve_column(
            columns,
            name_col,
            ("LSOA11NM", "LSOA21NM", "LSOA", "lsoa_name", "name"),
            "LSOA name",
        )
        pattern = "|".join(name_filter)
        return df.loc[df[resolved].astype(str).str.contains(pattern, case=False, regex=True)]

    if isinstance(lsoa_polygons, pd.DataFrame):
        df = apply_name_filter(lsoa_polygons.copy())
        columns = df.columns
        code_col = _resolve_column(
            columns,
            code_col,
            ("LSOA11CD", "LSOA21CD", "LSOA", "lsoa_code", "code"),
            "LSOA code",
        )
        geometry_col = _resolve_column(
            columns,
            geometry_col,
            ("geometry", "geom", "wkt", "WKT"),
            "geometry",
        )
        df = df.dropna(subset=[code_col, geometry_col]).drop_duplicates(code_col)
        if df.empty:
            raise ValueError("LSOA polygons dataframe is empty after filtering.")
        if not hasattr(df[geometry_col].iloc[0], "geom_type"):
            df[geometry_col] = df[geometry_col].map(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry=geometry_col, crs="EPSG:4326")
    else:
        path = Path(lsoa_polygons)
        if not path.exists():
            raise FileNotFoundError(f"LSOA polygons not found: {path}")

        if path.suffix.lower() in {".csv", ".txt"}:
            header = pd.read_csv(path, nrows=0)
            code_col = _resolve_column(
                header.columns,
                code_col,
                ("LSOA11CD", "LSOA21CD", "LSOA", "lsoa_code", "code"),
                "LSOA code",
            )
            geometry_col = _resolve_column(
                header.columns,
                geometry_col,
                ("geometry", "geom", "wkt", "WKT"),
                "geometry",
            )
            df = pd.read_csv(path, usecols=[code_col, geometry_col])
            df = apply_name_filter(df)
            df = df.dropna(subset=[code_col, geometry_col]).drop_duplicates(code_col)
            df[geometry_col] = df[geometry_col].map(wkt.loads)
            gdf = gpd.GeoDataFrame(df, geometry=geometry_col, crs="EPSG:4326")
        else:
            gdf = gpd.read_file(path)
            gdf = apply_name_filter(gdf)
            columns = gdf.columns
            code_col = _resolve_column(
                columns,
                code_col,
                ("LSOA11CD", "LSOA21CD", "LSOA", "lsoa_code", "code"),
                "LSOA code",
            )
            if geometry_col and geometry_col != gdf.geometry.name:
                if geometry_col not in gdf.columns:
                    raise ValueError(f"Geometry column '{geometry_col}' not found in {path}")
                gdf = gdf.set_geometry(geometry_col)

    gdf = gdf.loc[:, [code_col, gdf.geometry.name]].drop_duplicates(code_col)
    gdf = gdf.rename(columns={code_col: "lsoa_code"})
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def _read_edge_lsoa_map(edge_lsoa_map: EdgeLsoaMapLike) -> pd.DataFrame:
    if isinstance(edge_lsoa_map, pd.DataFrame):
        df = edge_lsoa_map.copy()
    else:
        df = pd.read_csv(Path(edge_lsoa_map))

    if "edge_id" not in df.columns:
        for alt in ("edge", "edge_idx", "edge_index", "idx"):
            if alt in df.columns:
                df = df.rename(columns={alt: "edge_id"})
                break
    if "lsoa_code" not in df.columns:
        for alt in ("LSOA11CD", "LSOA21CD", "LSOA"):
            if alt in df.columns:
                df = df.rename(columns={alt: "lsoa_code"})
                break

    if "edge_id" not in df.columns or "lsoa_code" not in df.columns:
        raise ValueError("edge_lsoa_map must contain 'edge_id' and 'lsoa_code' columns.")

    return df.loc[:, ["edge_id", "lsoa_code"]]


def build_edge_lsoa_map(
    graph: Digraph,
    *,
    nodes: NodesLike,
    lsoa_polygons: LsoaLike,
    out_path: Optional[Union[str, Path]] = None,
    nodes_crs: Optional[str] = None,
    lsoa_code_col: Optional[str] = None,
    lsoa_geometry_col: Optional[str] = None,
    lsoa_name_col: Optional[str] = None,
    lsoa_name_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    gpd, _ = _require_geopandas()

    node_xy = _node_xy_from_nodes(nodes, graph.n_nodes)
    nodes_crs = nodes_crs or _guess_nodes_crs(node_xy)
    midpoints = _edge_midpoints(graph, node_xy)

    points = gpd.GeoDataFrame(
        {
            "edge_id": np.arange(graph.u.size, dtype=int),
            "x": midpoints[:, 0],
            "y": midpoints[:, 1],
        },
        geometry=gpd.points_from_xy(midpoints[:, 0], midpoints[:, 1]),
        crs=nodes_crs,
    )

    lsoa_gdf = _read_lsoa_polygons(
        lsoa_polygons,
        code_col=lsoa_code_col,
        geometry_col=lsoa_geometry_col,
        name_col=lsoa_name_col,
        name_filter=lsoa_name_filter,
    )
    if lsoa_gdf.crs is None:
        lsoa_gdf = lsoa_gdf.set_crs(points.crs)
    elif lsoa_gdf.crs != points.crs:
        lsoa_gdf = lsoa_gdf.to_crs(points.crs)

    try:
        joined = gpd.sjoin(points, lsoa_gdf[["lsoa_code", "geometry"]], how="left", predicate="within")
    except TypeError:
        joined = gpd.sjoin(points, lsoa_gdf[["lsoa_code", "geometry"]], how="left", op="within")

    out = joined.loc[:, ["edge_id", "x", "y", "lsoa_code"]].copy()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
    return out


def aggregate_lsoa_flow_change(
    flow_init: NDArray[np.float64],
    flow_final: NDArray[np.float64],
    edge_lsoa_map: EdgeLsoaMapLike,
    *,
    stat: str = "median",
) -> pd.DataFrame:
    flow_init = np.asarray(flow_init, dtype=float).reshape(-1)
    flow_final = np.asarray(flow_final, dtype=float).reshape(-1)
    if flow_init.size != flow_final.size:
        raise ValueError("flow_init/flow_final must have same length")

    df = _read_edge_lsoa_map(edge_lsoa_map)
    df = df.dropna(subset=["lsoa_code"])
    edge_ids = df["edge_id"].to_numpy(dtype=int)
    if edge_ids.size == 0:
        return pd.DataFrame(columns=["lsoa_code", "flow_init", "flow_final", "flow_delta", "edge_count"])
    if edge_ids.max() >= flow_init.size:
        raise ValueError("edge_lsoa_map references edge ids outside flow arrays")

    values = pd.DataFrame(
        {
            "lsoa_code": df["lsoa_code"].astype(str).to_numpy(),
            "flow_init": flow_init[edge_ids],
            "flow_final": flow_final[edge_ids],
            "flow_delta": (flow_final - flow_init)[edge_ids],
        }
    )

    grouped = values.groupby("lsoa_code", sort=False)
    if stat == "median":
        agg = grouped.median(numeric_only=True)
    elif stat == "mean":
        agg = grouped.mean(numeric_only=True)
    elif stat == "sum":
        agg = grouped.sum(numeric_only=True)
    else:
        raise ValueError("stat must be one of: median, mean, sum")

    agg["edge_count"] = grouped.size()
    agg = agg.reset_index()
    return agg


def aggregate_lsoa_value_change(
    value_init: NDArray[np.float64],
    value_final: NDArray[np.float64],
    edge_lsoa_map: EdgeLsoaMapLike,
    *,
    stat: str = "median",
) -> pd.DataFrame:
    value_init = np.asarray(value_init, dtype=float).reshape(-1)
    value_final = np.asarray(value_final, dtype=float).reshape(-1)
    if value_init.size != value_final.size:
        raise ValueError("value_init/value_final must have same length")

    df = _read_edge_lsoa_map(edge_lsoa_map)
    df = df.dropna(subset=["lsoa_code"])
    edge_ids = df["edge_id"].to_numpy(dtype=int)
    if edge_ids.size == 0:
        return pd.DataFrame(
            columns=["lsoa_code", "value_init", "value_final", "value_delta", "edge_count"]
        )
    if edge_ids.max() >= value_init.size:
        raise ValueError("edge_lsoa_map references edge ids outside value arrays")

    values = pd.DataFrame(
        {
            "lsoa_code": df["lsoa_code"].astype(str).to_numpy(),
            "value_init": value_init[edge_ids],
            "value_final": value_final[edge_ids],
            "value_delta": (value_final - value_init)[edge_ids],
        }
    )

    grouped = values.groupby("lsoa_code", sort=False)
    if stat == "median":
        agg = grouped.median(numeric_only=True)
    elif stat == "mean":
        agg = grouped.mean(numeric_only=True)
    elif stat == "sum":
        agg = grouped.sum(numeric_only=True)
    else:
        raise ValueError("stat must be one of: median, mean, sum")

    agg["edge_count"] = grouped.size()
    agg = agg.reset_index()
    return agg


def plot_lsoa_value_comparison(
    graph: Digraph,
    value_init: NDArray[np.float64],
    value_final: NDArray[np.float64],
    *,
    nodes: NodesLike,
    lsoa_polygons: LsoaLike,
    edge_lsoa_map: Optional[EdgeLsoaMapLike] = None,
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    change_stat: str = "median",
    vmax: Optional[float] = None,
    vmax_quantile: float = 0.98,
    overlay_edges: bool = True,
    nodes_crs: Optional[str] = None,
    lsoa_name_col: Optional[str] = None,
    lsoa_name_filter: Optional[list[str]] = None,
    cbar_label: str = "Change in value",
) -> None:
    value_init = np.asarray(value_init, dtype=float).reshape(-1)
    value_final = np.asarray(value_final, dtype=float).reshape(-1)
    if value_init.size != graph.u.size or value_final.size != graph.u.size:
        raise ValueError("value_init/value_final must have length = number of edges")

    node_xy = _node_xy_from_nodes(nodes, graph.n_nodes)
    nodes_crs = nodes_crs or _guess_nodes_crs(node_xy)

    lsoa_gdf = _read_lsoa_polygons(
        lsoa_polygons,
        name_col=lsoa_name_col,
        name_filter=lsoa_name_filter,
    )
    if lsoa_gdf.crs is None:
        lsoa_gdf = lsoa_gdf.set_crs(nodes_crs)
    elif lsoa_gdf.crs != nodes_crs:
        lsoa_gdf = lsoa_gdf.to_crs(nodes_crs)

    if edge_lsoa_map is None:
        edge_lsoa_map = build_edge_lsoa_map(
            graph,
            nodes=nodes,
            lsoa_polygons=lsoa_polygons,
            nodes_crs=nodes_crs,
            lsoa_name_col=lsoa_name_col,
            lsoa_name_filter=lsoa_name_filter,
        )

    stats = aggregate_lsoa_value_change(
        value_init,
        value_final,
        edge_lsoa_map,
        stat=change_stat,
    )

    gdf = lsoa_gdf.merge(stats, on="lsoa_code", how="left")
    values = gdf["value_delta"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if vmax is None:
        if finite.size:
            vmax = float(np.quantile(np.abs(finite), vmax_quantile))
            if vmax <= 0.0:
                vmax = float(np.max(np.abs(finite)))
        else:
            vmax = 1.0
    vmax = max(float(vmax), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
    gdf.plot(
        ax=ax,
        column="value_delta",
        cmap="BrBG",
        norm=norm,
        linewidth=0.3,
        edgecolor="#9a9a9a",
        missing_kwds={"color": "#f2f2f2", "edgecolor": "#c0c0c0"},
    )

    if overlay_edges:
        segments = _edge_segments(graph, node_xy)
        lc = LineCollection(
            segments,
            colors=[(0, 0, 0, 0.18)],
            linewidths=0.35,
            capstyle="round",
        )
        ax.add_collection(lc)

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(norm=norm, cmap="BrBG")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"{cbar_label} ({change_stat})")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    plt.close(fig)


def plot_fw_lsoa_flow_comparison(
    graph: Digraph,
    flow_init: NDArray[np.float64],
    flow_final: NDArray[np.float64],
    *,
    nodes: NodesLike,
    lsoa_polygons: LsoaLike,
    edge_lsoa_map: Optional[EdgeLsoaMapLike] = None,
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    change_stat: str = "median",
    vmax: Optional[float] = None,
    vmax_quantile: float = 0.98,
    overlay_edges: bool = True,
    nodes_crs: Optional[str] = None,
    lsoa_name_col: Optional[str] = None,
    lsoa_name_filter: Optional[list[str]] = None,
) -> None:
    flow_init = np.asarray(flow_init, dtype=float).reshape(-1)
    flow_final = np.asarray(flow_final, dtype=float).reshape(-1)
    if flow_init.size != graph.u.size or flow_final.size != graph.u.size:
        raise ValueError("flow_init/flow_final must have length = number of edges")

    node_xy = _node_xy_from_nodes(nodes, graph.n_nodes)
    nodes_crs = nodes_crs or _guess_nodes_crs(node_xy)

    lsoa_gdf = _read_lsoa_polygons(
        lsoa_polygons,
        name_col=lsoa_name_col,
        name_filter=lsoa_name_filter,
    )
    if lsoa_gdf.crs is None:
        lsoa_gdf = lsoa_gdf.set_crs(nodes_crs)
    elif lsoa_gdf.crs != nodes_crs:
        lsoa_gdf = lsoa_gdf.to_crs(nodes_crs)

    if edge_lsoa_map is None:
        edge_lsoa_map = build_edge_lsoa_map(
            graph,
            nodes=nodes,
            lsoa_polygons=lsoa_polygons,
            nodes_crs=nodes_crs,
            lsoa_name_col=lsoa_name_col,
            lsoa_name_filter=lsoa_name_filter,
        )

    stats = aggregate_lsoa_flow_change(
        flow_init,
        flow_final,
        edge_lsoa_map,
        stat=change_stat,
    )

    gdf = lsoa_gdf.merge(stats, on="lsoa_code", how="left")
    values = gdf["flow_delta"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if vmax is None:
        if finite.size:
            vmax = float(np.quantile(np.abs(finite), vmax_quantile))
            if vmax <= 0.0:
                vmax = float(np.max(np.abs(finite)))
        else:
            vmax = 1.0
    vmax = max(float(vmax), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
    gdf.plot(
        ax=ax,
        column="flow_delta",
        cmap="BrBG",
        norm=norm,
        linewidth=0.3,
        edgecolor="#9a9a9a",
        missing_kwds={"color": "#f2f2f2", "edgecolor": "#c0c0c0"},
    )

    if overlay_edges:
        segments = _edge_segments(graph, node_xy)
        # Keep edges subtle so polygon colors remain readable.
        lc = LineCollection(
            segments,
            colors=[(0, 0, 0, 0.18)],
            linewidths=0.35,
            capstyle="round",
        )
        ax.add_collection(lc)

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(norm=norm, cmap="BrBG")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"Change in {change_stat} flow (veh/hour)")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    plt.close(fig)


def _linewidth_from_flow(flow: NDArray[np.float64], vmax: float) -> NDArray[np.float64]:
    if vmax <= 0.0:
        return np.full(flow.size, 0.6, dtype=float)
    s = np.sqrt(np.clip(flow / vmax, 0.0, 1.0))
    return 0.4 + 2.2 * s


def _plot_edge_flow(
    ax,
    segments: NDArray[np.float64],
    values: NDArray[np.float64],
    norm,
    cmap: str,
    lw: NDArray[np.float64],
    alpha: float,
) -> LineCollection:
    lc = LineCollection(
        segments,
        array=values.astype(float),
        cmap=cmap,
        norm=norm,
        linewidths=lw,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(lc)
    return lc


def _set_ax_style(ax, title: str, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _xy_limits(node_xy: NodeXY, pad_frac: float = 0.02) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin = float(np.min(node_xy.x))
    xmax = float(np.max(node_xy.x))
    ymin = float(np.min(node_xy.y))
    ymax = float(np.max(node_xy.y))

    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)

    xlim = (xmin - pad_frac * dx, xmax + pad_frac * dx)
    ylim = (ymin - pad_frac * dy, ymax + pad_frac * dy)
    return xlim, ylim


def plot_fw_flow_comparison(
    graph: Digraph,
    flow_init: NDArray[np.float64],
    flow_final: NDArray[np.float64],
    *,
    nodes: NodesLike,
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    change_threshold: Optional[float] = None,
    delta_only: bool = False,
    cbar_numbers: bool = True,
) -> None:
    flow_init = np.asarray(flow_init, dtype=float).reshape(-1)
    flow_final = np.asarray(flow_final, dtype=float).reshape(-1)

    if flow_init.size != graph.u.size or flow_final.size != graph.u.size:
        raise ValueError("flow_init/flow_final must have length = number of edges")

    node_xy = _node_xy_from_nodes(nodes, graph.n_nodes)
    segments = _edge_segments(graph, node_xy)
    xlim, ylim = _xy_limits(node_xy)

    delta = flow_final - flow_init
    max_abs_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs_delta <= 0.0:
        max_abs_delta = 1.0
    norm_delta = TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta)

    if change_threshold is None:
        change_threshold = 0.02 * max_abs_delta

    if delta_only:
        fig = plt.figure(figsize=(10, 7), dpi=dpi)
        gs = GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1.0, 0.06],
            wspace=0.05,
        )
        ax2 = fig.add_subplot(gs[0, 0])
        cax_delta = fig.add_subplot(gs[0, 1])
    else:
        vmax_flow = float(np.max([np.max(flow_init), np.max(flow_final), 1.0]))
        norm_flow = Normalize(vmin=0.0, vmax=vmax_flow)

        lw_init = _linewidth_from_flow(flow_init, vmax_flow)
        lw_final = _linewidth_from_flow(flow_final, vmax_flow)

        fig = plt.figure(figsize=(16, 8), dpi=dpi)

        gs = GridSpec(
            2,
            4,
            figure=fig,
            width_ratios=[1.0, 1.0, 0.30, 0.06],
            height_ratios=[1.0, 1.0],
            wspace=0.10,
            hspace=0.18,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        cax_flow = fig.add_subplot(gs[0, 3])

        ax2 = fig.add_subplot(gs[1, 0:2])
        cax_delta = fig.add_subplot(gs[1, 3])

        lc0 = _plot_edge_flow(ax0, segments, flow_init, norm=norm_flow, cmap="viridis", lw=lw_init, alpha=1.0)
        _set_ax_style(ax0, "Initial (AoN) edge flows", xlim=xlim, ylim=ylim)

        lc1 = _plot_edge_flow(ax1, segments, flow_final, norm=norm_flow, cmap="viridis", lw=lw_final, alpha=1.0)
        _set_ax_style(ax1, "Final (UE) edge flows", xlim=xlim, ylim=ylim)

        cb_flow = fig.colorbar(lc1, cax=cax_flow)
        cb_flow.set_label("Flow (veh/hour)")

    bg = LineCollection(segments, colors=[(0, 0, 0, 0.10)], linewidths=0.8, capstyle="round")
    ax2.add_collection(bg)

    show_mask = np.abs(delta) >= float(change_threshold)
    seg2 = segments[show_mask]
    delta2 = delta[show_mask]
    lw2 = 0.8 + 2.0 * np.sqrt(np.clip(np.abs(delta2) / max_abs_delta, 0.0, 1.0))

    if seg2.size:
        lc2 = _plot_edge_flow(ax2, seg2, delta2, norm=norm_delta, cmap="BrBG", lw=lw2, alpha=1.0)
    else:
        lc2 = _plot_edge_flow(
            ax2,
            segments[:1],
            np.zeros(1, dtype=float),
            norm=norm_delta,
            cmap="BrBG",
            lw=np.array([0.0], dtype=float),
            alpha=0.0,
        )

    _set_ax_style(ax2, "Final - Initial (flow change)", xlim=xlim, ylim=ylim)

    cb_delta = fig.colorbar(lc2, cax=cax_delta)
    cb_delta.set_label("Flow change (veh/hour)")
    if not cbar_numbers:
        cb_delta.set_ticks([])
        cb_delta.ax.tick_params(length=0)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)

    plt.close(fig)

def plot_edge_value_comparison(
    graph: Digraph,
    value_init: NDArray[np.float64],
    value_final: NDArray[np.float64],
    *,
    nodes: NodesLike,
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    change_threshold: Optional[float] = None,
    title_init: str = "Initial values",
    title_final: str = "Final values",
    cbar_label: str = "Value",
    delta_label: str = "Value change",
    delta_only: bool = False,
    cbar_numbers: bool = True,
) -> None:
    value_init = np.asarray(value_init, dtype=float).reshape(-1)
    value_final = np.asarray(value_final, dtype=float).reshape(-1)

    if value_init.size != graph.u.size or value_final.size != graph.u.size:
        raise ValueError("value_init/value_final must have length = number of edges")

    node_xy = _node_xy_from_nodes(nodes, graph.n_nodes)
    segments = _edge_segments(graph, node_xy)
    xlim, ylim = _xy_limits(node_xy)

    delta = value_final - value_init
    max_abs_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs_delta <= 0.0:
        max_abs_delta = 1.0
    norm_delta = TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta)

    if change_threshold is None:
        change_threshold = 0.02 * max_abs_delta

    if delta_only:
        fig = plt.figure(figsize=(10, 7), dpi=dpi)
        gs = GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1.0, 0.06],
            wspace=0.05,
        )
        ax2 = fig.add_subplot(gs[0, 0])
        cax_delta = fig.add_subplot(gs[0, 1])
    else:
        vmax_val = float(np.max([np.max(value_init), np.max(value_final), 1.0]))
        norm_val = Normalize(vmin=0.0, vmax=vmax_val)

        lw_init = _linewidth_from_flow(value_init, vmax_val)
        lw_final = _linewidth_from_flow(value_final, vmax_val)

        fig = plt.figure(figsize=(16, 8), dpi=dpi)

        gs = GridSpec(
            2,
            4,
            figure=fig,
            width_ratios=[1.0, 1.0, 0.30, 0.06],
            height_ratios=[1.0, 1.0],
            wspace=0.10,
            hspace=0.18,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        cax_val = fig.add_subplot(gs[0, 3])

        ax2 = fig.add_subplot(gs[1, 0:2])
        cax_delta = fig.add_subplot(gs[1, 3])

        lc0 = _plot_edge_flow(ax0, segments, value_init, norm=norm_val, cmap="viridis", lw=lw_init, alpha=1.0)
        _set_ax_style(ax0, title_init, xlim=xlim, ylim=ylim)

        lc1 = _plot_edge_flow(ax1, segments, value_final, norm=norm_val, cmap="viridis", lw=lw_final, alpha=1.0)
        _set_ax_style(ax1, title_final, xlim=xlim, ylim=ylim)

        cb_val = fig.colorbar(lc1, cax=cax_val)
        cb_val.set_label(cbar_label)

    bg = LineCollection(segments, colors=[(0, 0, 0, 0.10)], linewidths=0.8, capstyle="round")
    ax2.add_collection(bg)

    show_mask = np.abs(delta) >= float(change_threshold)
    seg2 = segments[show_mask]
    delta2 = delta[show_mask]
    lw2 = 0.8 + 2.0 * np.sqrt(np.clip(np.abs(delta2) / max_abs_delta, 0.0, 1.0))

    if seg2.size:
        lc2 = _plot_edge_flow(ax2, seg2, delta2, norm=norm_delta, cmap="BrBG", lw=lw2, alpha=1.0)
    else:
        lc2 = _plot_edge_flow(
            ax2,
            segments[:1],
            np.zeros(1, dtype=float),
            norm=norm_delta,
            cmap="BrBG",
            lw=np.array([0.0], dtype=float),
            alpha=0.0,
        )

    _set_ax_style(ax2, f"{title_final} - {title_init}", xlim=xlim, ylim=ylim)

    cb_delta = fig.colorbar(lc2, cax=cax_delta)
    cb_delta.set_label(delta_label)
    if not cbar_numbers:
        cb_delta.set_ticks([])
        cb_delta.ax.tick_params(length=0)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)

    plt.close(fig)


