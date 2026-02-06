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

from no2d_code.frank_wolfe.digraph import Digraph


@dataclass(frozen=True)
class NodeXY:
    x: NDArray[np.float64]
    y: NDArray[np.float64]


NodesLike = Union[pd.DataFrame, str, Path, NodeXY]


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
) -> None:
    flow_init = np.asarray(flow_init, dtype=float).reshape(-1)
    flow_final = np.asarray(flow_final, dtype=float).reshape(-1)

    if flow_init.size != graph.u.size or flow_final.size != graph.u.size:
        raise ValueError("flow_init/flow_final must have length = number of edges")

    node_xy = _node_xy_from_nodes(nodes, graph.n_nodes)
    segments = _edge_segments(graph, node_xy)
    xlim, ylim = _xy_limits(node_xy)

    vmax_flow = float(np.max([np.max(flow_init), np.max(flow_final), 1.0]))
    norm_flow = Normalize(vmin=0.0, vmax=vmax_flow)

    delta = flow_final - flow_init
    max_abs_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs_delta <= 0.0:
        max_abs_delta = 1.0
    norm_delta = TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta)

    if change_threshold is None:
        change_threshold = 0.02 * max_abs_delta

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

    _set_ax_style(ax2, "Final − Initial (flow change)", xlim=xlim, ylim=ylim)

    cb_delta = fig.colorbar(lc2, cax=cax_delta)
    cb_delta.set_label("Flow change (veh/hour)")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

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
) -> None:
    value_init = np.asarray(value_init, dtype=float).reshape(-1)
    value_final = np.asarray(value_final, dtype=float).reshape(-1)

    if value_init.size != graph.u.size or value_final.size != graph.u.size:
        raise ValueError("value_init/value_final must have length = number of edges")

    node_xy = _node_xy_from_nodes(nodes, graph.n_nodes)
    segments = _edge_segments(graph, node_xy)
    xlim, ylim = _xy_limits(node_xy)

    vmax_val = float(np.max([np.max(value_init), np.max(value_final), 1.0]))
    norm_val = Normalize(vmin=0.0, vmax=vmax_val)

    delta = value_final - value_init
    max_abs_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs_delta <= 0.0:
        max_abs_delta = 1.0
    norm_delta = TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta)

    if change_threshold is None:
        change_threshold = 0.02 * max_abs_delta

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

    _set_ax_style(ax2, f"{title_final} − {title_init}", xlim=xlim, ylim=ylim)

    cb_delta = fig.colorbar(lc2, cax=cax_delta)
    cb_delta.set_label(delta_label)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    plt.close(fig)

