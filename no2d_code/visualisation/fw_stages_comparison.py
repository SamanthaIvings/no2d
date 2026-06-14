import numpy as np
import pandas as pd

from no2d_code.solver.digraph import Digraph


def plot_car_stage1_vs_stage2(
    *,
    graph: Digraph,
    flows_stage1: np.ndarray,
    flows_stage2: np.ndarray,
    nodes_csv: str,
    out_png: str,
    title: str = "Car infrastructure: Stage 1 vs Stage 2",
) -> None:
    import matplotlib.pyplot as plt

    nodes = pd.read_csv(nodes_csv)
    if "NodeID" in nodes.columns:
        node_id = nodes["NodeID"].to_numpy(dtype=int)
        x = nodes["x"].to_numpy(dtype=float)
        y = nodes["y"].to_numpy(dtype=float)
        coord = {int(n): (float(xx), float(yy)) for n, xx, yy in zip(node_id, x, y)}
    else:
        x = nodes["x"].to_numpy(dtype=float)
        y = nodes["y"].to_numpy(dtype=float)
        coord = {int(i): (float(x[i]), float(y[i])) for i in range(len(nodes))}

    u = graph.u.astype(int)
    v = graph.v.astype(int)

    f1 = np.asarray(flows_stage1, dtype=float)
    f2 = np.asarray(flows_stage2, dtype=float)
    if f1.shape != f2.shape:
        raise ValueError("flows_stage1 and flows_stage2 must have same shape")

    d = f2 - f1

    def _draw(ax, values: np.ndarray, mode: str) -> None:
        if mode == "abs":
            vmax = float(np.quantile(np.abs(values), 0.98))
            if vmax <= 0:
                vmax = float(np.max(np.abs(values)) + 1e-12)
            norm = plt.Normalize(vmin=0.0, vmax=vmax)
            cmap = plt.cm.viridis
            colors = cmap(norm(values))
            widths = 0.2 + 2.5 * (values / max(vmax, 1e-12))
        else:
            vmax = float(np.quantile(np.abs(values), 0.98))
            if vmax <= 0:
                vmax = float(np.max(np.abs(values)) + 1e-12)
            norm = plt.Normalize(vmin=-vmax, vmax=vmax)
            cmap = plt.cm.seismic
            colors = cmap(norm(values))
            widths = 0.2 + 2.5 * (np.abs(values) / max(vmax, 1e-12))

        for i in range(u.size):
            a = coord.get(int(u[i]))
            b = coord.get(int(v[i]))
            if a is None or b is None:
                continue
            ax.plot([a[0], b[0]], [a[1], b[1]], color=colors[i], linewidth=float(widths[i]), alpha=0.95)

        ax.set_axis_off()
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title)

    axes[0].set_title("Stage 1 (car flows)")
    _draw(axes[0], f1, mode="abs")

    axes[1].set_title("Stage 2 (car flows)")
    _draw(axes[1], f2, mode="abs")

    axes[2].set_title("Delta (stage2 - stage1)")
    _draw(axes[2], d, mode="diff")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)
