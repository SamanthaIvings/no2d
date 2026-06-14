from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOL_METERS = 100

IN_INPUTS_DIR = PROJECT_ROOT / "data" / "inputs" / "fw_inputs_sy"

IN_NODES = IN_INPUTS_DIR / "nodes.csv"
IN_EDGES = IN_INPUTS_DIR / "edges.csv"
IN_OD_LIST = IN_INPUTS_DIR / "OD_list.csv"
IN_DEMAND = IN_INPUTS_DIR / "demand.csv"

OUT_INPUTS_DIR = PROJECT_ROOT / "data" / "inputs" / "fw_inputs_sy_simplified"

OUT_NODES = OUT_INPUTS_DIR / "nodes.csv"
OUT_EDGES = OUT_INPUTS_DIR / "edges.csv"
OUT_OD_LIST = OUT_INPUTS_DIR / f"OD_list_tol{TOL_METERS}.csv"
OUT_DEMAND = OUT_INPUTS_DIR / "demand.csv"

USE_BBOX_PRUNE = True
BBOX_BUFFER_M = 10_000.0

ENABLE_TOL_MERGE = True
MERGE_TOL_M = TOL_METERS


def _ensure_out_dirs() -> None:
    OUT_INPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_nodes_edges(nodes_csv: Path, edges_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    req_nodes = {"node", "x", "y"}
    req_edges = {"u", "v", "length", "speedlim", "capacity", "criticalDensity"}

    missing_n = req_nodes - set(nodes.columns)
    missing_e = req_edges - set(edges.columns)

    if missing_n:
        raise ValueError(f"nodes.csv missing columns: {sorted(missing_n)}")
    if missing_e:
        raise ValueError(f"edges.csv missing columns: {sorted(missing_e)}")

    nodes = nodes.copy()
    edges = edges.copy()

    nodes["node"] = pd.to_numeric(nodes["node"], errors="raise").astype(int)
    nodes["x"] = pd.to_numeric(nodes["x"], errors="coerce")
    nodes["y"] = pd.to_numeric(nodes["y"], errors="coerce")
    nodes = nodes.dropna(subset=["x", "y"]).reset_index(drop=True)

    edges["u"] = pd.to_numeric(edges["u"], errors="raise").astype(int)
    edges["v"] = pd.to_numeric(edges["v"], errors="raise").astype(int)

    for c in ["length", "speedlim", "capacity", "criticalDensity"]:
        edges[c] = pd.to_numeric(edges[c], errors="coerce")
    edges = edges.dropna(subset=["length", "speedlim", "capacity", "criticalDensity"]).reset_index(drop=True)

    return nodes, edges


def _detect_node_cols(od: pd.DataFrame) -> Tuple[str, str]:
    candidates = [
        ("origin_node", "dest_node"),
        ("o_node", "d_node"),
        ("from_node", "to_node"),
        ("from", "to"),
        ("o", "d"),
        ("u", "v"),
    ]
    cols = set(od.columns)
    for a, b in candidates:
        if a in cols and b in cols:
            return a, b

    numeric_cols: List[str] = []
    for c in od.columns:
        s = pd.to_numeric(od[c], errors="coerce")
        ok = np.isfinite(s.to_numpy(dtype=float)).mean() > 0.95
        if ok:
            numeric_cols.append(c)

    if len(numeric_cols) >= 2:
        return numeric_cols[-2], numeric_cols[-1]

    raise ValueError(
        "OD_list must contain node columns (e.g., origin_node/dest_node or o_node/d_node). "
        f"Available columns: {list(od.columns)}"
    )


def _load_od_and_demand(od_list_csv: Path, demand_csv: Path) -> Tuple[pd.DataFrame, np.ndarray, str, str]:
    od = pd.read_csv(od_list_csv)
    o_node_col, d_node_col = _detect_node_cols(od)

    dem_df = pd.read_csv(demand_csv)
    if dem_df.shape[1] == 0:
        raise ValueError("demand.csv is empty")

    dem = pd.to_numeric(dem_df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    if dem.shape[0] != od.shape[0]:
        raise ValueError(f"demand rows ({dem.shape[0]}) != OD_list rows ({od.shape[0]})")

    od = od.copy()
    od[o_node_col] = pd.to_numeric(od[o_node_col], errors="raise").astype(int)
    od[d_node_col] = pd.to_numeric(od[d_node_col], errors="raise").astype(int)

    return od, dem, o_node_col, d_node_col


def _bbox_prune_nodes(nodes_df: pd.DataFrame, protected: Set[int], buffer_m: float) -> Set[int]:
    pos = nodes_df.set_index("node")[["x", "y"]]
    keep = [n for n in protected if n in pos.index]
    if not keep:
        return set(nodes_df["node"].tolist())

    xy = pos.loc[keep].to_numpy(dtype=float)
    xmin = float(np.min(xy[:, 0]) - buffer_m)
    xmax = float(np.max(xy[:, 0]) + buffer_m)
    ymin = float(np.min(xy[:, 1]) - buffer_m)
    ymax = float(np.max(xy[:, 1]) + buffer_m)

    x = nodes_df["x"].to_numpy(dtype=float)
    y = nodes_df["y"].to_numpy(dtype=float)
    n = nodes_df["node"].to_numpy(dtype=int)

    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    in_box = set(n[mask].tolist())
    return in_box.union(protected)


def _prune_to_kept_nodes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, keep_nodes: Set[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_out = nodes_df.loc[nodes_df["node"].isin(keep_nodes)].copy().reset_index(drop=True)
    keep = nodes_out["node"].to_numpy(dtype=int)
    keep_set = set(keep.tolist())

    e = edges_df
    mask = e["u"].isin(keep_set) & e["v"].isin(keep_set)
    edges_out = e.loc[mask].copy().reset_index(drop=True)

    return nodes_out, edges_out


def _uf_find(parent: np.ndarray, a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _uf_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def _merge_nearby_nodes(
    nodes_df: pd.DataFrame,
    protected: Set[int],
    tol_m: float,
) -> Tuple[Dict[int, int], pd.DataFrame]:
    node_ids = nodes_df["node"].to_numpy(dtype=int)
    x = nodes_df["x"].to_numpy(dtype=float)
    y = nodes_df["y"].to_numpy(dtype=float)

    n = node_ids.size
    idx_of: Dict[int, int] = {int(node_ids[i]): int(i) for i in range(n)}

    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=np.int8)

    cell = float(tol_m)
    gx = np.floor(x / cell).astype(int)
    gy = np.floor(y / cell).astype(int)

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n):
        key = (int(gx[i]), int(gy[i]))
        buckets.setdefault(key, []).append(i)

    prot_mask = np.array([int(node_ids[i]) in protected for i in range(n)], dtype=bool)

    checks = 0
    merges = 0

    for i in range(n):
        xi = x[i]
        yi = y[i]
        ci = (int(gx[i]), int(gy[i]))

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand = buckets.get((ci[0] + dx, ci[1] + dy))
                if not cand:
                    continue

                for j in cand:
                    if j <= i:
                        continue

                    if prot_mask[i] and prot_mask[j]:
                        continue

                    dxm = xi - x[j]
                    dym = yi - y[j]
                    if dxm * dxm + dym * dym > tol_m * tol_m:
                        continue

                    if prot_mask[i] and not prot_mask[j]:
                        _uf_union(parent, rank, i, j)
                        merges += 1
                    elif prot_mask[j] and not prot_mask[i]:
                        _uf_union(parent, rank, j, i)
                        merges += 1
                    else:
                        _uf_union(parent, rank, i, j)
                        merges += 1

                    checks += 1

        if (i + 1) % 10_000 == 0:
            print(f"[merge] scanned={i+1}/{n} merges_so_far={merges}")

    roots = np.array([_uf_find(parent, i) for i in range(n)], dtype=int)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = int(roots[i])
        groups.setdefault(r, []).append(i)

    rep_idx: Dict[int, int] = {}
    for r, members in groups.items():
        prot_members = [m for m in members if prot_mask[m]]
        if prot_members:
            rep_idx[r] = int(prot_members[0])
        else:
            rep_idx[r] = int(members[0])

    old_to_rep: Dict[int, int] = {}
    rep_rows: List[Tuple[int, float, float]] = []

    for r, members in groups.items():
        rep = rep_idx[r]
        rep_node = int(node_ids[rep])

        mx = float(np.mean(x[members]))
        my = float(np.mean(y[members]))

        if prot_mask[rep]:
            mx = float(x[rep])
            my = float(y[rep])

        rep_rows.append((rep_node, mx, my))

        for m in members:
            old_to_rep[int(node_ids[m])] = rep_node

    merged_nodes = pd.DataFrame(rep_rows, columns=["node", "x", "y"]).drop_duplicates(subset=["node"]).reset_index(drop=True)

    print(f"[merge] nodes_in={n} nodes_out={merged_nodes.shape[0]} merges={merges}")
    return old_to_rep, merged_nodes


def _apply_node_mapping_to_edges(edges_df: pd.DataFrame, old_to_rep: Dict[int, int]) -> pd.DataFrame:
    e = edges_df.copy()

    e["u"] = e["u"].map(old_to_rep)
    e["v"] = e["v"].map(old_to_rep)
    e = e.dropna(subset=["u", "v"]).copy()

    e["u"] = e["u"].astype(int)
    e["v"] = e["v"].astype(int)

    e = e.loc[e["u"] != e["v"]].copy()

    length = e["length"].to_numpy(dtype=float)
    speed = e["speedlim"].to_numpy(dtype=float)
    tt_h = (length / 1000.0) / np.maximum(speed, 1e-12)
    e["_tt_h"] = tt_h

    e = e.sort_values("_tt_h", ascending=True).drop_duplicates(subset=["u", "v"], keep="first").copy()
    e = e.drop(columns=["_tt_h"]).reset_index(drop=True)

    return e


def _to_simple_digraph(edges_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    length = edges_df["length"].to_numpy(dtype=float)
    speed = edges_df["speedlim"].to_numpy(dtype=float)
    tt_h = (length / 1000.0) / np.maximum(speed, 1e-12)

    u = edges_df["u"].to_numpy(dtype=int)
    v = edges_df["v"].to_numpy(dtype=int)

    cap = edges_df["capacity"].to_numpy(dtype=float)
    rho = edges_df["criticalDensity"].to_numpy(dtype=float)

    for i in range(edges_df.shape[0]):
        ui = int(u[i])
        vi = int(v[i])
        if ui == vi:
            continue

        data = {
            "length": float(length[i]),
            "speedlim": float(speed[i]),
            "capacity": float(cap[i]),
            "criticalDensity": float(rho[i]),
            "tt_h": float(tt_h[i]),
        }

        if G.has_edge(ui, vi):
            if data["tt_h"] < float(G[ui][vi]["tt_h"]):
                G[ui][vi].update(data)
        else:
            G.add_edge(ui, vi, **data)

    return G


def _prune_dead_ends(G: nx.DiGraph, protected: Set[int]) -> nx.DiGraph:
    it = 0
    while True:
        it += 1
        remove = []
        for n in G.nodes:
            if n in protected:
                continue
            if G.in_degree(n) + G.out_degree(n) <= 1:
                remove.append(n)

        if not remove:
            break

        G.remove_nodes_from(remove)
        if it % 5 == 0:
            print(f"[dead-ends] iter={it} removed={len(remove)} nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    return G


def _contract_degree2(G: nx.DiGraph, protected: Set[int]) -> nx.DiGraph:
    changed = True
    n_contract = 0

    while changed:
        changed = False
        for n in list(G.nodes):
            if n in protected:
                continue
            if G.in_degree(n) != 1 or G.out_degree(n) != 1:
                continue

            pred = next(iter(G.predecessors(n)))
            succ = next(iter(G.successors(n)))
            if pred == succ or pred == n or succ == n:
                continue

            a = G[pred][n]
            b = G[n][succ]

            tt_h = float(a["tt_h"]) + float(b["tt_h"])
            length = float(a["length"]) + float(b["length"])

            speed = (length / 1000.0) / max(tt_h, 1e-12)
            cap = min(float(a["capacity"]), float(b["capacity"]))
            rho = min(float(a["criticalDensity"]), float(b["criticalDensity"]))

            data = {
                "length": length,
                "speedlim": speed,
                "capacity": cap,
                "criticalDensity": rho,
                "tt_h": tt_h,
            }

            if G.has_edge(pred, succ):
                if tt_h < float(G[pred][succ]["tt_h"]):
                    G[pred][succ].update(data)
            else:
                G.add_edge(pred, succ, **data)

            G.remove_node(n)
            n_contract += 1
            changed = True

            if n_contract % 50_000 == 0:
                print(f"[contract] contracted={n_contract} nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
            break

    print(f"[contract] total_contracted={n_contract}")
    return G


def _reindex_to_dense(
    G: nx.DiGraph,
    nodes_df: pd.DataFrame,
    od_df: pd.DataFrame,
    demand: np.ndarray,
    o_node_col: str,
    d_node_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    old_nodes = sorted(G.nodes())
    mapping = {old: i for i, old in enumerate(old_nodes)}

    pos = nodes_df.set_index("node").loc[old_nodes, ["x", "y"]].reset_index()
    pos["node"] = pos["node"].map(mapping).astype(int)
    nodes_out = pos[["node", "x", "y"]].sort_values("node").reset_index(drop=True)

    rows = []
    for u, v, data in G.edges(data=True):
        rows.append(
            {
                "u": int(mapping[int(u)]),
                "v": int(mapping[int(v)]),
                "length": float(data["length"]),
                "speedlim": float(data["speedlim"]),
                "capacity": float(data["capacity"]),
                "criticalDensity": float(data["criticalDensity"]),
            }
        )
    edges_out = pd.DataFrame(rows)

    od_out = od_df.copy()
    od_out["_o_new"] = od_out[o_node_col].map(mapping)
    od_out["_d_new"] = od_out[d_node_col].map(mapping)

    keep_mask = od_out["_o_new"].notna() & od_out["_d_new"].notna()
    dropped = int((~keep_mask).sum())
    if dropped:
        print(f"[OD] Dropping {dropped} OD rows because nodes disappeared during simplification")

    od_out = od_out.loc[keep_mask].copy()
    demand_out = demand[keep_mask.to_numpy(dtype=bool)].copy()

    od_out[o_node_col] = od_out["_o_new"].astype(int)
    od_out[d_node_col] = od_out["_d_new"].astype(int)
    od_out = od_out.drop(columns=["_o_new", "_d_new"]).reset_index(drop=True)
    demand_out = demand_out.reshape(-1)

    return nodes_out, edges_out, od_out, demand_out


def main() -> None:
    _ensure_out_dirs()

    for p in [IN_NODES, IN_EDGES, IN_OD_LIST, IN_DEMAND]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    print("1) Loading existing nodes/edges")
    nodes_df, edges_df = _load_nodes_edges(IN_NODES, IN_EDGES)
    print(f"   nodes={nodes_df.shape[0]} edges={edges_df.shape[0]}")

    print("2) Loading OD_list and demand")
    od_df, demand, o_node_col, d_node_col = _load_od_and_demand(IN_OD_LIST, IN_DEMAND)
    protected = set(od_df[o_node_col].tolist()) | set(od_df[d_node_col].tolist())
    print(
        f"   od_rows={od_df.shape[0]} demand_sum={float(np.sum(demand)):.0f} protected_nodes={len(protected)} "
        f"node_cols=({o_node_col},{d_node_col})"
    )

    if USE_BBOX_PRUNE:
        print(f"3) BBox prune inputs (+{BBOX_BUFFER_M}m)")
        keep_nodes = _bbox_prune_nodes(nodes_df, protected, BBOX_BUFFER_M)
        before_n, before_e = nodes_df.shape[0], edges_df.shape[0]
        nodes_df, edges_df = _prune_to_kept_nodes(nodes_df, edges_df, keep_nodes)
        print(f"   kept_nodes={nodes_df.shape[0]}/{before_n} kept_edges={edges_df.shape[0]}/{before_e}")

    if ENABLE_TOL_MERGE:
        print(f"4) Tolerance-merge nearby nodes (tol={MERGE_TOL_M}m), protect OD nodes")
        old_to_rep, merged_nodes = _merge_nearby_nodes(nodes_df, protected, MERGE_TOL_M)

        print("5) Applying merge mapping to OD_list")
        od_df = od_df.copy()
        od_df[o_node_col] = od_df[o_node_col].map(old_to_rep).astype(int)
        od_df[d_node_col] = od_df[d_node_col].map(old_to_rep).astype(int)
        protected = set(od_df[o_node_col].tolist()) | set(od_df[d_node_col].tolist())
        print(f"   protected_nodes(after_merge)={len(protected)}")

        print("6) Applying merge mapping to edges and collapsing duplicates")
        edges_df = _apply_node_mapping_to_edges(edges_df, old_to_rep)
        nodes_df = merged_nodes
        print(f"   nodes={nodes_df.shape[0]} edges={edges_df.shape[0]}")

    print("7) Building DiGraph")
    G = _to_simple_digraph(edges_df)
    print(f"   G nodes={G.number_of_nodes()} G edges={G.number_of_edges()}")

    print("8) Pruning dead ends (protect OD nodes)")
    G = _prune_dead_ends(G, protected)
    print(f"   after_deadends nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    print("9) Contracting degree-2 nodes (protect OD nodes)")
    G = _contract_degree2(G, protected)
    print(f"   after_contract nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    print("10) Reindexing to dense and updating OD_list/demand")
    nodes_out, edges_out, od_out, demand_out = _reindex_to_dense(
        G=G,
        nodes_df=nodes_df,
        od_df=od_df,
        demand=demand,
        o_node_col=o_node_col,
        d_node_col=d_node_col,
    )
    print(
        f"   out nodes={nodes_out.shape[0]} edges={edges_out.shape[0]} od_rows={od_out.shape[0]} "
        f"demand_sum={float(np.sum(demand_out)):.0f}"
    )

    print("11) Saving simplified inputs")
    nodes_out.to_csv(OUT_NODES, index=False)
    edges_out.to_csv(OUT_EDGES, index=False)
    od_out.to_csv(OUT_OD_LIST, index=False)
    pd.Series(demand_out, name="demand").to_csv(OUT_DEMAND, index=False)

    print(f"   saved: {OUT_NODES}")
    print(f"   saved: {OUT_EDGES}")
    print(f"   saved: {OUT_OD_LIST}")
    print(f"   saved: {OUT_DEMAND}")
    print(f"Done. Output parent dir: {OUT_INPUTS_DIR}")


if __name__ == "__main__":
    main()
