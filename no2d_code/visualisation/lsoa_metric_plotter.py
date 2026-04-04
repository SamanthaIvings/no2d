from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def plot_lsoa_value_state(
    *,
    value: np.ndarray,
    lsoa_polygons: "pd.GeoDataFrame",
    edge_lsoa_map,
    out_path: str | Path,
    title: str,
    cbar_label: str,
    lsoa_name_filter: str | None = None,
    agg: str = "mean",
    missing_value: float = np.nan,
):
    lsoa_codes, lsoa_values = _aggregate_edge_values_to_lsoa(
        value=value,
        edge_lsoa_map=edge_lsoa_map,
        agg=agg,
    )

    if isinstance(lsoa_polygons, (str, Path)) or (
        isinstance(lsoa_polygons, pd.DataFrame) and not hasattr(lsoa_polygons, "geometry")
    ):
        lsoa_polygons = _read_lsoa_polygons(lsoa_polygons, name_filter=lsoa_name_filter)

    gdf = lsoa_polygons.copy()

    code_col = _infer_lsoa_code_column(gdf)
    if lsoa_name_filter:
        if isinstance(lsoa_name_filter, (list, tuple, set)):
            pats = [str(x) for x in lsoa_name_filter]
            pat = "|".join(pats)
        else:
            pats = [str(lsoa_name_filter)]
            pat = pats[0]

        try:
            name_col = _infer_lsoa_name_column(gdf)
            gdf = gdf[gdf[name_col].astype(str).str.contains(pat, case=False, regex=True, na=False)]
        except ValueError:
            # Only fall back to code filtering if filters look like LSOA codes.
            code_like = all(re.match(r"^[A-Z]\d{8}$", p) for p in pats)
            if code_like:
                gdf = gdf[gdf[code_col].astype(str).str.contains(pat, na=False)]

    series = pd.Series(lsoa_values, index=pd.Index(lsoa_codes, name=code_col), name="metric")
    gdf = gdf.join(series, on=code_col)
    gdf["metric"] = gdf["metric"].astype(float)

    if np.isfinite(missing_value):
        gdf["metric"] = gdf["metric"].fillna(float(missing_value))

    ax = gdf.plot(
        column="metric",
        legend=True,
        linewidth=0.2,
        edgecolor="black",
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

    ax.set_title(title)
    ax.set_axis_off()

    fig = ax.get_figure()
    if fig is not None:
        fig.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=250)


def _aggregate_edge_values_to_lsoa(
    *,
    value: np.ndarray,
    edge_lsoa_map,
    agg: str,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(value, dtype=float)

    rows_edge, rows_lsoa = _iter_edge_lsoa_pairs(edge_lsoa_map)

    edge_ids = np.fromiter(rows_edge, dtype=int)
    lsoa_codes = np.fromiter(rows_lsoa, dtype="U32")

    if edge_ids.size == 0:
        return np.array([], dtype="U32"), np.array([], dtype=float)

    ok = (edge_ids >= 0) & (edge_ids < v.size) & np.isfinite(v[edge_ids])
    edge_ids = edge_ids[ok]
    lsoa_codes = lsoa_codes[ok]
    vals = v[edge_ids]

    df = pd.DataFrame({"lsoa": lsoa_codes, "val": vals})

    if agg == "mean":
        g = df.groupby("lsoa", sort=False)["val"].mean()
    elif agg == "sum":
        g = df.groupby("lsoa", sort=False)["val"].sum()
    elif agg == "median":
        g = df.groupby("lsoa", sort=False)["val"].median()
    else:
        raise ValueError(f"Unsupported agg={agg!r}. Use 'mean', 'sum', or 'median'.")

    return g.index.to_numpy(dtype="U32"), g.to_numpy(dtype=float)


def _iter_edge_lsoa_pairs(edge_lsoa_map) -> Tuple[Iterable[int], Iterable[str]]:
    # Case 0: path to CSV -> load to DataFrame first
    if isinstance(edge_lsoa_map, (str, Path)):
        p = Path(edge_lsoa_map)

        if not p.exists():
            raise ValueError(f"edge_lsoa_map path does not exist: {p}")

        # Try with header first; if it looks headerless, fall back to header=None
        df = pd.read_csv(p)
        if df.shape[1] < 2:
            df = pd.read_csv(p, header=None)

        edge_lsoa_map = df

    # Case 1: dict[int, list[str] | str]
    if isinstance(edge_lsoa_map, dict):
        def edge_iter():
            for e, lsoas in edge_lsoa_map.items():
                if lsoas is None:
                    continue
                if isinstance(lsoas, (str, bytes)):
                    yield int(e), str(lsoas)
                else:
                    for code in lsoas:
                        yield int(e), str(code)

        pairs = list(edge_iter())
        if not pairs:
            return iter(()), iter(())
        edges, lsoas = zip(*pairs)
        return iter(edges), iter(lsoas)

    # Case 2: pandas DataFrame
    if isinstance(edge_lsoa_map, pd.DataFrame):
        df = edge_lsoa_map

        colset = set(map(str, df.columns))

        # Long format (preferred): edge id column + LSOA code column
        edge_candidates = ["edge", "edge_id", "eid", "Edge", "EDGE"]
        lsoa_candidates = ["lsoa", "lsoa_code", "LSOA11CD", "LSOA21CD", "lsoa11cd", "lsoa21cd"]

        edge_col = next((c for c in edge_candidates if c in colset), None)
        lsoa_col = next((c for c in lsoa_candidates if c in colset), None)

        if edge_col is not None and lsoa_col is not None:
            return (df[edge_col].astype(int).to_numpy(), df[lsoa_col].astype(str).to_numpy())

        # Headerless long format: assume first col=edge id, second col=lsoa code
        if df.shape[1] >= 2 and ("0" in colset or 0 in df.columns):
            c0, c1 = df.columns[0], df.columns[1]
            return (df[c0].astype(int).to_numpy(), df[c1].astype(str).to_numpy())

        # Wide incidence: index=edge_id, columns=LSOA codes, values 0/1
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            raise ValueError("edge_lsoa_map DataFrame has no numeric incidence columns and no (edge,lsoa) columns.")

        nz = np.nonzero(num.to_numpy() > 0)
        if nz[0].size == 0:
            return iter(()), iter(())

        edge_ids = num.index.to_numpy(dtype=int)[nz[0]]
        lsoa_codes = num.columns.to_numpy(dtype=str)[nz[1]]
        return (edge_ids, lsoa_codes)

    # Case 3: array-like
    arr = np.asarray(edge_lsoa_map, dtype=object)

    if arr.ndim == 2 and arr.shape[1] >= 2:
        return (arr[:, 0].astype(int), arr[:, 1].astype(str))

    if arr.ndim == 1:
        def edge_iter_1d():
            for e, lsoas in enumerate(arr):
                if lsoas is None:
                    continue
                if isinstance(lsoas, (str, bytes)):
                    yield int(e), str(lsoas)
                else:
                    try:
                        for code in lsoas:
                            yield int(e), str(code)
                    except TypeError:
                        yield int(e), str(lsoas)

        pairs = list(edge_iter_1d())
        if not pairs:
            return iter(()), iter(())
        edges, lsoas = zip(*pairs)
        return iter(edges), iter(lsoas)

    raise ValueError(
        "edge_lsoa_map must be one of:\n"
        "- path to CSV\n"
        "- dict[int, list[str] | str]\n"
        "- array-like shape (n,2) with (edge_id, lsoa_code)\n"
        "- 1D array/list where index=edge_id and value=list of LSOAs\n"
        "- pandas.DataFrame in wide incidence form (index=edge_id, columns=LSOA codes, values 0/1)\n"
        "- pandas.DataFrame in long form (columns like edge/edge_id and lsoa/LSOA11CD)"
    )


def _infer_lsoa_code_column(lsoa_polygons) -> str:
    candidates = ["LSOA11CD", "LSOA21CD", "lsoa11cd", "lsoa21cd", "LSOA", "lsoa", "lsoa_code"]
    cols = set(map(str, lsoa_polygons.columns))
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(
        "Could not infer LSOA code column. Expected one of: "
        + ", ".join(candidates)
        + f". Available columns: {sorted(cols)[:25]}"
    )


def _infer_lsoa_name_column(lsoa_polygons) -> str:
    candidates = ["LSOA11NM", "LSOA21NM", "LSOA", "lsoa_name", "name"]
    cols = set(map(str, lsoa_polygons.columns))
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(
        "Could not infer LSOA name column. Expected one of: "
        + ", ".join(candidates)
        + f". Available columns: {sorted(cols)[:25]}"
    )


def _read_lsoa_polygons(lsoa_polygons, *, name_filter=None):
    # Reuse the same parsing logic as other plotters to handle CSV/shapefiles.
    from no2d_code.visualisation.fw_flow_plotter import _read_lsoa_polygons as _read

    return _read(lsoa_polygons, name_filter=name_filter)
