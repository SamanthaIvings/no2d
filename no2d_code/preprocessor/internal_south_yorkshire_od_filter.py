from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "inputs" / "symca_OD_2019_230523"

KEY_XLSX = INPUT_DIR / "OD_Region_Key.xlsx"
OD_CSV = INPUT_DIR / "OD_WD_AM_PEAK_Local_Authority.csv"

OUT_CSV = INPUT_DIR / "OD_WD_AM_PEAK_Local_Authority_south_yorkshire_only.csv"

SOUTH_YORKSHIRE_LADS = {"Sheffield", "Rotherham", "Doncaster", "Barnsley"}

MANUAL_EXCLUDE_ODREGIONS = {
    "E02002324",
    "E02002327",
    "E02002328",
    "E02002329",  # Kirklees
    "E02002475",
    "E02002478",
    "E02002482",  # Wakefield
    "E02004068",  # Derbyshire Dales
    "E02004105",
    "E02004106",  # North East Derbyshire
    "E02005835",  # Bassetlaw
}


def load_square_od_matrix(od_csv: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(od_csv)
    if df.shape[1] < 2:
        raise ValueError("OD CSV does not look like a wide OD matrix (needs >= 2 columns).")

    origin_col = df.columns[0]
    m = df.set_index(origin_col)

    m.index = m.index.astype(str)
    m.columns = m.columns.astype(str)

    if set(m.index) != set(m.columns):
        extra_in_rows = sorted(set(m.index) - set(m.columns))
        extra_in_cols = sorted(set(m.columns) - set(m.index))
        raise ValueError(
            "OD matrix labels mismatch (not square / inconsistent labels).\n"
            f"Only-in-rows: {extra_in_rows[:20]}{'...' if len(extra_in_rows) > 20 else ''}\n"
            f"Only-in-cols: {extra_in_cols[:20]}{'...' if len(extra_in_cols) > 20 else ''}"
        )

    m = m.reindex(index=sorted(m.index), columns=sorted(m.columns))
    return m, origin_col


def load_key_map(key_xlsx: Path) -> tuple[dict[str, set[str]], set[str]]:
    df = pd.read_excel(key_xlsx, sheet_name=0)

    required = {"ODRegion", "LAD22NM", "externalSector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Key file is missing columns: {sorted(missing)}")

    df["ODRegion"] = df["ODRegion"].astype(str).str.strip()
    df["LAD22NM"] = df["LAD22NM"].astype(str).str.strip()

    external_mask = (
        df["externalSector"]
        .astype("string")
        .fillna("")
        .str.strip()
        .ne("")
    )
    external_od_regions = set(df.loc[external_mask, "ODRegion"].astype(str).str.strip().tolist())

    m: dict[str, set[str]] = {}
    for od_region, sub in df.groupby("ODRegion"):
        lads = {
            x.strip()
            for x in sub["LAD22NM"].tolist()
            if x and x.strip() and x.strip() not in {"Blank", "nan"}
        }
        m[str(od_region)] = lads

    return m, external_od_regions


def choose_keep_labels(
    labels: list[str],
    key_map: dict[str, set[str]],
    external_od_regions: set[str],
) -> tuple[list[str], list[str]]:
    keep: list[str] = []
    drop: list[str] = []

    for lab in labels:
        if lab in MANUAL_EXCLUDE_ODREGIONS:
            drop.append(lab)
            continue

        if lab in external_od_regions:
            drop.append(lab)
            continue

        lads = key_map.get(lab, set())
        if not lads:
            drop.append(lab)
            continue

        if lads.issubset(SOUTH_YORKSHIRE_LADS):
            keep.append(lab)
        else:
            drop.append(lab)

    return sorted(keep), sorted(drop)


def save_od_matrix(m: pd.DataFrame, origin_col: str, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out = m.reset_index().rename(columns={"index": origin_col})
    out.to_csv(out_csv, index=False)


def main() -> None:
    if not KEY_XLSX.exists():
        raise FileNotFoundError(f"Missing key file: {KEY_XLSX}")
    if not OD_CSV.exists():
        raise FileNotFoundError(f"Missing OD file: {OD_CSV}")

    m, origin_col = load_square_od_matrix(OD_CSV)

    manual_present = sorted(set(MANUAL_EXCLUDE_ODREGIONS) & set(m.index))
    if manual_present:
        print("Manually excluded OD labels:")
        for x in manual_present:
            print(f"  - {x}")
        m = m.drop(index=MANUAL_EXCLUDE_ODREGIONS, columns=MANUAL_EXCLUDE_ODREGIONS, errors="ignore")

    key_map, external_od_regions = load_key_map(KEY_XLSX)

    labels = sorted(m.index.tolist())
    missing_in_key = [x for x in labels if x not in key_map]
    if missing_in_key:
        raise ValueError(f"Some OD labels are missing in the key file: {missing_in_key[:20]}")

    keep, drop = choose_keep_labels(labels, key_map, external_od_regions)

    original_total = int(m.to_numpy().sum())
    filtered = m.loc[keep, keep]
    filtered_total = int(filtered.to_numpy().sum())

    removed_total = original_total - filtered_total
    removed_pct = (removed_total / original_total * 100.0) if original_total else 0.0

    save_od_matrix(filtered, origin_col, OUT_CSV)

    print("\nSouth Yorkshire LADs:")
    for x in sorted(SOUTH_YORKSHIRE_LADS):
        print(f"  - {x}")

    print("\nOD matrix size:")
    print(f"  Original: {m.shape[0]} x {m.shape[1]}")
    print(f"  Filtered: {filtered.shape[0]} x {filtered.shape[1]}")

    print("\nTotals:")
    print(f"  Original total demand: {original_total}")
    print(f"  Filtered total demand: {filtered_total}")
    print(f"  Removed demand:        {removed_total} ({removed_pct:.2f}%)")

    print("\nDropped OD labels:")
    for x in drop:
        print(f"  - {x}")

    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
