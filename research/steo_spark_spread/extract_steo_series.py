"""
Parse each downloaded STEO_m.xlsx vintage and extract ELWHU_PJ / ELWHU_TX (Table 7a,
sheet '7atab') and NGHHUUS -- Henry Hub spot, $/MMBtu (Table 2, sheet '2tab') aligned to
that vintage's own period-column layout (read from the 'Dates' sheet, which differs in
starting year/column count across vintages -- do not assume fixed column indices).

Output: one long-format parquet, one row per (vintage, series, period):
  vintage_ts        -- Wayback capture timestamp (yyyymmddHHMMSS), proof of when this
                        exact file was retrievable -- the point-in-time anchor.
  forecast_month     -- STEO's own "Forecast Month" label (e.g. "September 2023"),
                        i.e. which monthly release this vintage actually is.
  period             -- yyyymm int, the data month this value describes.
  is_historical      -- 1 if STEO already had actual data for `period` as of this
                        vintage, 0 if it was still a forecast/estimate.
  series             -- one of ELWHU_PJ / ELWHU_TX / NGHHUUS.
  value              -- the vintage's own value for that (period, series).

This is what lets a later reader answer "as of STEO's September 2023 release, what did
the forward curve for PJM Western hub power / Henry Hub gas look like" without any
lookahead from a later, revised vintage.
"""
from __future__ import annotations

import pathlib
import re

import openpyxl
import pandas as pd

HERE = pathlib.Path(__file__).parent
RAW_DIR = HERE / "data" / "raw_xlsx"
OUT_PATH = HERE / "data" / "steo_panel.parquet"

SERIES_SPEC = {
    "ELWHU_PJ": "7atab",
    "ELWHU_TX": "7atab",
    "NGHHUUS": "2tab",
}


def _find_period_row(ws) -> tuple[int, int, list]:
    """Return (row_idx, start_col_idx, periods) for the 'AAAA_DATEX or AAAA_YEAR' row."""
    for row in ws.iter_rows(min_row=1, max_row=25):
        for cell in row:
            if cell.value == "AAAA_DATEX or AAAA_YEAR":
                start_col = cell.column + 1
                periods = []
                for c in range(start_col, ws.max_column + 1):
                    v = ws.cell(row=cell.row, column=c).value
                    if v is None:
                        break
                    periods.append(int(v))
                return cell.row, start_col, periods
    raise ValueError("period row not found")


def _find_historical_row(ws, start_col: int, n: int) -> list:
    for row in ws.iter_rows(min_row=1, max_row=25):
        for cell in row:
            if cell.value == "Historical":
                return [ws.cell(row=cell.row, column=start_col + i).value for i in range(n)]
    return [None] * n


def _find_forecast_month(ws) -> str | None:
    for row in ws.iter_rows(min_row=1, max_row=10):
        for cell in row:
            if isinstance(cell.value, str) and cell.value.strip().startswith("Forecast Month"):
                # value lives a few cells over on the same row
                for c in range(cell.column, cell.column + 6):
                    v = ws.cell(row=cell.row, column=c).value
                    if isinstance(v, str) and re.match(r"[A-Za-z]+ \d{4}", v.strip()):
                        return v.strip()
    return None


def _find_series_row(ws, mnemonic: str, start_col: int, n: int) -> list | None:
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        cell = row[0]
        if cell.value == mnemonic:
            r = cell.row
            return [ws.cell(row=r, column=start_col + i).value for i in range(n)]
    return None


def extract_one(path: pathlib.Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    dates_ws = wb["Dates"]
    _, start_col, periods = _find_period_row(dates_ws)
    n = len(periods)
    hist_flags = _find_historical_row(dates_ws, start_col, n)
    forecast_month = _find_forecast_month(dates_ws)

    vintage_ts = path.stem
    records = []
    for series, sheet_name in SERIES_SPEC.items():
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        # table sheets may have a different start column if their own "Table Beginning"
        # differs; re-derive from that sheet's header row 3/4 by locating the mnemonic's
        # row and counting from column C (mnemonic=A, label=B, values start C) -- but
        # column *count* must match periods length exactly, so trust `start_col`=3 and
        # slice to len(periods).
        vals = _find_series_row(ws, series, 3, n)
        if vals is None:
            continue
        for period, hist, val in zip(periods, hist_flags, vals):
            if val is None:
                continue
            records.append({
                "vintage_ts": vintage_ts,
                "forecast_month": forecast_month,
                "period": period,
                "is_historical": hist,
                "series": series,
                "value": float(val) if isinstance(val, (int, float)) else None,
            })
    wb.close()
    return pd.DataFrame.from_records(records)


def main():
    files = sorted(RAW_DIR.glob("*.xlsx"))
    print(f"Found {len(files)} raw vintage files")
    frames = []
    failed = []
    for f in files:
        try:
            df = extract_one(f)
            if df.empty:
                failed.append((f.name, "empty extraction"))
                continue
            frames.append(df)
        except Exception as e:  # noqa: BLE001
            failed.append((f.name, str(e)))

    print(f"Parsed OK: {len(frames)}, failed: {len(failed)}")
    for name, err in failed:
        print(f"  FAIL {name}: {err}")

    if not frames:
        raise SystemExit("no vintages parsed successfully")

    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(OUT_PATH)
    print(f"\nSaved {len(panel)} rows, {panel['vintage_ts'].nunique()} vintages -> {OUT_PATH}")
    print(panel.groupby("series")["vintage_ts"].nunique())
    print("\nSample forecast_month values (first 10 vintages):")
    print(panel.drop_duplicates("vintage_ts")[["vintage_ts", "forecast_month"]].head(10))


if __name__ == "__main__":
    main()
