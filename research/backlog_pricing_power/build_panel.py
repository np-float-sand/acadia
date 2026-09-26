"""
Point-in-time monthly panel of the pricing-power density score, per
docs/backlog-pricing-power-preregistration.md §4: each filing's score becomes
"known" on filingDate + 1 calendar day; the panel takes, per ticker per
month-end, the most recently known filing's score, forward-filled.

Not production code. Throwaway research script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cik_map import SCREEN_27

SCORES_PATH = Path(__file__).parent / "_cache" / "filing_scores.parquet"
LAG_DAYS = 1


def load_scores() -> pd.DataFrame:
    df = pd.read_parquet(SCORES_PATH)
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["known_date"] = df["filingDate"] + pd.Timedelta(days=LAG_DAYS)
    df = df.sort_values(["ticker", "known_date"]).drop_duplicates(
        ["ticker", "accessionNumber"])
    return df


def monthly_panel(score_col: str = "density_per_1k_words",
                   start: str = "2018-06-01", end: str = "2026-08-31") -> pd.DataFrame:
    """
    Wide DataFrame: index = month-end dates, columns = tickers, values =
    point-in-time-known `score_col` as of that month-end (last filing with
    known_date <= month-end, forward-filled; NaN before a ticker's first
    known filing in the window).
    """
    df = load_scores()
    month_ends = pd.date_range(start, end, freq="ME")
    out = pd.DataFrame(index=month_ends, columns=SCREEN_27, dtype=float)
    for t in SCREEN_27:
        sub = df[df.ticker == t].set_index("known_date")[score_col].sort_index()
        if sub.empty:
            continue
        # last known value <= each month-end
        combined = sub.reindex(sub.index.union(month_ends)).sort_index().ffill()
        out[t] = combined.reindex(month_ends)
    return out


if __name__ == "__main__":
    panel = monthly_panel()
    print(panel.shape)
    print(panel.notna().sum().sort_values())
    print("\nmonths with >=8 tickers covered:",
          (panel.notna().sum(axis=1) >= 8).sum(), "/", len(panel))
    panel.to_parquet(Path(__file__).parent / "_cache" / "density_monthly_panel.parquet")
