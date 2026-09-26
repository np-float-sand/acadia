"""
Monthly cross-sectional residualization of raw_count on filing word count and
log(point-in-time market cap), per docs/backlog-pricing-power-length-control-
preregistration.md §3.

Point-in-time convention mirrors build_panel.py exactly (known_date =
filingDate + 1 calendar day, forward-filled to month-end) but is
reimplemented here (not imported) because it needs to carry three columns
--total_count, n_words, log_market_cap-- from the SAME filing row through the
forward-fill in lockstep, which build_panel.monthly_panel's single-score_col
signature doesn't support. Same LAG_DAYS, same date range, same forward-fill
semantics as build_panel.py -- not a new convention.

Reuses grid_resilience.factor.neutralize (winsorize, cross_section_zscore)
unmodified -- the project's existing neutralization helpers, not new ones
written for this test.

Not production code. Throwaway research script.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
# repo root (two levels up: research/backlog_pricing_power/ -> repo root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from cik_map import SCREEN_27
from grid_resilience.factor.neutralize import cross_section_zscore, winsorize

_CACHE = Path(__file__).parent / "_cache"
SCORES_PATH = _CACHE / "filing_scores.parquet"
CONTROLS_PATH = _CACHE / "filing_controls.parquet"

LAG_DAYS = 1
START, END = "2018-06-01", "2026-08-31"
MIN_CROSS_SECTION = 8


def load_merged() -> pd.DataFrame:
    """One row per (ticker, accessionNumber): total_count, n_words, log_market_cap."""
    scores = pd.read_parquet(SCORES_PATH)[
        ["ticker", "accessionNumber", "filingDate", "total_count", "n_words"]]
    controls = pd.read_parquet(CONTROLS_PATH)[
        ["ticker", "accessionNumber", "log_market_cap"]]
    df = scores.merge(controls, on=["ticker", "accessionNumber"], how="left")
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["known_date"] = df["filingDate"] + pd.Timedelta(days=LAG_DAYS)
    df = df.sort_values(["ticker", "known_date"]).drop_duplicates(
        ["ticker", "accessionNumber"])
    return df


def _point_in_time_panel(df: pd.DataFrame, col: str, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Same forward-fill convention as build_panel.monthly_panel, generalized
    to any column of the merged frame."""
    out = pd.DataFrame(index=month_ends, columns=SCREEN_27, dtype=float)
    for t in SCREEN_27:
        sub = df[df.ticker == t].set_index("known_date")[col].sort_index()
        sub = sub[~sub.index.duplicated(keep="last")]
        if sub.empty:
            continue
        combined = sub.reindex(sub.index.union(month_ends)).sort_index().ffill()
        out[t] = combined.reindex(month_ends)
    return out


def monthly_residual_panel(controls: tuple[str, ...] = ("n_words", "log_market_cap"),
                            start: str = START, end: str = END) -> pd.DataFrame:
    """
    Month x ticker panel of the residual of z(total_count) regressed
    cross-sectionally, per month, on an intercept + z(control) for each
    control in `controls` (after winsorizing all series at 2.5/97.5%
    cross-sectionally that month). Months with fewer than
    MIN_CROSS_SECTION tickers carrying a complete row (score + all controls)
    are left all-NaN for that month (dropped downstream by ic_test's own
    MIN_CROSS_SECTION check, same as the original test).
    """
    df = load_merged()
    month_ends = pd.date_range(start, end, freq="ME")

    score_panel = _point_in_time_panel(df, "total_count", month_ends)
    control_panels = {c: _point_in_time_panel(df, c, month_ends) for c in controls}

    resid_panel = pd.DataFrame(index=month_ends, columns=SCREEN_27, dtype=float)

    for d in month_ends:
        y_raw = score_panel.loc[d].dropna()
        X_raw = {c: control_panels[c].loc[d] for c in controls}
        # complete-case cross-section: score + every control non-null
        idx = y_raw.index
        for c in controls:
            idx = idx.intersection(X_raw[c].dropna().index)
        if len(idx) < MIN_CROSS_SECTION:
            continue

        y = winsorize(y_raw.loc[idx].astype(float))
        y = cross_section_zscore(y)

        X_cols = [np.ones(len(idx))]
        for c in controls:
            xc = winsorize(X_raw[c].loc[idx].astype(float))
            xc = cross_section_zscore(xc)
            X_cols.append(xc.values)
        X = np.column_stack(X_cols)

        beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        fitted = X @ beta
        resid = y.values - fitted
        resid_panel.loc[d, idx] = resid

    return resid_panel


if __name__ == "__main__":
    for name, ctrls in [
        ("two-control (n_words + log_market_cap)", ("n_words", "log_market_cap")),
        ("one-control (n_words only, secondary check)", ("n_words",)),
    ]:
        panel = monthly_residual_panel(controls=ctrls)
        n_months_covered = panel.notna().any(axis=1).sum()
        print(f"{name}: {n_months_covered}/{len(panel)} months with a residual cross-section "
              f"(n>={MIN_CROSS_SECTION})")
        out_name = "raw_count_residual_panel.parquet" if len(ctrls) == 2 \
            else "raw_count_residual_panel_nwords_only.parquet"
        panel.to_parquet(_CACHE / out_name)
