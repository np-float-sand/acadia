"""
Pre-registered rank-IC test (docs/backlog-pricing-power-preregistration.md §6),
same convention as research/narrative_mispricing/spike_a_ic_test.py: monthly
cross-sectional Spearman correlation of the point-in-time score vs. forward
return, mean/t-stat across the monthly IC series, reported pooled and on two
non-overlapping sub-windows (2019-01->2022-12, 2023-01->2026-06).

Not production code. Throwaway research script.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from build_panel import monthly_panel
from cik_map import SCREEN_27

MIN_CROSS_SECTION = 8
WINDOWS = {
    "pooled": ("2019-01-01", "2026-06-30"),
    "2019-2022 (pre-AI)": ("2019-01-01", "2022-12-31"),
    "2023-2026 (AI regime)": ("2023-01-01", "2026-06-30"),
}
HORIZONS = [1, 3]


def get_monthly_prices(start: str, end: str) -> pd.DataFrame:
    from grid_equipment_basket.data.prices import fetch_prices
    px = fetch_prices(SCREEN_27, start, end)
    monthly = px.resample("ME").last()
    return monthly


def monthly_ic_series(score: pd.Series, fwd_ret_by_month: pd.DataFrame) -> pd.Series:
    """score: single month's cross-section (index=ticker). Not used directly;
    see rank_ic_series below which loops over months."""
    raise NotImplementedError


def rank_ic_series(score_panel: pd.DataFrame, monthly_px: pd.DataFrame, h: int) -> pd.Series:
    fwd = monthly_px.pct_change(h).shift(-h)
    common_idx = score_panel.index.intersection(fwd.index)
    ics = {}
    for d in common_idx:
        sig = score_panel.loc[d].dropna()
        ret = fwd.loc[d].reindex(sig.index).dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < MIN_CROSS_SECTION:
            continue
        rho = sig.loc[common].astype(float).corr(ret.loc[common], method="spearman")
        if pd.notna(rho):
            ics[d] = rho
    return pd.Series(ics).sort_index()


def summarize(ic: pd.Series) -> dict:
    n = len(ic)
    if n < 5:
        return {"n_months": n, "mean_ic": np.nan, "t_stat": np.nan, "pct_neg": np.nan}
    mean_ic = ic.mean()
    se = ic.std(ddof=1) / np.sqrt(n)
    t_stat = mean_ic / se if se > 0 else np.nan
    return {"n_months": n, "mean_ic": mean_ic, "t_stat": t_stat,
            "pct_neg": 100 * (ic < 0).mean()}


def run(score_col: str = "density_per_1k_words") -> pd.DataFrame:
    score_panel = monthly_panel(score_col=score_col, start="2018-06-01", end="2026-08-31")
    monthly_px = get_monthly_prices("2018-01-01", "2026-08-31")

    rows = []
    for h in HORIZONS:
        ic_full = rank_ic_series(score_panel, monthly_px, h)
        for wname, (wstart, wend) in WINDOWS.items():
            ic_w = ic_full[(ic_full.index >= wstart) & (ic_full.index <= wend)]
            stats = summarize(ic_w)
            rows.append({"horizon_months": h, "window": wname, **stats})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for score_col in ["density_per_1k_words", "total_count", "n_distinct_phrases"]:
        print(f"\n=== score_col = {score_col} ===")
        res = run(score_col=score_col)
        pd.set_option("display.width", 120)
        print(res.to_string(index=False))
