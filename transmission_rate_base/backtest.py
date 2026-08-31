from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(weights_daily: pd.DataFrame, monthly_ret: pd.DataFrame,
                 rf_annual: float = 0.04) -> tuple[pd.Series, dict]:
    """Monthly P&L from a daily forward-filled weight matrix + monthly returns.

    Returns (monthly strategy return Series, metrics dict). Months where every
    held name has a NaN return (e.g. the month before the first return) are
    dropped rather than counted as a zero.
    """
    w = weights_daily.resample("ME").last()
    common = w.index.intersection(monthly_ret.index)
    cols = [c for c in w.columns if c in monthly_ret.columns]
    w = w.loc[common, cols].fillna(0.0)
    r = monthly_ret.loc[common, cols]

    strat = (w * r).sum(axis=1, min_count=1).dropna()

    ann = 12
    ann_ret = strat.mean() * ann
    ann_vol = strat.std() * np.sqrt(ann)
    excess = strat - rf_annual / ann
    sharpe = (excess.mean() / excess.std() * np.sqrt(ann)) if excess.std() > 0 else np.nan
    cum = (1 + strat).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min()) if len(strat) else np.nan
    yr = annual_returns(strat)

    metrics = {
        "ann_return": float(ann_ret) if len(strat) else np.nan,
        "ann_vol": float(ann_vol) if len(strat) else np.nan,
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "hit_rate": float((strat > 0).mean()) if len(strat) else np.nan,
        "positive_years_frac": float((yr > 0).mean()) if len(yr) else np.nan,
    }
    return strat, metrics


def annual_returns(monthly_strat_ret: pd.Series) -> pd.Series:
    if monthly_strat_ret.empty:
        return pd.Series(dtype=float)
    return monthly_strat_ret.groupby(monthly_strat_ret.index.year).apply(
        lambda s: (1 + s).prod() - 1)


def signal_rank_ic(neutral_signal_df: pd.DataFrame, daily_ret: pd.DataFrame,
                   ew_util_daily: pd.Series, rebal_dates) -> dict:
    """Spearman IC between the signal known at rebalance year Y-1 and each name's
    next-252-trading-day return relative to the equal-weight utility universe."""
    sig = neutral_signal_df.set_index(["year", "ticker"])["neutral_signal"]
    ics: dict[pd.Timestamp, float] = {}
    for d in rebal_dates:
        try:
            sy = sig.loc[d.year - 1]
        except KeyError:
            continue
        fwd = daily_ret.index[daily_ret.index >= d][:252]
        if len(fwd) < 60:
            continue
        block = daily_ret.loc[fwd]
        tick_fwd = (1 + block).prod() - 1
        ew_fwd = (1 + ew_util_daily.reindex(fwd)).prod() - 1
        rel = tick_fwd - ew_fwd
        pair = pd.DataFrame({"s": sy, "r": rel}).dropna()
        if len(pair) >= 5:
            ics[d] = pair["s"].corr(pair["r"], method="spearman")

    s = pd.Series(ics).sort_index().dropna()
    if s.empty:
        return {"ic": s, "mean_ic": np.nan, "t_stat": np.nan, "n": 0}
    sd = s.std(ddof=1)
    t_stat = float(s.mean() / sd * np.sqrt(len(s))) if sd and sd > 0 else np.nan
    return {"ic": s, "mean_ic": float(s.mean()), "t_stat": t_stat, "n": len(s)}
