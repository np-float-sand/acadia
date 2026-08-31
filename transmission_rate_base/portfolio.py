from __future__ import annotations

import numpy as np
import pandas as pd

from transmission_rate_base import config


def rebalance_dates(trading_index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    """First trading day on/after May 1 of each calendar year in [start, end]."""
    idx = trading_index[(trading_index >= start) & (trading_index <= end)]
    out: list[pd.Timestamp] = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        may1 = pd.Timestamp(year=year, month=config.REBALANCE_MONTH, day=1)
        after = idx[idx >= may1]
        if len(after):
            out.append(after[0])
    return out


def _leg_beta(leg_ret: pd.Series, mkt: pd.Series, asof: pd.Timestamp) -> float:
    window = leg_ret.loc[:asof].iloc[-config.BETA_WINDOW:]
    m = mkt.reindex(window.index)
    pair = pd.DataFrame({"r": window, "m": m}).dropna()
    if len(pair) < 60 or pair["m"].var() == 0:
        return 1.0
    return float(np.cov(pair["r"], pair["m"])[0, 1] / pair["m"].var())


def _quintile_members(sig_year: pd.Series) -> tuple[list[str], list[str]]:
    s = sig_year.dropna().sort_values()
    k = max(1, int(round(len(s) / config.QUINTILES)))
    return s.index[-k:].tolist(), s.index[:k].tolist()


def quintile_ls_weights(signal_df: pd.DataFrame, daily_ret: pd.DataFrame,
                        ew_util_ret: pd.Series) -> pd.DataFrame:
    """Daily forward-filled weight matrix. At each May rebalance in year Y: long
    the top signal quintile / short the bottom, equal-weight per leg; long leg
    gross +1, short leg gross -clip(beta_long/beta_short, *BETA_CLIP)."""
    idx = daily_ret.index
    W = pd.DataFrame(0.0, index=idx, columns=daily_ret.columns)
    sig = signal_df.set_index(["year", "ticker"])["neutral_signal"]
    for d in rebalance_dates(idx, str(idx.min().date()), str(idx.max().date())):
        try:
            sig_year = sig.loc[d.year - 1]
        except KeyError:
            continue
        longs, shorts = _quintile_members(sig_year.reindex(daily_ret.columns))
        if not longs or not shorts:
            continue
        bl = _leg_beta(daily_ret[longs].mean(axis=1), ew_util_ret, d)
        bs = _leg_beta(daily_ret[shorts].mean(axis=1), ew_util_ret, d)
        scale = float(np.clip(bl / bs if bs else 1.0, *config.BETA_CLIP))
        row = pd.Series(0.0, index=daily_ret.columns)
        row[longs] = 1.0 / len(longs)
        row[shorts] = -(1.0 / len(shorts)) * scale
        W.loc[d:] = row.values
    return W


def long_tilt_weights(signal_df: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    """Long-only book: equal weight over signalled names, x1.25 top quintile,
    x0.75 bottom quintile, renormalised to sum 1."""
    idx = daily_ret.index
    W = pd.DataFrame(0.0, index=idx, columns=daily_ret.columns)
    sig = signal_df.set_index(["year", "ticker"])["neutral_signal"]
    for d in rebalance_dates(idx, str(idx.min().date()), str(idx.max().date())):
        try:
            sig_year = sig.loc[d.year - 1].reindex(daily_ret.columns).dropna()
        except KeyError:
            continue
        if sig_year.empty:
            continue
        longs, shorts = _quintile_members(sig_year)
        w = pd.Series(1.0, index=sig_year.index)
        w[longs] *= 1.25
        w[shorts] *= 0.75
        w = w / w.sum()
        row = pd.Series(0.0, index=daily_ret.columns)
        row[w.index] = w.values
        W.loc[d:] = row.values
    return W
