from __future__ import annotations

"""Hedge overlays and drawdown-episode scoring for the value-chain reframe
(spec 2026-08-29 §6, §7). Simple-return convention."""

import numpy as np
import pandas as pd

from grid_equipment_basket import config, value_chain
from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket.basket import simulate_basket


def simulate_pair(prices: pd.DataFrame, start: str, end: str,
                  fund_df: pd.DataFrame, backlog_df: pd.DataFrame,
                  lag_days: int = 42) -> pd.Series:
    """Market-neutral maker/contractor pair return series (spec §6): long makers,
    short contractors, dollar-neutral. Gross exposure is ``config.PAIR_GROSS`` = 1.00
    per leg — implicit here because each leg is renormalized to sum to 1 inside
    ``simulate_basket`` and the pair is ``long_leg - short_leg``."""
    known = set(config.BUCKET_MAKERS) | set(config.BUCKET_CONTRACTORS)
    cols = [c for c in prices.columns if c in known]
    makers = [c for c in cols if value_chain.bucket_of(c) == "maker"]
    contractors = [c for c in cols if value_chain.bucket_of(c) == "contractor"]

    def _long_fn(available, asof):
        return value_chain.pair_weights(available, asof, fund_df, backlog_df)[0]

    def _short_fn(available, asof):
        return value_chain.pair_weights(available, asof, fund_df, backlog_df)[1]

    long_leg = simulate_basket(prices[makers], start, end, lag_days, cap=1.0, target_fn=_long_fn)
    short_leg = simulate_basket(prices[contractors], start, end, lag_days, cap=1.0, target_fn=_short_fn)
    pair = long_leg.returns.subtract(short_leg.returns, fill_value=0.0)
    return pair.dropna()


def pair_overlay(base_returns: pd.Series, pair_returns: pd.Series, weight: float) -> pd.Series:
    aligned = pd.concat([base_returns.rename("b"), pair_returns.rename("p")], axis=1)
    return (aligned["b"].fillna(0.0) + weight * aligned["p"].fillna(0.0)).reindex(base_returns.index)


def conditional_short_mask(basket_prices: pd.Series, ma_days: int = 100,
                           vol_days: int = 20, vol_ref_days: int = 252) -> pd.Series:
    """Boolean daily mask for the conditional QQQ short (spec §7.2).

    At each month-end on the basket's own price series the condition
    ``(close < close.rolling(ma_days).mean()) AND (rv > rv.rolling(vol_ref_days).median())``
    is evaluated, where ``rv = close.pct_change().rolling(vol_days).std()``. That
    month-end verdict is held for every trading day of the *following* calendar
    month (the decision is known at the prior month-end). Days before the first
    evaluable month-end are ``False``.
    """
    close = basket_prices.dropna().astype(float).sort_index()
    ma = close.rolling(ma_days).mean()
    rv = close.pct_change().rolling(vol_days).std()
    rv_ref = rv.rolling(vol_ref_days).median()
    daily_cond = ((close < ma) & (rv > rv_ref)).fillna(False)

    periods = close.index.to_period("M")
    month_end_verdict = daily_cond.groupby(periods).last()
    prior_month_verdict = month_end_verdict.shift(1).fillna(False)
    return pd.Series(prior_month_verdict.reindex(periods).to_numpy(), index=close.index).astype(bool)


def conditional_short_overlay(base_returns: pd.Series, hedge_returns: pd.Series,
                              mask: pd.Series, weight: float) -> pd.Series:
    """``base_returns - weight * hedge_returns`` on masked days, ``base_returns`` elsewhere."""
    df = pd.concat([base_returns.rename("b"), hedge_returns.rename("h")], axis=1)
    m = mask.reindex(df.index).fillna(False)
    return (df["b"].fillna(0.0) - weight * df["h"].fillna(0.0).where(m, 0.0)).reindex(base_returns.index)


def find_drawdown_episode(basket_returns: pd.Series, peak_window, trough_end):
    curve = (1.0 + basket_returns.dropna()).cumprod()
    lo, hi = pd.Timestamp(peak_window[0]), pd.Timestamp(peak_window[1])
    peak_seg = curve.loc[lo:hi]
    if peak_seg.empty:
        raise ValueError(f"no data in peak window {peak_window}")
    peak_ts = peak_seg.idxmax()
    trough_seg = curve.loc[peak_ts:pd.Timestamp(trough_end)]
    trough_ts = trough_seg.idxmin()
    return peak_ts, trough_ts


def episode_drawdown(returns: pd.Series, peak_ts, trough_ts) -> float:
    seg = returns.loc[peak_ts:trough_ts].dropna()
    if len(seg) < 2:
        return float("nan")
    curve = (1.0 + seg).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def annualized_carry(returns: pd.Series) -> float:
    return float(compute_metrics(returns.dropna())["cagr"])


def risk_match_weight(base_returns: pd.Series, pair_returns: pd.Series,
                      target_returns: pd.Series, lo: float = 0.0, hi: float = 3.0) -> float:
    """Smallest non-negative ``k`` with ``std(base + k*pair) == std(target)``.

    ``vol(k)**2 = var(b) + 2k*cov(b,p) + k**2*var(p)`` is U-shaped in ``k``; when the
    pair actually hedges the base (``cov(b,p) < 0``) its minimum sits at a positive
    ``k`` and a pure bisection can converge to the upper (levered) root. So: coarse
    grid-scan for the argmin, return NaN when ``target`` vol is below the achievable
    minimum (unreachable — *not* silently 0), else bisect on the branch that holds
    the economically intended (small-``k``) root.
    """
    df = pd.concat([base_returns.rename("b"), pair_returns.rename("p")], axis=1).dropna()
    target_vol = float(target_returns.dropna().std())

    def vol(k):
        return float((df["b"] + k * df["p"]).std())

    grid = np.linspace(lo, hi, 61)
    vols = [vol(k) for k in grid]
    j = int(np.argmin(vols))
    k_min, vol_min = float(grid[j]), vols[j]

    if target_vol <= vol_min:
        return float("nan")

    if vol(lo) >= target_vol:
        a, b, decreasing = lo, k_min, True      # intended: root left of the vertex, vol falls in k
    else:
        a, b, decreasing = k_min, hi, False     # target only reachable right of the vertex, vol rises in k

    for _ in range(80):
        mid = 0.5 * (a + b)
        if (vol(mid) < target_vol) == decreasing:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)
