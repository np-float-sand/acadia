"""Hedge overlay (spec 4.2): a 15% GLD/IEF diversifier sleeve plus an optional
0.25x DLR+EQIX short that engages only while the 6-month change in the 10-year
real yield is positive; plus an optional solar-rotation long (open item #17)
that engages only while the same 6-month real-yield change is negative.

The sleeve is the clean winner (raises Sharpe, +13% in the feared solar-squeeze
scenario). The short is a levered rising-real-yield bet -- 2-name concentration,
~2.5%/yr dividend paid while short, value concentrated in the 2022 rate shock and
the DeepSeek drawdown, ~15%/yr drag inside a rate-rising equity bull. The solar
rotation is the opposite-sign trade: a small (0.10x) conditional long in the
beaten-down DER sleeve (ENPH/SEDG/CHPT/RUN/BLNK/STEM), sized up only when real
yields are falling -- a convex cushion for the same feared scenario (2026-09-06
probe, `docs/electrification-strategy-v1-results.md`).
"""
from __future__ import annotations

import pandas as pd

from electrification_strategy import config
from electrification_strategy._util import month_hold


def _ew_returns(prices: pd.DataFrame, tickers) -> pd.Series:
    cols = [t for t in tickers if t in prices.columns]
    return prices[cols].pct_change().mean(axis=1).fillna(0.0)


def sleeve_returns(prices: pd.DataFrame) -> pd.Series:
    return _ew_returns(prices, config.HEDGE_SLEEVE_TICKERS)


def short_returns(prices: pd.DataFrame) -> pd.Series:
    return _ew_returns(prices, config.HEDGE_SHORT_TICKERS)


def solar_rotation_returns(prices: pd.DataFrame) -> pd.Series:
    return _ew_returns(prices, config.SOLAR_ROTATION_TICKERS)


def real_yield_rising_mask(dfii10: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    s = dfii10.astype(float).sort_index()
    rising = (s - s.shift(config.HEDGE_SHORT_LOOKBACK_DAYS)) > 0
    daily = rising.reindex(index.union(rising.index)).ffill().reindex(index).fillna(False)
    lagged = daily.shift(config.HEDGE_SHORT_LAG_DAYS).fillna(False)
    return month_hold(lagged.astype(float), index, fill=0.0) > 0.5


def real_yield_falling_mask(dfii10: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Same 6-month-change / 5-day-lag / month-held construction as
    `real_yield_rising_mask`, opposite sign -- the solar-rotation trigger
    (`scratchpad/utility_rotation.py`, 2026-09-06: fires ~42% of days)."""
    s = dfii10.astype(float).sort_index()
    falling = (s - s.shift(config.HEDGE_SHORT_LOOKBACK_DAYS)) < 0
    daily = falling.reindex(index.union(falling.index)).ffill().reindex(index).fillna(False)
    lagged = daily.shift(config.HEDGE_SHORT_LAG_DAYS).fillna(False)
    return month_hold(lagged.astype(float), index, fill=0.0) > 0.5


def apply_hedge(core_r: pd.Series, sleeve_r: pd.Series, short_r: pd.Series,
                mask: pd.Series, use_sleeve: bool, use_short: bool,
                solar_r: pd.Series | None = None, solar_mask: pd.Series | None = None,
                use_solar: bool = False) -> pd.Series:
    w_s = config.HEDGE_SLEEVE_WEIGHT if use_sleeve else 0.0
    w_h = config.HEDGE_SHORT_WEIGHT if use_short else 0.0
    idx = core_r.index
    c = core_r.fillna(0.0)
    s = sleeve_r.reindex(idx).fillna(0.0)
    h = short_r.reindex(idx).fillna(0.0)
    m = mask.reindex(idx).fillna(False).astype(float)
    out = (1.0 - w_s) * c + w_s * s - w_h * (h * m)
    if use_solar:
        w_r = config.SOLAR_ROTATION_WEIGHT
        d = solar_r.reindex(idx).fillna(0.0)
        rm = solar_mask.reindex(idx).fillna(False).astype(float)
        out = (1.0 - w_r * rm) * out + w_r * rm * d
    return out
