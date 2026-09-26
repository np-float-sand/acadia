"""
Satellite-sleeve return construction + fixed-weight blend.

Follows the exact same mechanical convention already shipped in
`electrification_strategy/hedge_overlay.py` for the GLD/IEF sleeve -- no new
methodology is invented for this probe:

  - Within-sleeve weighting: equal-weight, implicitly rebalanced daily (the
    mean of each name's own daily simple return -- same form as
    `hedge_overlay._ew_returns`).
  - Blend into the base book: a FIXED weight applied every day,
    `(1 - w) * book_r + w * sleeve_r` -- same form as `hedge_overlay.
    apply_hedge`'s sleeve term. A constant daily weight is equivalent to
    rebalancing the book/sleeve split back to target every day; this project's
    hedge overlay uses no slower cadence for its own sleeve, so this probe
    matches it rather than inventing a monthly-hold convention.
"""
from __future__ import annotations

import pandas as pd


def sleeve_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Equal-weight daily simple-return series across `tickers`. Any name
    missing a price on a given day is simply excluded from that day's mean
    (pandas' default `.mean(axis=1)` behaviour) -- callers must restrict the
    index to each sleeve's `effective_start`-gated window (see data.py) rather
    than rely on this to silently reweight around missing names.
    """
    cols = [t for t in tickers if t in prices.columns]
    return prices[cols].pct_change().mean(axis=1)


def blend(book_r: pd.Series, sleeve_r: pd.Series, weight: float) -> pd.Series:
    """Fixed-weight daily blend, funded pro-rata out of the base book."""
    idx = book_r.index
    s = sleeve_r.reindex(idx)
    return (1.0 - weight) * book_r + weight * s
