from __future__ import annotations

"""Hedge overlays and drawdown-episode scoring for the value-chain reframe
(spec 2026-08-29 §6, §7). Simple-return convention."""

import numpy as np
import pandas as pd

from grid_equipment_basket import config, value_chain
from grid_equipment_basket.basket import simulate_basket


def simulate_pair(prices: pd.DataFrame, start: str, end: str,
                  fund_df: pd.DataFrame, backlog_df: pd.DataFrame,
                  lag_days: int = 42) -> pd.Series:
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
