"""
Sector-neutral long/short portfolio construction.

Given a cross-section of Grid Resilience factor scores, this module:
  1. Ranks tickers by score
  2. Selects top-N for the long book, bottom-N for the short book
  3. Assigns equal weights within each book
  4. Ensures dollar-neutrality (long weights sum = 0.5, short = -0.5)

The portfolio is re-built at each rebalance date specified in config.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from grid_resilience.config import PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N, REBALANCE_FREQ


def build_weights(
    factor_scores: pd.Series,
    n_long:  int = PORTFOLIO_LONG_N,
    n_short: int = PORTFOLIO_SHORT_N,
) -> pd.Series:
    """
    Construct dollar-neutral long/short weights from a factor score vector.

    Parameters
    ----------
    factor_scores : Series indexed by ticker (higher = more resilient = long)
    n_long        : number of names in the long book
    n_short       : number of names in the short book

    Returns
    -------
    Series of portfolio weights indexed by ticker.
    Longs sum to +0.5, shorts sum to -0.5 → portfolio is dollar-neutral.
    """
    scores = factor_scores.dropna().sort_values(ascending=False)

    if len(scores) < n_long + n_short:
        n_long  = max(1, len(scores) // 2)
        n_short = max(1, len(scores) - n_long)

    long_tickers  = scores.iloc[:n_long].index
    short_tickers = scores.iloc[-n_short:].index

    weights = pd.Series(0.0, index=scores.index)
    weights[long_tickers]  =  0.5 / n_long
    weights[short_tickers] = -0.5 / n_short

    return weights


def build_rolling_weights(
    rolling_factors: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex | None = None,
    n_long:  int = PORTFOLIO_LONG_N,
    n_short: int = PORTFOLIO_SHORT_N,
) -> pd.DataFrame:
    """
    Build a time series of portfolio weights, one set per rebalance date.

    Parameters
    ----------
    rolling_factors : DataFrame with columns [date, ticker, factor_score]
                      from factor.resilience_score.build_rolling_factor()
    rebalance_dates : if None, uses all unique dates in rolling_factors
    n_long / n_short: book sizes

    Returns
    -------
    DataFrame with columns [date, ticker, weight].
    """
    if rebalance_dates is None:
        rebalance_dates = pd.DatetimeIndex(rolling_factors["date"].unique())

    rows = []
    for date in rebalance_dates:
        day_factors = rolling_factors[rolling_factors["date"] == date]
        if day_factors.empty:
            continue
        scores  = day_factors.set_index("ticker")["factor_score"]
        weights = build_weights(scores, n_long=n_long, n_short=n_short)
        for ticker, w in weights.items():
            if w != 0.0:
                rows.append({"date": date, "ticker": ticker, "weight": w})

    return pd.DataFrame(rows)


def weights_to_matrix(
    weights_df: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    all_tickers: list[str],
) -> pd.DataFrame:
    """
    Pivot weights_df into a (date × ticker) matrix.
    Weights are forward-filled between rebalance dates.
    Missing tickers are filled with 0.
    """
    pivot = (
        weights_df
        .pivot(index="date", columns="ticker", values="weight")
        .reindex(columns=all_tickers, fill_value=0.0)
    )
    pivot = pivot.reindex(all_dates).ffill().fillna(0.0)
    return pivot


def portfolio_summary(weights: pd.Series) -> dict:
    """Return a human-readable summary of a weight vector."""
    longs  = weights[weights > 0].sort_values(ascending=False)
    shorts = weights[weights < 0].sort_values()
    return {
        "long_tickers":  list(longs.index),
        "short_tickers": list(shorts.index),
        "long_weight":   round(longs.sum(), 4),
        "short_weight":  round(shorts.sum(), 4),
        "n_names":       len(longs) + len(shorts),
    }
