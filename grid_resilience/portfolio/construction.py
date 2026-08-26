"""
Sector-neutral long/short portfolio construction.

Given a cross-section of Grid Resilience factor scores, this module:
  1. Ranks tickers by score
  2. Selects top-N for the long book; either shorts bottom-N individual names
     (xlu_hedge=False) or takes a -0.5 position in XLU (xlu_hedge=True, default)
  3. Assigns equal weights within each book
  4. Ensures dollar-neutrality (long weights sum = 0.5, short = -0.5)

The portfolio is re-built at each rebalance date specified in config.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from grid_resilience.config import (
    PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N, REBALANCE_FREQ, XLU_HEDGE,
    PEER_GROUP_BOOK_SIZES,
)
from grid_resilience.data.utility_node_map import TICKER_NODE_MAP


def build_weights(
    factor_scores: pd.Series,
    n_long:    int  = PORTFOLIO_LONG_N,
    n_short:   int  = PORTFOLIO_SHORT_N,  # ignored when xlu_hedge=True
    xlu_hedge: bool = XLU_HEDGE,
) -> pd.Series:
    """
    Construct dollar-neutral long/short weights from a factor score vector.

    When xlu_hedge=True (default), the short book is replaced by a single
    -0.5 position in XLU (SPDR Utilities ETF) to avoid idiosyncratic short risk.
    When xlu_hedge=False, the bottom n_short names are shorted as before.

    XLU is always excluded from ranking even if it appears in factor_scores.
    """
    scores = factor_scores.drop("XLU", errors="ignore").dropna().sort_values(ascending=False)

    if len(scores) == 0:
        return pd.Series(dtype=float)  # no data → no position

    if not xlu_hedge and len(scores) < n_long + n_short:
        n_long  = max(1, len(scores) // 2)
        n_short = max(1, len(scores) - n_long)
    elif xlu_hedge and len(scores) < n_long:
        n_long = max(1, len(scores))

    long_tickers = scores.iloc[:n_long].index
    weights = pd.Series(0.0, index=scores.index)
    weights[long_tickers] = 0.5 / n_long

    if xlu_hedge:
        weights["XLU"] = -0.5
    else:
        short_tickers = scores.iloc[-n_short:].index
        weights[short_tickers] = -0.5 / n_short

    return weights


def build_grouped_weights(factor_scores: pd.Series) -> pd.Series:
    """
    Construct dollar-neutral long/short weights using peer-group (basket-vs-basket)
    construction: the universe is split into business_model sleeves (merchant, mixed,
    regulated), and build_weights() is called independently within each sleeve.

    This cancels sector-beta exposure that whole-universe ranking carries on both legs
    (see docs/superpowers/specs/2026-08-15-peer-group-construction-design.md) while
    preserving diversification within each leg. Each sleeve's capital share is
    proportional to its share of the groupable universe; sleeve outputs are concatenated
    into a single Series covering all input tickers.

    XLU is always excluded, matching build_weights(). Merchant uses the global
    PORTFOLIO_LONG_N/PORTFOLIO_SHORT_N defaults (shrinks to 1 long/1 short for its
    2-name universe via build_weights()'s existing fallback); mixed and regulated use
    the sizes in config.PEER_GROUP_BOOK_SIZES.
    """
    scores = factor_scores.drop("XLU", errors="ignore")

    groups: dict[str, list[str]] = {}
    for ticker in scores.index:
        model = TICKER_NODE_MAP.get(ticker, {}).get("business_model")
        if model in ("merchant", "mixed", "regulated"):
            groups.setdefault(model, []).append(ticker)

    total = sum(len(tickers) for tickers in groups.values())
    if total == 0:
        return pd.Series(dtype=float)

    sleeves = []
    for model, tickers in groups.items():
        sleeve_scores = scores.loc[tickers]
        if model == "merchant":
            sleeve_weights = build_weights(sleeve_scores, xlu_hedge=False)
        else:
            n_long, n_short = PEER_GROUP_BOOK_SIZES[model]
            sleeve_weights = build_weights(sleeve_scores, n_long=n_long, n_short=n_short, xlu_hedge=False)
        sleeves.append(sleeve_weights * (len(tickers) / total))

    if not sleeves:
        return pd.Series(dtype=float)

    return pd.concat(sleeves)


def build_rolling_weights(
    rolling_factors: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex | None = None,
    n_long:    int  = PORTFOLIO_LONG_N,
    n_short:   int  = PORTFOLIO_SHORT_N,
    xlu_hedge: bool = XLU_HEDGE,
    grouped:   bool = False,
) -> pd.DataFrame:
    """
    Build a time series of portfolio weights, one set per rebalance date.

    Parameters
    ----------
    rolling_factors : DataFrame with columns [date, ticker, factor_score]
                      from factor.resilience_score.build_rolling_factor()
    rebalance_dates : if None, uses all unique dates in rolling_factors
    n_long / n_short: book sizes (ignored when grouped=True — sizes come from
                      config.PEER_GROUP_BOOK_SIZES instead)
    xlu_hedge       : if True, replace short book with -0.5 XLU position
                      (ignored when grouped=True)
    grouped         : if True, use build_grouped_weights() (peer-group,
                      basket-vs-basket construction) instead of build_weights()

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
        scores = day_factors.set_index("ticker")["factor_score"]
        if grouped:
            weights = build_grouped_weights(scores)
        else:
            weights = build_weights(scores, n_long=n_long, n_short=n_short, xlu_hedge=xlu_hedge)
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
    # pandas 2.x/3.x: date_range produces datetime64[us] while yfinance returns
    # datetime64[ms]; reindex requires matching units so we normalise the pivot
    # index to match all_dates before reindexing.
    if not pivot.empty:
        try:
            pivot.index = pivot.index.as_unit(all_dates.unit)
        except AttributeError:
            pass  # pandas < 2.0 — no unit attribute, no mismatch issue
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
