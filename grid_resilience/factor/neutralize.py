"""
Factor neutralization utilities.

Removes known risk premia (sector average, market beta) from raw scores
before portfolio construction, as described in the strategy pitch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section_zscore(series: pd.Series) -> pd.Series:
    """Demean and scale by cross-sectional standard deviation."""
    mean = series.mean()
    std  = series.std()
    if std == 0 or np.isnan(std):
        return series - mean
    return (series - mean) / std


def winsorize(series: pd.Series, limits: tuple[float, float] = (0.025, 0.975)) -> pd.Series:
    """Clip series at the given quantile limits."""
    lo = series.quantile(limits[0])
    hi = series.quantile(limits[1])
    return series.clip(lo, hi)


def remove_sector_return(
    returns: pd.DataFrame,
    sector_return: pd.Series,
) -> pd.DataFrame:
    """
    Subtract the equal-weighted sector return from each ticker's return.
    Returns a DataFrame of sector-demeaned (excess) returns.
    """
    aligned_sector = sector_return.reindex(returns.index).fillna(0.0)
    return returns.subtract(aligned_sector, axis=0)


def remove_market_beta(
    excess_returns: pd.DataFrame,
    market_return: pd.Series,
    estimation_window: int = 252,
) -> pd.DataFrame:
    """
    Remove the market-beta component from excess returns using a rolling OLS.
    The residuals are the market-beta-neutralised excess returns.

    Parameters
    ----------
    excess_returns     : sector-demeaned daily returns (from remove_sector_return)
    market_return      : broad market return series (e.g., SPY log returns)
    estimation_window  : rolling window in trading days for beta estimation

    Returns
    -------
    DataFrame of residual returns with same shape as excess_returns.
    """
    residuals = pd.DataFrame(index=excess_returns.index, columns=excess_returns.columns, dtype=float)

    mkt = market_return.reindex(excess_returns.index).fillna(0.0)

    for ticker in excess_returns.columns:
        er = excess_returns[ticker].dropna()
        mkt_aligned = mkt.reindex(er.index)

        betas = (
            er.rolling(estimation_window, min_periods=60)
            .cov(mkt_aligned)
            / mkt_aligned.rolling(estimation_window, min_periods=60).var()
        )
        residuals[ticker] = er - betas * mkt_aligned

    return residuals.astype(float)


def neutralise_scores(
    factor_scores: pd.Series,
    group_map: dict[str, str] | None = None,
) -> pd.Series:
    """
    Demean factor scores within sub-groups (e.g., generator vs. wires-only).
    If group_map is None, a simple cross-sectional demean is applied.

    Parameters
    ----------
    factor_scores : Series indexed by ticker
    group_map     : {ticker: group_label}

    Returns
    -------
    Group-demeaned factor scores.
    """
    if group_map is None:
        return cross_section_zscore(factor_scores)

    result = factor_scores.copy()
    groups = pd.Series(group_map).reindex(factor_scores.index)

    for grp in groups.dropna().unique():
        mask = groups == grp
        grp_scores = factor_scores[mask]
        result[mask] = cross_section_zscore(grp_scores)

    return result
