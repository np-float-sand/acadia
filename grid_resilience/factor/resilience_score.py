"""
Grid Resilience Factor construction.

The factor score for each utility is assembled from two components:

  1. Stress beta (primary, ~80% weight)
     Source: conditional_beta.compute_stress_betas()
     Higher stress beta = more resilient (generators profit from LMP spikes;
     positive = long candidate). More negative stress beta = more hurt by grid
     stress (T&D names; negative = short candidate). The raw stress beta is used
     directly — no sign flip — because the OLS regression already encodes
     the correct direction.

  2. Renewable quality adjustment (secondary, ~20% weight)
     Source: eia_data.compute_renewable_share()
     Utilities with higher renewable generation share score better,
     but only if they also have the storage depth to back it up
     (penalise pure intermittent exposure without storage buffer).

The combined raw score is cross-sectionally z-scored, winsorised,
and rescaled to [-1, +1] for portfolio use.
"""

from __future__ import annotations

import pandas as pd

from grid_resilience.config import WINSOR_LIMITS
from grid_resilience.factor.neutralize import winsorize, cross_section_zscore


# Weight on each sub-component
_W_BETA = 0.80
_W_RENEW = 0.20


def build_factor(
    stress_betas: pd.DataFrame,
    renewable_share: pd.Series | None = None,
) -> pd.Series:
    """
    Construct a single cross-sectional Grid Resilience factor score.

    Parameters
    ----------
    stress_betas     : output of conditional_beta.compute_stress_betas()
                       Must have index = ticker and column 'stress_beta'.
    renewable_share  : optional Series keyed by ticker of renewable gen share (0–1).
                       If None, only the stress-beta component is used.

    Returns
    -------
    Series indexed by ticker, values in ~[-1, +1].
    High score = more resilient (candidate for long book).
    Low score  = stress-sensitive (candidate for short book).
    """
    if stress_betas.empty:
        return pd.Series(dtype=float)

    scores = pd.DataFrame(index=stress_betas.index)

    # Component 1: signed stress beta
    # Positive beta = stock benefits from grid stress (generators profiting from high LMPs).
    # Negative beta = stock is hurt by grid stress (T&D facing higher costs).
    # Using the raw beta directly preserves this semantics for all ticker types.
    scores["signed_stress_beta"] = stress_betas["stress_beta"]
    scores["signed_stress_beta"] = cross_section_zscore(
        winsorize(scores["signed_stress_beta"], WINSOR_LIMITS)
    )

    # Component 2: renewable quality (optional)
    if renewable_share is not None and not renewable_share.empty:
        renew_aligned = renewable_share.reindex(scores.index).fillna(0.0)
        scores["renewable_quality"] = cross_section_zscore(
            winsorize(renew_aligned, WINSOR_LIMITS)
        )
        factor = _W_BETA * scores["signed_stress_beta"] + _W_RENEW * scores["renewable_quality"]
    else:
        factor = scores["signed_stress_beta"]

    # Final cross-sectional normalisation
    factor = cross_section_zscore(winsorize(factor, WINSOR_LIMITS))
    return factor.rename("grid_resilience_factor")


def build_rolling_factor(
    rolling_betas: pd.DataFrame,
    renewable_share_by_period: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build time-varying factor scores from rolling_stress_betas output.

    Parameters
    ----------
    rolling_betas              : MultiIndex (date, ticker) DataFrame from
                                 conditional_beta.rolling_stress_betas()
    renewable_share_by_period  : Optional DataFrame indexed by (period, ticker)
                                 with 'renewable_share' column.

    Returns
    -------
    DataFrame with columns [date, ticker, factor_score], sorted by date.
    """
    if rolling_betas.empty:
        return pd.DataFrame()

    rows = []
    dates = rolling_betas.index.get_level_values("date").unique()

    for date in dates:
        betas_at_date = rolling_betas.loc[date]

        renew = None
        if renewable_share_by_period is not None:
            period_key = date.strftime("%Y-%m")
            if period_key in renewable_share_by_period.index:
                renew = renewable_share_by_period.loc[period_key]

        scores = build_factor(betas_at_date, renewable_share=renew)
        for ticker, score in scores.items():
            rows.append({"date": date, "ticker": ticker, "factor_score": score})

    return pd.DataFrame(rows).sort_values(["date", "factor_score"], ascending=[True, False])


def factor_quintile_ranks(factor_scores: pd.Series) -> pd.Series:
    """
    Assign each ticker to a quintile rank (1=worst, 5=best) based on factor score.
    Used for portfolio construction.
    """
    return pd.qcut(factor_scores, q=5, labels=[1, 2, 3, 4, 5]).astype(int)
