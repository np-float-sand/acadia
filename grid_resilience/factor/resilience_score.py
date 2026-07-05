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


# Weight on each sub-component (must sum to 1.0 when all three are present)
_W_BETA  = 0.70
_W_RENEW = 0.15
_W_ICR   = 0.15

# Reporting lag: SEC 10-Q is due 40 days after quarter-end for large accelerated filers
_ICR_LAG_DAYS = 45


def build_factor(
    stress_betas: pd.DataFrame,
    renewable_share: pd.Series | None = None,
    icr: pd.Series | None = None,
    arch: str = "hard_switch",
    pass_through: pd.Series | None = None,
) -> pd.Series:
    """
    Construct a single cross-sectional Grid Resilience factor score.

    Parameters
    ----------
    stress_betas    : output of conditional_beta.compute_stress_betas()
                      Must have index = ticker and column 'stress_beta'.
    renewable_share : optional Series keyed by ticker of renewable gen share (0–1).
    icr             : optional Series keyed by ticker of interest coverage ratio.
    arch            : blending architecture when pass_through is provided.
                      One of "hard_switch", "revenue_mix", "dual_track".
                      Ignored when pass_through is None.
    pass_through    : optional Series keyed by ticker (0.0–1.0) — fraction of
                      revenue exposed to merchant/spot prices. When provided,
                      the specified arch is used to blend the beta and ICR signals.
                      When None, the original weight-based blending is used.

    Returns
    -------
    Series indexed by ticker, values in ~[-1, +1].
    High score = more resilient (candidate for long book).
    Low score  = stress-sensitive (candidate for short book).
    """
    if stress_betas.empty:
        return pd.Series(dtype=float)

    scores = pd.DataFrame(index=stress_betas.index)

    # Component 1: signed stress beta (always computed — needed for all archs)
    scores["signed_stress_beta"] = stress_betas["stress_beta"]
    scores["signed_stress_beta"] = cross_section_zscore(
        winsorize(scores["signed_stress_beta"], WINSOR_LIMITS)
    )

    has_renew = renewable_share is not None and not renewable_share.empty
    has_icr   = icr is not None and not icr.empty

    # ── Business-model-aware architectures ───────────────────────────────────
    if pass_through is not None:
        pt = pass_through.reindex(scores.index).fillna(1.0)  # unknown → treat as merchant

        # Pre-compute ICR z-score (used by all three archs)
        icr_z = pd.Series(0.0, index=scores.index)
        if has_icr:
            icr_raw = icr.reindex(scores.index)
            icr_raw = icr_raw.fillna(icr_raw.mean())        # fill gaps with cross-section mean
            icr_z = cross_section_zscore(winsorize(icr_raw, WINSOR_LIMITS))

        beta_z = scores["signed_stress_beta"]                # already z-scored above

        # When ICR is unavailable, all arch variants fall back to pure beta
        # (revenue_mix/dual_track would otherwise collapse regulated scores to ~0)
        if not has_icr:
            arch_score = beta_z

        elif arch == "hard_switch":
            is_regulated = pt < 0.5
            arch_score = beta_z.copy()
            arch_score[is_regulated] = icr_z[is_regulated]

        elif arch == "revenue_mix":
            arch_score = beta_z * pt + icr_z * (1 - pt)

        elif arch == "dual_track":
            merchant_mask  = pt > 0.05
            regulated_mask = pt < 1.0

            # Z-score beta within the merchant subgroup
            beta_zd = pd.Series(0.0, index=scores.index)
            if merchant_mask.sum() > 1:
                raw_beta = winsorize(stress_betas["stress_beta"][merchant_mask], WINSOR_LIMITS)
                beta_zd[merchant_mask] = cross_section_zscore(raw_beta)
            else:
                beta_zd = beta_z

            # Z-score ICR within the regulated subgroup
            icr_zd = pd.Series(0.0, index=scores.index)
            if regulated_mask.sum() > 1:
                icr_raw_d = icr.reindex(scores.index).fillna(icr.mean())
                icr_zd[regulated_mask] = cross_section_zscore(
                    winsorize(icr_raw_d[regulated_mask], WINSOR_LIMITS)
                )

            arch_score = beta_zd * pt + icr_zd * (1 - pt)

        else:
            arch_score = beta_z  # unknown arch → fall back to pure beta

        # Renewable quality: 15% addon for all tickers regardless of arch
        factor = 0.85 * arch_score
        if has_renew:
            renew_aligned = renewable_share.reindex(scores.index).fillna(0.0)
            renew_z = cross_section_zscore(winsorize(renew_aligned, WINSOR_LIMITS))
            factor = factor + 0.15 * renew_z

        factor = cross_section_zscore(winsorize(factor, WINSOR_LIMITS))
        return factor.rename("grid_resilience_factor")

    # ── Original weight-based blending (pass_through=None, backward-compatible) ──
    w_beta  = _W_BETA  + (0 if has_renew else _W_RENEW) + (0 if has_icr else _W_ICR)
    w_renew = _W_RENEW if has_renew else 0.0
    w_icr   = _W_ICR   if has_icr   else 0.0

    factor = w_beta * scores["signed_stress_beta"]

    if has_renew:
        renew_aligned = renewable_share.reindex(scores.index).fillna(0.0)
        scores["renewable_quality"] = cross_section_zscore(
            winsorize(renew_aligned, WINSOR_LIMITS)
        )
        factor = factor + w_renew * scores["renewable_quality"]

    if has_icr:
        icr_aligned = icr.reindex(scores.index)
        scores["icr_score"] = cross_section_zscore(
            winsorize(icr_aligned, WINSOR_LIMITS)
        )
        factor = factor + w_icr * scores["icr_score"]

    factor = cross_section_zscore(winsorize(factor, WINSOR_LIMITS))
    return factor.rename("grid_resilience_factor")


def build_rolling_factor(
    rolling_betas: pd.DataFrame,
    renewable_share_by_period: pd.DataFrame | None = None,
    icr_history: pd.DataFrame | None = None,
    arch: str = "hard_switch",
    pass_through: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build time-varying factor scores from rolling_stress_betas output.

    Parameters
    ----------
    rolling_betas              : MultiIndex (date, ticker) DataFrame from
                                 conditional_beta.rolling_stress_betas()
    renewable_share_by_period  : Optional DataFrame indexed by (period, ticker).
    icr_history                : Optional DataFrame indexed by quarter-end date
                                 with ticker columns (output of fetch_icr).
                                 A 45-day reporting lag is applied automatically.
    arch                       : blending architecture — passed to build_factor().
    pass_through               : Series keyed by ticker (0.0–1.0) — passed to
                                 build_factor(). None = original behaviour.

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

        icr = _icr_at_date(icr_history, date)

        scores = build_factor(
            betas_at_date,
            renewable_share=renew,
            icr=icr,
            arch=arch,
            pass_through=pass_through,
        )
        for ticker, score in scores.items():
            rows.append({"date": date, "ticker": ticker, "factor_score": score})

    return pd.DataFrame(rows).sort_values(["date", "factor_score"], ascending=[True, False])


def _icr_at_date(
    icr_history: pd.DataFrame | None,
    date: pd.Timestamp,
) -> pd.Series | None:
    """Return the most recently available ICR row as of `date` (with reporting lag)."""
    if icr_history is None or icr_history.empty:
        return None
    cutoff = date - pd.Timedelta(days=_ICR_LAG_DAYS)
    available = icr_history[icr_history.index <= cutoff]
    if available.empty:
        return None
    return available.iloc[-1]


def factor_quintile_ranks(factor_scores: pd.Series) -> pd.Series:
    """
    Assign each ticker to a quintile rank (1=worst, 5=best) based on factor score.
    Used for portfolio construction.
    """
    return pd.qcut(factor_scores, q=5, labels=[1, 2, 3, 4, 5]).astype(int)
