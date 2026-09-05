"""Utility capex-guidance revision signal ("Deliverable D").

Spec: docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md

Aggregate US electric utilities revise their forward 5-yr capital-program
guidance at each earnings call / 10-K / analyst day; since ~2023 those
revisions are increasingly data-center-driven and increasingly stated as
such. This module turns the hand/web-assembled panel in
`grid_resilience.data.utility_capex_guidance` into (1) a daily z-scored
composite, (2) a one-directional de-risk multiplier and a two-sided scaler
for the grid-equipment basket's exposure, and (3) the pre-registered gate
report testing both as a timing signal and as an exposure scaler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from grid_equipment_basket import config
from grid_equipment_basket.ftr_signal import broadcast_daily
from grid_equipment_basket.grid_regime import _trailing_zscore


def guidance_composite(events: pd.Series, start: str, end: str, *,
                       zscore_window: int = config.DC_GUIDANCE_ZSCORE_WINDOW,
                       zscore_minp: int = config.DC_GUIDANCE_ZSCORE_MINP,
                       winsor: float = config.DC_GUIDANCE_ZSCORE_WINSOR) -> pd.Series:
    """`events` indexed by `report_date` (already point-in-time -- see
    `utility_capex_guidance.aggregate_revision_series`). Forward-filled onto a
    daily calendar index over `[start, end]`, then trailing-z-scored
    (`grid_regime._trailing_zscore`'s rolling window only uses
    past-and-current observations, so this composite carries the same
    no-future-leak contract as `ftr_signal.ftr_composite` /
    `grid_regime.regime_composite`)."""
    if events.empty:
        return pd.Series(dtype=float, name="guidance_composite")
    daily = broadcast_daily(events, start, end)
    return _trailing_zscore(daily, zscore_window, zscore_minp, winsor).rename("guidance_composite")


def guidance_derisk_multiplier(composite: pd.Series, *,
                               floor_z: float = config.DC_GUIDANCE_DERISK_FLOOR_Z,
                               lo_mult: float = config.DC_GUIDANCE_DERISK_LO_MULT) -> pd.Series:
    """`{lo_mult, 1.0}` -- `lo_mult` on any day the composite z-score is below
    `floor_z` (revision-flow decelerating/reversing), back to 1.0 once it
    clears. NaN days (not-yet-warm) default to 1.0. Never exceeds 1.0. Same
    one-directional shape as `capex_signal.capex_derisk_multiplier` (spec s6.1)."""
    lo = composite < floor_z
    return pd.Series(np.where(lo.fillna(False), lo_mult, 1.0),
                     index=composite.index).rename("guidance_derisk_mult")


def guidance_scaler(composite: pd.Series, *,
                    k: float = config.DC_GUIDANCE_SCALER_K,
                    lo: float = config.DC_GUIDANCE_SCALER_LO,
                    hi: float = config.DC_GUIDANCE_SCALER_HI) -> pd.Series:
    """`clip(1 + k*z, lo, hi)` -- two-sided: leans in when the composite is
    positive (revision-flow accelerating), leans out when negative. NaN days
    default to 1.0. Same shape as `grid_regime`'s continuous-mode multiplier
    (spec s6.2)."""
    z = composite.fillna(0.0)
    raw = (1.0 + k * z).clip(lower=lo, upper=hi)
    return raw.where(composite.notna(), 1.0).rename("guidance_scaler")


def _hac_ols(y: pd.Series, X: pd.DataFrame, *, lag: int) -> dict:
    """OLS of `y` on `X` (a constant is added automatically) with Newey-West
    HAC standard errors at `lag`. Used both for the rank-IC t-stat (on ranked
    series) and the multi-control regression (on raw series) -- spec s5.1/s5.2.
    Returns `{"coef": {col: value}, "t": {col: value}, "n": n_obs}`; a NaN dict
    when there aren't enough observations to fit, rather than raising."""
    import statsmodels.api as sm

    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    cols = list(X.columns)
    if len(frame) < len(cols) + 3:
        return {"coef": {c: np.nan for c in cols}, "t": {c: np.nan for c in cols}, "n": len(frame)}
    yy = frame["__y__"]
    XX = sm.add_constant(frame[cols])
    fit = sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": max(int(lag), 1)})
    return {"coef": {c: float(fit.params[c]) for c in cols},
           "t": {c: float(fit.tvalues[c]) for c in cols}, "n": int(len(frame))}


def _rank_ic(signal: pd.Series, fwd_return: pd.Series, *, lag: int) -> dict:
    """Spearman rank-IC (point estimate via `scipy.stats.spearmanr`) plus a
    serial-correlation-robust t-stat: `_hac_ols` of `rank(fwd_return)` on
    `rank(signal)` with Newey-West lag=`lag` (the overlapping-forward-window
    horizons h=3/6 have serially correlated residuals month to month, which a
    plain Spearman significance test would understate). Spec s5.1."""
    from scipy.stats import spearmanr

    pair = pd.concat([signal.rename("s"), fwd_return.rename("r")], axis=1).dropna()
    if len(pair) < 5:
        return {"ic": np.nan, "t": np.nan, "n": len(pair)}
    ic, _ = spearmanr(pair["s"], pair["r"])
    ranks = pair.rank()
    hac = _hac_ols(ranks["r"], ranks[["s"]].rename(columns={"s": "signal"}), lag=lag)
    return {"ic": float(ic), "t": hac["t"]["signal"], "n": len(pair)}


def _monthly_nav(daily_ret: pd.Series) -> pd.Series:
    """Month-end NAV level (base 1.0) from a daily simple-return series."""
    nav = (1.0 + daily_ret.fillna(0.0)).cumprod()
    return nav.resample("ME").last()


def _forward_return(monthly_nav: pd.Series, h: int) -> pd.Series:
    """At each month-end, the realized return over the NEXT `h` months
    (`nav[t+h] / nav[t] - 1`); NaN for the trailing `h` month-ends where the
    future NAV isn't known yet."""
    return (monthly_nav.shift(-h) / monthly_nav - 1.0).rename(f"fwd_{h}m")
