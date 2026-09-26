"""Point-in-time cross-sectional quality composite for the `screen` universe --
RESEARCH-LOG open item 13, docs/quality-weighting-preregistration.md §3.

Not production code. Throwaway research script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_STABILITY_LOOKBACK_Q = 8   # trailing quarters of gross margin used for the std
_MIN_STABILITY_Q = 6        # need at least this many known quarters to trust it
_TILT_COEF = 0.25           # matches backlog_tilt_targets / coverage_tilt_targets's +-25%
_Z_CLIP = 2.0


def _visible(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df[df["availability_date"] <= pd.Timestamp(asof)]


def leverage_at(leverage_by_ticker: dict, tickers, asof) -> pd.Series:
    out = {}
    for t in tickers:
        df = leverage_by_ticker.get(t)
        v = _visible(df, asof) if df is not None else None
        out[t] = float(v.sort_values("quarter_end")["leverage"].iloc[-1]) if v is not None and not v.empty else float("nan")
    return pd.Series(out, dtype=float)


def margin_stability_at(margins_by_ticker: dict, tickers, asof) -> pd.Series:
    out = {}
    for t in tickers:
        df = margins_by_ticker.get(t)
        v = _visible(df, asof) if df is not None else None
        if v is None or len(v) < _MIN_STABILITY_Q:
            out[t] = float("nan")
            continue
        window = v.sort_values("quarter_end").tail(_STABILITY_LOOKBACK_Q)["gross_margin"]
        out[t] = float(-window.std()) if window.notna().sum() >= _MIN_STABILITY_Q else float("nan")
    return pd.Series(out, dtype=float)


def fcf_conversion_at(fcf_by_ticker: dict, tickers, asof) -> pd.Series:
    out = {}
    for t in tickers:
        df = fcf_by_ticker.get(t)
        v = _visible(df, asof) if df is not None else None
        out[t] = float(v.sort_values("quarter_end")["fcf_conversion"].iloc[-1]) if v is not None and not v.empty else float("nan")
    return pd.Series(out, dtype=float)


def _winsorize(s: pd.Series, low=0.05, high=0.95) -> pd.Series:
    s = s.astype(float)
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile(low), s.quantile(high)
    return s.clip(lower=lo, upper=hi)


def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.notna().sum() < 3 or not (s.std(skipna=True) > 0):
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean(skipna=True)) / s.std(skipna=True)


def composite_z(tickers, asof, leverage_by_ticker, margins_by_ticker, fcf_by_ticker) -> pd.Series:
    """Equal-weighted composite z-score (pre-registration §3): mean(z(-leverage),
    z(margin_stability), z(fcf_conversion)), computed only for names with ALL
    THREE inputs known as of `asof`. Missing any one -> NaN (caller treats NaN as
    neutral -- weight factor 1.0)."""
    tickers = list(tickers)
    lev = _winsorize(leverage_at(leverage_by_ticker, tickers, asof))
    stab = _winsorize(margin_stability_at(margins_by_ticker, tickers, asof))
    fcf = _winsorize(fcf_conversion_at(fcf_by_ticker, tickers, asof))

    z_lev = -_zscore(lev)
    z_stab = _zscore(stab)
    z_fcf = _zscore(fcf)

    z_df = pd.DataFrame({"lev": z_lev, "stab": z_stab, "fcf": z_fcf}).reindex(tickers)
    complete = z_df.dropna(how="any")
    comp = complete.mean(axis=1)
    return comp.reindex(tickers)


def weight_factor(comp_z: pd.Series) -> pd.Series:
    """Pre-registration §3 step 5: factor = 1 + 0.25 * clip(z, -2, 2); NaN -> 1.0."""
    f = 1.0 + _TILT_COEF * comp_z.clip(lower=-_Z_CLIP, upper=_Z_CLIP)
    return f.fillna(1.0)
