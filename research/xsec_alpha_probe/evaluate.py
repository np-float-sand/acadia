"""Cross-sectional signal evaluation: winsorize, IC, t-stats, quantile books.

All frames are wide (month-end x ticker). A signal row at month-end t may use
only information available at t; the forward-return row at t is the return
earned *after* t. Alignment is the caller's job (see `run.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR_T: float = 1.5          # user-set bar (2026-09-24): |t| >= 1.5 pooled AND in both halves
MIN_NAMES: int = 15


def winsorize(df: pd.DataFrame, pct: float = 0.05) -> pd.DataFrame:
    """Clip each row to its own [pct, 1-pct] quantiles."""
    lo = df.quantile(pct, axis=1)
    hi = df.quantile(1 - pct, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)


def prep(signal: pd.DataFrame, pct: float = 0.05) -> pd.DataFrame:
    """The one preprocessing every signal goes through: winsorize then z-score."""
    return zscore(winsorize(signal, pct))


def ic_series(signal: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman",
              min_names: int = MIN_NAMES) -> pd.Series:
    """Per-date cross-sectional correlation between signal and forward return."""
    signal, fwd = signal.align(fwd, join="inner")
    out = {}
    for dt in signal.index:
        s, r = signal.loc[dt], fwd.loc[dt]
        ok = s.notna() & r.notna()
        if ok.sum() < min_names:
            out[dt] = np.nan
            continue
        out[dt] = s[ok].corr(r[ok], method=method)
    return pd.Series(out, dtype=float)


def nw_tstat(x: pd.Series, lags: int = 0) -> float:
    """Newey-West t-stat of the mean (Bartlett kernel)."""
    x = pd.Series(x).dropna().to_numpy(float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else np.nan


def verdict(t_full: float, t_h1: float, t_h2: float, bar: float = BAR_T) -> str:
    """PASS: pooled and both halves clear the bar with one sign.
    CLOSE: pooled clears, halves agree in sign, one half short of the bar.
    """
    ts = np.array([t_full, t_h1, t_h2], dtype=float)
    if np.isnan(ts).any():
        return "N/A"
    signs = np.sign(ts)
    if not (signs == signs[0]).all():
        return "FAIL"
    tag = "" if signs[0] > 0 else " (inverted)"
    if (np.abs(ts) >= bar).all():
        return "PASS" + tag
    if abs(t_full) >= bar and np.abs(ts[1:]).min() < bar:
        return "CLOSE" + tag
    return "FAIL"


def semivol(returns: pd.DataFrame) -> pd.Series:
    """Downside semi-deviation (below zero), per column."""
    neg = returns.clip(upper=0)
    return np.sqrt((neg ** 2).mean())


def bucket_weights(sig: pd.Series, q: float = 0.2, top: bool = True,
                   scale: pd.Series | None = None) -> pd.Series:
    """Weights on the top (or bottom) q of a cross-section, summing to 1.

    `scale` is a per-name risk estimate; when given, weights are proportional to
    1/scale (inverse vol or inverse downside semivol) inside the bucket.
    """
    s = sig.dropna()
    n = max(1, int(round(len(s) * q)))
    names = s.nlargest(n).index if top else s.nsmallest(n).index
    if scale is None:
        raw = pd.Series(1.0, index=names)
    else:
        r = scale.reindex(names)
        raw = (1.0 / r.where(r > 0)).dropna()
        if raw.empty:
            raw = pd.Series(1.0, index=names)
    w = raw / raw.sum()
    return w.reindex(sig.index).fillna(0.0)


def perf(r: pd.Series, periods: int = 12) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return {"ann": np.nan, "vol": np.nan, "sharpe": np.nan, "t": np.nan, "maxdd": np.nan}
    ann = r.mean() * periods
    vol = r.std() * np.sqrt(periods)
    eq = (1 + r).cumprod()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol > 0 else np.nan,
            "t": nw_tstat(r), "maxdd": float((eq / eq.cummax() - 1).min())}
