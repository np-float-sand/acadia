from __future__ import annotations

"""Backlog/RPO growth-surprise signal (spec 2026-08-31 §4). No fitted model:
expected_g is the trailing 4-quarter mean of g."""

import numpy as np
import pandas as pd

from backlog_factor import config
from grid_resilience.factor.neutralize import cross_section_zscore

_BREAK = np.log(3.0)   # |single-quarter log change| beyond this = suspected RPO-definition break


def name_surprise_series(rpo_one: pd.DataFrame, *, min_quarters: int = config.HISTORY_MIN_QUARTERS,
                         span_min: int = config.SPAN_MIN_DAYS,
                         span_max: int = config.SPAN_MAX_DAYS) -> pd.DataFrame:
    g_df = rpo_one.sort_values("quarter_end").reset_index(drop=True)
    g_df["g"] = np.log(g_df["metric_value"] / g_df["metric_value"].shift(1))
    g_df["expected_g"] = g_df["g"].shift(1).rolling(4).mean()
    g_df["surprise"] = g_df["g"] - g_df["expected_g"]

    # history + span guards: need >= min_quarters rows and a clean iloc[i-4]..iloc[i] span
    span_ok = (g_df["quarter_end"] - g_df["quarter_end"].shift(4)).dt.days.between(span_min, span_max)
    g_df.loc[(g_df.index < min_quarters - 1) | ~span_ok, "surprise"] = np.nan

    # RPO-definition-break screen: a |g| spike NaNs this row and the next 4
    for i in g_df.index[g_df["g"].abs() > _BREAK].tolist():
        g_df.loc[i: i + 4, "surprise"] = np.nan

    return g_df[["availability_date", "g", "expected_g", "surprise"]]


def surprise_panel(rpo_panel: pd.DataFrame, *, min_quarters: int = config.HISTORY_MIN_QUARTERS,
                   span_min: int = config.SPAN_MIN_DAYS,
                   span_max: int = config.SPAN_MAX_DAYS) -> pd.DataFrame:
    parts = []
    for tkr, g in rpo_panel.groupby("ticker"):
        s = name_surprise_series(g, min_quarters=min_quarters, span_min=span_min, span_max=span_max)
        s = s.dropna(subset=["surprise"]).copy()
        s["ticker"] = tkr
        parts.append(s[["ticker", "availability_date", "surprise"]])
    if not parts:
        return pd.DataFrame(columns=["ticker", "availability_date", "surprise"])
    return pd.concat(parts, ignore_index=True).sort_values(
        ["availability_date", "ticker"]).reset_index(drop=True)


def _winsorize_sigma(s: pd.Series, sigma: float) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s
    return s.clip(mu - sigma * sd, mu + sigma * sd)


def neutralize_cross_section(raw: pd.Series, industry_group: pd.Series, log_mktcap: pd.Series,
                             winsor_sigma: float = config.WINSOR_SIGMA) -> pd.Series:
    names = [t for t in raw.index if t in industry_group.index and t in log_mktcap.index]
    if len(names) < 2:
        return pd.Series(dtype=float)
    z = cross_section_zscore(_winsorize_sigma(raw.reindex(names).astype(float), winsor_sigma))
    grp = industry_group.reindex(names).astype("category")
    X = pd.get_dummies(grp, drop_first=False).astype(float)
    X["_lmc"] = log_mktcap.reindex(names).astype(float).to_numpy()
    X["_const"] = 1.0
    Xv = X.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Xv, z.to_numpy(dtype=float), rcond=None)
    resid = z.to_numpy(dtype=float) - Xv @ beta
    return pd.Series(resid, index=names)


def staleness_decay(days_since, horizon_days: int):
    frac = np.clip(1.0 - (np.asarray(days_since, dtype=float) / float(horizon_days)), 0.0, 1.0)
    if isinstance(days_since, pd.Series):
        return pd.Series(frac, index=days_since.index)
    return float(frac) if np.ndim(frac) == 0 else frac


def signal_on_date(sp: pd.DataFrame, asof: pd.Timestamp, *, industry_group_map: dict,
                   log_mktcap: pd.Series, horizon_days: int,
                   staleness_max_days: int = config.STALENESS_MAX_DAYS,
                   winsor_sigma: float = config.WINSOR_SIGMA) -> pd.Series:
    asof = pd.Timestamp(asof)
    vis = sp[sp["availability_date"] <= asof]
    if vis.empty:
        return pd.Series(dtype=float)
    latest = (vis.sort_values("availability_date").groupby("ticker").tail(1).set_index("ticker"))
    age = (asof - latest["availability_date"]).dt.days
    keep = latest[age <= staleness_max_days].copy()
    if keep.empty:
        return pd.Series(dtype=float)
    keep_age = (asof - keep["availability_date"]).dt.days
    decayed = keep["surprise"].astype(float) * staleness_decay(keep_age, horizon_days)
    grp = pd.Series({t: industry_group_map.get(t) for t in decayed.index})
    return neutralize_cross_section(decayed, grp, log_mktcap, winsor_sigma).dropna()
