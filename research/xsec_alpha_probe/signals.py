"""Candidate cross-sectional signals, each oriented so HIGHER = expected to outperform.

Every signal at month-end t uses only information available at t:
fundamentals filed strictly before t, prices up to t, power data for months <= t.

Families
--------
fundamental (point-in-time EDGAR, filed-date gated)
    rev_growth     TTM revenue YoY growth
    rev_accel      change in TTM revenue YoY growth vs one quarter earlier
    rev_sue        standardized unexpected revenue (seasonal random walk w/ drift)
    ni_sue         standardized unexpected net income (post-earnings drift)
    capex_light    -(TTM capex / TTM revenue)
    capex_restraint -(TTM capex YoY growth)            [asset-growth anomaly]
price (baselines)
    mom_12_1, reversal_1m
power (ISO factor exposure, see power.py)
    beta_<f>       rolling 36m exposure to the seasonally-differenced factor shock,
                   controlling for SPY
    bxs_<f>        beta_<f> x trailing-3m factor shock ("who benefits from the move")
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.xsec_alpha_probe import data, power

STALE_DAYS = 150          # latest quarter must end within this many days of t
LOOKBACK_YEARS = 5
SUE_MIN_HIST = 4
BETA_WINDOW = 36
BETA_MIN_OBS = 24
SHOCK_MONTHS = 3


# -- fundamentals --------------------------------------------------------------

def _regular_tail(s: pd.Series) -> pd.Series:
    """Longest run of consecutive quarters ending at the latest one (gaps 80-100 days)."""
    s = s.sort_index()
    if len(s) < 2:
        return s
    gaps = pd.Series(s.index).diff().dt.days.to_numpy()
    start = len(s) - 1
    while start > 0 and 80 <= gaps[start] <= 100:
        start -= 1
    return s.iloc[start:]


def _ttm(v: np.ndarray, lag: int) -> float:
    """Sum of 4 quarters ending `lag` quarters before the latest."""
    end = len(v) - lag
    return float(v[end - 4:end].sum()) if end >= 4 else np.nan


def _sue(v: np.ndarray) -> float:
    if len(v) < 4 + SUE_MIN_HIST + 1:
        return np.nan
    x = v[4:] - v[:-4]                      # seasonal differences
    hist, x0 = x[-(SUE_MIN_HIST + 4):-1], x[-1]
    floor = 0.01 * np.abs(v[-8:]).mean()
    sd = max(hist.std(ddof=1), floor, 1e-12)
    return float((x0 - hist.mean()) / sd)


def _features_from_series(rev: pd.Series, ni: pd.Series | None,
                          capex: pd.Series | None) -> dict:
    out = dict.fromkeys(["rev_growth", "rev_accel", "rev_sue", "ni_sue",
                         "capex_light", "capex_restraint"], np.nan)
    r = rev.to_numpy(float)
    if len(r) >= 8:
        t0, t4 = _ttm(r, 0), _ttm(r, 4)
        if t4 > 0:
            out["rev_growth"] = t0 / t4 - 1
        if len(r) >= 9:
            t1, t5 = _ttm(r, 1), _ttm(r, 5)
            if t4 > 0 and t5 > 0:
                out["rev_accel"] = (t0 / t4 - 1) - (t1 / t5 - 1)
    out["rev_sue"] = _sue(r)
    if ni is not None and len(ni):
        out["ni_sue"] = _sue(ni.to_numpy(float))
    if capex is not None and len(capex) >= 4 and capex.index[-1] == rev.index[-1]:
        c = np.abs(capex.to_numpy(float))
        rev_ttm = _ttm(r, 0)
        if rev_ttm > 0:
            out["capex_light"] = -_ttm(c, 0) / rev_ttm
        if len(c) >= 8 and _ttm(c, 4) > 0:
            out["capex_restraint"] = -(_ttm(c, 0) / _ttm(c, 4) - 1)
    return out


def _best_tag_series(group: pd.DataFrame) -> pd.Series:
    order = (group.groupby(["tag", "rank"])
             .agg(latest=("end", "max"), n=("end", "nunique"))
             .reset_index()
             .sort_values(["latest", "n", "rank"], ascending=[False, False, True]))
    tag = order["tag"].iloc[0]
    g = group[group["tag"] == tag].sort_values("filed").groupby("end")["value"].last()
    return _regular_tail(g)


def fundamental_panel(facts: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Long frame (date, ticker, features...) using facts filed strictly before each date."""
    F = data.import_capex_fundamentals()
    facts = F.load_facts(facts)
    recs = []
    for asof in dates:
        vis = facts[(facts["filed"] < asof)
                    & (facts["end"] >= asof - pd.DateOffset(years=LOOKBACK_YEARS))]
        if vis.empty:
            continue
        q = F.derive_quarters(vis)
        for ticker, g in q.groupby("ticker", sort=False):
            series = {item: _best_tag_series(gi) for item, gi in g.groupby("item")}
            rev = series.get("revenue")
            if rev is None or rev.empty or (asof - rev.index[-1]).days > STALE_DAYS:
                continue
            ni = series.get("net_income")
            if ni is not None and len(ni) and (asof - ni.index[-1]).days > STALE_DAYS:
                ni = None
            recs.append({"date": asof, "ticker": ticker,
                         **_features_from_series(rev, ni, series.get("capex"))})
    return pd.DataFrame(recs)


def fundamental_signals(control: bool = False, refresh: bool = False) -> dict[str, pd.DataFrame]:
    path = data.CACHE / f"fundamental_panel{'_control' if control else ''}.parquet"
    if path.exists() and not refresh:
        panel = pd.read_parquet(path)
    else:
        dates = pd.date_range("2018-06-30", "2026-08-31", freq="ME")
        panel = fundamental_panel(data.facts(control), dates)
        panel.to_parquet(path)
    feats = [c for c in panel.columns if c not in ("date", "ticker")]
    return {f: panel.pivot_table(index="date", columns="ticker", values=f) for f in feats}


# -- price ---------------------------------------------------------------------

def price_signals(px_m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ret = px_m.pct_change(fill_method=None)
    return {
        "mom_12_1": px_m.shift(1) / px_m.shift(12) - 1,
        "reversal_1m": -ret,
    }


# -- power exposure --------------------------------------------------------------

def rolling_beta(ret: pd.DataFrame, mkt: pd.Series, factor: pd.Series,
                 window: int = BETA_WINDOW, min_obs: int = BETA_MIN_OBS) -> pd.DataFrame:
    """Coefficient on `factor` in r_i = a + b_m*mkt + b_f*factor, rolling window."""
    idx = ret.index
    X_all = pd.concat([mkt, factor], axis=1).reindex(idx)
    out = pd.DataFrame(np.nan, index=idx, columns=ret.columns)
    for end in range(min_obs, len(idx) + 1):
        lo = max(0, end - window)
        X = X_all.iloc[lo:end]
        Y = ret.iloc[lo:end]
        xok = X.notna().all(axis=1)
        for col in ret.columns:
            ok = xok & Y[col].notna()
            if ok.sum() < min_obs:
                continue
            A = np.column_stack([np.ones(ok.sum()), X[ok].to_numpy()])
            coef = np.linalg.lstsq(A, Y.loc[ok, col].to_numpy(), rcond=None)[0]
            out.iat[end - 1, out.columns.get_loc(col)] = coef[2]
    return out


def beta_x_shock(beta: pd.DataFrame, shock: pd.Series) -> pd.DataFrame:
    return beta.mul(shock.reindex(beta.index), axis=0)


def power_signals(px_m: pd.DataFrame, mkt_m: pd.Series, refresh: bool = False,
                  tag: str = "") -> dict[str, pd.DataFrame]:
    path = data.CACHE / f"power_betas{tag}.parquet"
    shocks = power.shocks(power.build())
    ret = px_m.pct_change(fill_method=None)
    mret = mkt_m.pct_change()
    if path.exists() and not refresh:
        betas = pd.read_parquet(path)
        betas = {f: betas.xs(f, axis=1, level=0) for f in power.FACTORS}
    else:
        betas = {f: rolling_beta(ret, mret, shocks[f]) for f in power.FACTORS}
        pd.concat(betas, axis=1).to_parquet(path)
    out = {}
    for f in power.FACTORS:
        trail = shocks[f].rolling(SHOCK_MONTHS).sum()
        out[f"beta_{f}"] = betas[f]
        out[f"bxs_{f}"] = beta_x_shock(betas[f], trail)
    return out
