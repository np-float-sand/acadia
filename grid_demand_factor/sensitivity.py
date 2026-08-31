"""
Cross-sectional grid-demand-sensitivity factor mechanics.

Pipeline
--------
1. monthly_total_returns(prices)      : daily adj-close -> month-end simple returns
2. rolling_sensitivity(rets, nowcast) : trailing-window OLS beta of each stock's
                                        monthly return on the monthly nowcast
3. quintile_spread(betas, fwd_rets)   : sort into quintiles each month, track the
                                        Q5-Q1 equal-weight long/short return
4. rank_ic(betas, fwd_rets)           : monthly Spearman corr(beta_t, ret_{t+1})
5. evaluate_gate(...)                 : apply the pre-registered pass/fail bar

All functions are pure (no I/O) so the unit tests can drive them with synthetic
data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_total_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily adjusted-close prices -> month-end simple returns (date x ticker)."""
    month_end = prices.resample("ME").last()
    return month_end.pct_change().dropna(how="all")


def rolling_sensitivity(
    monthly_returns: pd.DataFrame,
    nowcast: pd.Series,
    window: int = 36,
    min_periods: int = 24,
) -> pd.DataFrame:
    """
    Trailing-window OLS slope of each ticker's monthly return on the nowcast.

    For month t the beta uses returns and nowcast values in (t-window, t]. A
    ticker-month with fewer than ``min_periods`` overlapping observations is NaN.

    Returns a DataFrame (index = month-end, columns = ticker) of betas, aligned
    to the return index. These betas are the point-in-time factor score: the
    score known at month t is used to sort for the month t+1 return.
    """
    rets, now = monthly_returns.align(nowcast, join="inner", axis=0)
    now = now.astype(float)
    idx = rets.index
    betas = pd.DataFrame(index=idx, columns=rets.columns, dtype=float)

    x_full = now.values
    for i in range(len(idx)):
        lo = max(0, i - window + 1)
        xs = x_full[lo : i + 1]
        if len(xs) < min_periods:
            continue
        xc = xs - xs.mean()
        denom = float(xc @ xc)
        if denom == 0.0:
            continue
        block = rets.iloc[lo : i + 1]
        # column-wise slope with pairwise-complete obs
        for col in rets.columns:
            ys = block[col].values
            mask = ~np.isnan(ys)
            if mask.sum() < min_periods:
                continue
            xcm = xs[mask] - xs[mask].mean()
            d = float(xcm @ xcm)
            if d == 0.0:
                continue
            yc = ys[mask] - ys[mask].mean()
            betas.iat[i, betas.columns.get_loc(col)] = float(xcm @ yc) / d

    return betas


def _forward_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """ret_{t+1} aligned onto month t (so it pairs with a beta known at t)."""
    return monthly_returns.shift(-1)


def quintile_spread(
    betas: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    n_quantiles: int = 5,
    min_names: int = 25,
) -> dict:
    """
    Each month: rank tickers by beta, split into ``n_quantiles`` equal buckets,
    take the equal-weight next-month return of each bucket. Track the top-minus-
    bottom (Q5-Q1) long/short series.

    Returns a dict with:
      quantile_means : Series, mean monthly return per bucket (Q1..Qn)
      ls_returns     : Series, monthly Q5-Q1 return
      ls_sharpe_ann  : annualized Sharpe of ls_returns
      ls_t           : t-stat of mean(ls_returns)
      monotonic      : bool, quantile_means sorted (<=1 adjacent inversion)
      n_months       : number of months contributing
    """
    fwd = _forward_returns(monthly_returns)
    b, f = betas.align(fwd, join="inner")

    bucket_rows: list[list[float]] = []
    ls: dict[pd.Timestamp, float] = {}
    for dt in b.index:
        bt = b.loc[dt].dropna()
        common = bt.index.intersection(f.columns[f.loc[dt].notna()])
        bt = bt.loc[common]
        if len(bt) < min_names:
            continue
        ft = f.loc[dt, common]
        try:
            labels = pd.qcut(bt.rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue
        means = [ft[labels == q].mean() for q in range(n_quantiles)]
        bucket_rows.append(means)
        ls[dt] = means[-1] - means[0]

    if not bucket_rows:
        return {
            "quantile_means": pd.Series(dtype=float),
            "ls_returns": pd.Series(dtype=float),
            "ls_sharpe_ann": float("nan"),
            "ls_t": float("nan"),
            "monotonic": False,
            "n_months": 0,
        }

    qmeans = pd.Series(
        np.nanmean(np.array(bucket_rows), axis=0),
        index=[f"Q{i+1}" for i in range(n_quantiles)],
    )
    ls_s = pd.Series(ls).sort_index()
    mu, sd, n = ls_s.mean(), ls_s.std(ddof=1), len(ls_s)
    sharpe_ann = (mu / sd * np.sqrt(12)) if sd > 0 else float("nan")
    t_stat = (mu / sd * np.sqrt(n)) if sd > 0 else float("nan")

    diffs = np.diff(qmeans.values)
    inversions = int((diffs < 0).sum())
    monotonic_up = inversions <= 1
    inversions_dn = int((diffs > 0).sum())
    monotonic = monotonic_up or (inversions_dn <= 1)

    return {
        "quantile_means": qmeans,
        "ls_returns": ls_s,
        "ls_sharpe_ann": float(sharpe_ann),
        "ls_t": float(t_stat),
        "monotonic": bool(monotonic),
        "n_months": int(n),
    }


def rank_ic(betas: pd.DataFrame, monthly_returns: pd.DataFrame, min_names: int = 25) -> dict:
    """
    Monthly cross-sectional Spearman correlation between beta known at month t
    and the realized return in month t+1.

    Returns {"ic": Series (monthly IC), "mean_ic", "t_stat", "ir", "pct_pos",
             "n_months"}.
    """
    fwd = _forward_returns(monthly_returns)
    b, f = betas.align(fwd, join="inner")
    ics: dict[pd.Timestamp, float] = {}
    for dt in b.index:
        bt = b.loc[dt].dropna()
        ft = f.loc[dt].dropna()
        common = bt.index.intersection(ft.index)
        if len(common) < min_names:
            continue
        ics[dt] = bt.loc[common].corr(ft.loc[common], method="spearman")

    ic_s = pd.Series(ics).sort_index().dropna()
    if ic_s.empty:
        return {"ic": ic_s, "mean_ic": float("nan"), "t_stat": float("nan"),
                "ir": float("nan"), "pct_pos": float("nan"), "n_months": 0}

    mu, sd, n = ic_s.mean(), ic_s.std(ddof=1), len(ic_s)
    return {
        "ic": ic_s,
        "mean_ic": float(mu),
        "t_stat": float(mu / sd * np.sqrt(n)) if sd > 0 else float("nan"),
        "ir": float(mu / sd) if sd > 0 else float("nan"),
        "pct_pos": float((ic_s > 0).mean()),
        "n_months": int(n),
    }


# ── Pre-registered gate ──────────────────────────────────────────────────────

GATE_MIN_ABS_MEAN_IC = 0.03
GATE_MIN_ABS_IC_T = 2.0
GATE_MIN_LS_SHARPE = 0.35


def evaluate_gate(ic_result: dict, spread_result: dict) -> dict:
    """
    Apply the pre-registered Probe-D bar. D is 'worth a full gated build' only
    if BOTH conditions hold:

      1. |mean monthly rank-IC| >= 0.03 AND |t-stat| >= 2.0
      2. Q5-Q1 annualized Sharpe >= 0.35 AND quintile means monotone

    Returns {"passed": bool, "ic_ok": bool, "spread_ok": bool, "reasons": [...]}.
    """
    reasons: list[str] = []

    mean_ic = ic_result.get("mean_ic", float("nan"))
    ic_t = ic_result.get("t_stat", float("nan"))
    ic_ok = (
        np.isfinite(mean_ic)
        and np.isfinite(ic_t)
        and abs(mean_ic) >= GATE_MIN_ABS_MEAN_IC
        and abs(ic_t) >= GATE_MIN_ABS_IC_T
    )
    reasons.append(
        f"IC: mean={mean_ic:.4f} (need |.|>={GATE_MIN_ABS_MEAN_IC}), "
        f"t={ic_t:.2f} (need |.|>={GATE_MIN_ABS_IC_T}) -> {'OK' if ic_ok else 'FAIL'}"
    )

    sharpe = spread_result.get("ls_sharpe_ann", float("nan"))
    mono = spread_result.get("monotonic", False)
    spread_ok = np.isfinite(sharpe) and abs(sharpe) >= GATE_MIN_LS_SHARPE and mono
    reasons.append(
        f"Spread: Q5-Q1 ann Sharpe={sharpe:.2f} (need |.|>={GATE_MIN_LS_SHARPE}), "
        f"monotone={mono} -> {'OK' if spread_ok else 'FAIL'}"
    )

    passed = bool(ic_ok and spread_ok)
    reasons.append(f"GATE {'PASSED' if passed else 'FAILED'}")
    return {"passed": passed, "ic_ok": bool(ic_ok), "spread_ok": bool(spread_ok), "reasons": reasons}
