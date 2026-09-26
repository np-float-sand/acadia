"""
Structural-break-adaptive (exponentially-weighted) beta window probe
(untried-tasks-log item 3, fix #4 of 5).

Diagnosis: the flat, equal-weighted 252-day OLS window takes ~1 quarter to fully
absorb a narrative flip (every observation in the window counts equally, so a
regime that changed 20 days ago is still ~92% diluted by the other 232 days).
Fix #4 -- reweight observations within the same window by recency (exponential
decay, halflife = 63 trading days ~ 1 quarter) so a recent regime dominates the
estimate without shortening the window itself (which would just add noise).

Method: monkeypatch `conditional_beta.compute_stress_betas` with a WLS variant
(same stress-day filter, same window, only the observation weighting changes),
run the real, unmodified `grid_resilience.main.run()`.

Pre-registered gate: full-sample Sharpe and stress-period Sharpe should not get
worse than baseline (full -0.007, stress -0.145), and the merchant names' beta
series should show LESS lag around the four documented flip windows (2023-09,
2024-04/06, 2025-01/02, 2025-05/06) than the unweighted baseline -- i.e. the
sign/magnitude should move closer to "resolved" faster after a flip, not just
be a scaled copy of the same series (which is what fix #2's regressor turned out
to be).

Throwaway. Re-run to regenerate. Findings -> docs/RESEARCH-LOG.md,
docs/untried-tasks-log.md item 3.
"""
from __future__ import annotations

import functools
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import grid_resilience.signals.conditional_beta as cb
from grid_resilience.config import MIN_STRESS_OBS

try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

HALFLIFE_DAYS = 63  # ~1 trading quarter


def compute_stress_betas_ewls(
    returns: pd.DataFrame,
    gsi_by_iso: dict,
    ticker_iso_map: dict,
    sector_return: pd.Series,
    lookback_days: int = 252,
    stress_threshold: float = 0.5,
) -> pd.DataFrame:
    results = []
    decay = np.log(2) / HALFLIFE_DAYS
    for ticker in returns.columns:
        iso = ticker_iso_map.get(ticker)
        gsi_series = cb.gsi_for_ticker(ticker, gsi_by_iso, ticker_iso_map)
        if gsi_series.empty:
            continue
        excess_ret = (returns[ticker] - sector_return).dropna()
        common = excess_ret.index.intersection(gsi_series.index)
        if len(common) < MIN_STRESS_OBS:
            continue
        er, gsi = excess_ret.loc[common], gsi_series.loc[common]
        stress_mask = gsi >= stress_threshold
        er_stress, gsi_stress = er[stress_mask], gsi[stress_mask]
        if len(er_stress) < MIN_STRESS_OBS:
            continue
        gsi_stress = gsi_stress.clip(upper=gsi_stress.quantile(0.95))

        window_end = common.max()
        age_days = (window_end - gsi_stress.index).days.values.astype(float)
        weights = np.exp(-decay * age_days)
        weights = weights / weights.mean()  # normalize so scale matches unweighted OLS

        if _HAS_SM:
            X = sm.add_constant(gsi_stress.values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.WLS(er_stress.values, X, weights=weights).fit()
            stress_beta, alpha, r_sq = model.params[1], model.params[0], model.rsquared
        else:
            w_sqrt = np.sqrt(weights)
            stress_beta, alpha = np.polyfit(gsi_stress.values * w_sqrt, er_stress.values * w_sqrt, 1)
            r_sq = 0.0

        results.append({
            "ticker": ticker, "iso": iso, "stress_beta": stress_beta, "alpha": alpha,
            "r_squared": r_sq, "n_obs": len(er_stress),
            "mean_gsi_on_stress_days": gsi_stress.mean(),
        })
    return pd.DataFrame(results).set_index("ticker") if results else pd.DataFrame()


_CAPTURED: list[pd.DataFrame] = []


def _capture_wrapper(fn):
    @functools.wraps(fn)
    def wrapped(*a, **k):
        out = fn(*a, **k)
        _CAPTURED.append(out)
        return out
    return wrapped


def main():
    cb.compute_stress_betas = compute_stress_betas_ewls

    import grid_resilience.main as main_mod
    main_mod.rolling_stress_betas = _capture_wrapper(main_mod.rolling_stress_betas)

    print(f"Running full pipeline with exponentially-weighted beta (halflife={HALFLIFE_DAYS}d)...")
    result = main_mod.run(plot=False, save_dir="/tmp/gr_ewls_test")
    print("\nMETRICS:", result.get("metrics"))

    if _CAPTURED:
        rb = _CAPTURED[-1]
        rb.to_parquet("research/merchant_beta_orthogonalization/rolling_betas_ewls.parquet")
        print(f"Saved rolling betas: {rb.shape}")


if __name__ == "__main__":
    main()
