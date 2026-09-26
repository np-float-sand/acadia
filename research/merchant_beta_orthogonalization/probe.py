"""
Merchant stress-beta orthogonalization probe (untried-tasks-log item 3, fix #2).

Diagnosis being tested (docs/grid-stress-signal-evidence.md §5): VST/NRG's rolling
single-variable stress beta (excess_return ~ GSI, high-stress days only) whipsaws
because it can't distinguish "acute grid-stress response" from "the market's slower
structural narrative about whether high power prices are a tailwind or headwind for
this merchant generator." Fix #2 (of 5 candidates, none previously built): add the
ISO's own slow-moving power-price trend as a second regressor and take the GSI
coefficient, now orthogonalized to the structural component, as the stress beta.

Method: monkeypatch `conditional_beta.compute_stress_betas` (the function
`rolling_stress_betas` calls internally) with a two-variable OLS variant, then run
the REAL, unmodified `grid_resilience.main.run()` end to end so every other stage
(GSI construction, factor scoring, portfolio backtest, IC test) is identical to the
documented baseline -- only the beta estimation changes. No production files edited;
this stays a probe until/unless it clears a gate.

Pre-registered gate (both must hold to recommend adoption):
  (a) IC@21d t-stat improves vs the single-variable baseline run in the same session
      (baseline was run first, see docs/merchant-beta-orthogonalization-results.md).
  (b) full-sample Sharpe does not get worse, AND the merchant names' (VST, NRG, CEG,
      TLN) rolling stress-beta series shows measurably fewer / smaller sign flips
      across the four documented narrative-flip windows (late-2023, 2024-05,
      2025-01 DeepSeek, 2025-05).
Anything less -> FAIL, log and move to fix #1 or #4.

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
from grid_resilience.config import MIN_STRESS_OBS, ISO_BENCHMARK_NODES
from grid_resilience.data.grid_data import fetch_lmp, daily_lmp_summary

try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False


# ── Step 1: build the "structural power-price trend" regressor per ISO ────────

def build_power_trend(iso: str, start: str, end: str) -> pd.Series:
    """
    Slow-moving structural power-price trend: 63-day-smoothed daily mean LMP,
    trailing 252-trading-day % change, then rolling-756d z-scored (matches GSI's
    own scaling convention) and winsorized to +-3. Deliberately a much longer
    time constant than the GSI's daily z-score / stress mask, so it captures
    "is the multi-year power-price narrative up or down" rather than today's spike.
    """
    raw = fetch_lmp(iso, start, end)
    if raw.empty:
        return pd.Series(dtype=float)
    loc_col = "location" if "location" in raw.columns else ("zone" if "zone" in raw.columns else None)
    if loc_col:
        hub = raw[raw[loc_col].isin(ISO_BENCHMARK_NODES.get(iso, []))]
        if hub.empty:
            hub = raw
    else:
        hub = raw
    daily = daily_lmp_summary(hub, iso)
    if daily.empty or "lmp_mean" not in daily.columns:
        return pd.Series(dtype=float)
    smoothed = daily["lmp_mean"].rolling(63, min_periods=42).mean()
    trend_raw = smoothed.pct_change(252)
    roll_mean = trend_raw.rolling(756, min_periods=252).mean()
    roll_std = trend_raw.rolling(756, min_periods=252).std()
    z = ((trend_raw - roll_mean) / roll_std.replace(0, np.nan)).clip(-3, 3)
    return z.dropna()


# ── Step 2: orthogonalized two-variable stress-beta estimator ─────────────────

def compute_stress_betas_orthogonal(
    returns: pd.DataFrame,
    gsi_by_iso: dict,
    ticker_iso_map: dict,
    sector_return: pd.Series,
    lookback_days: int = 252,
    stress_threshold: float = 0.5,
    power_trend_by_iso: dict | None = None,
) -> pd.DataFrame:
    power_trend_by_iso = power_trend_by_iso or {}
    results = []
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

        trend_stress = None
        trend_series = power_trend_by_iso.get(iso)
        if trend_series is not None and not trend_series.empty:
            t = trend_series.reindex(gsi_stress.index)
            if t.notna().sum() >= max(10, int(0.5 * len(gsi_stress))):
                trend_stress = t.fillna(t.mean())

        trend_beta = np.nan
        if trend_stress is not None and _HAS_SM:
            X = sm.add_constant(np.column_stack([gsi_stress.values, trend_stress.values]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.OLS(er_stress.values, X).fit()
            alpha, stress_beta, trend_beta = model.params[0], model.params[1], model.params[2]
            r_sq = model.rsquared
        elif _HAS_SM:
            X = sm.add_constant(gsi_stress.values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.OLS(er_stress.values, X).fit()
            alpha, stress_beta, r_sq = model.params[0], model.params[1], model.rsquared
        else:
            stress_beta, alpha = np.polyfit(gsi_stress.values, er_stress.values, 1)
            r_sq = 0.0

        results.append({
            "ticker": ticker, "iso": iso, "stress_beta": stress_beta, "alpha": alpha,
            "r_squared": r_sq, "n_obs": len(er_stress),
            "mean_gsi_on_stress_days": gsi_stress.mean(), "trend_beta": trend_beta,
        })
    return pd.DataFrame(results).set_index("ticker") if results else pd.DataFrame()


_CAPTURED_ROLLING_BETAS: list[pd.DataFrame] = []


def _capture_wrapper(original_fn):
    @functools.wraps(original_fn)
    def wrapped(*args, **kwargs):
        out = original_fn(*args, **kwargs)
        _CAPTURED_ROLLING_BETAS.append(out)
        return out
    return wrapped


def main():
    from grid_resilience.config import BACKTEST_START, BACKTEST_END

    print("Building structural power-price trend for ERCOT + PJM...")
    trend_by_iso = {
        "ERCOT": build_power_trend("ERCOT", BACKTEST_START, BACKTEST_END),
        "PJM":   build_power_trend("PJM", BACKTEST_START, BACKTEST_END),
    }
    for iso, s in trend_by_iso.items():
        print(f"  {iso}: {len(s)} days, {s.index.min().date() if len(s) else None} -> "
              f"{s.index.max().date() if len(s) else None}")

    cb.compute_stress_betas = functools.partial(
        compute_stress_betas_orthogonal, power_trend_by_iso=trend_by_iso
    )
    # NB: `main.py` does `from conditional_beta import rolling_stress_betas`, binding its
    # own name in main's namespace -- patching cb.rolling_stress_betas would not reach it.
    # `rolling_stress_betas`'s *body* still resolves `compute_stress_betas` via
    # conditional_beta's globals at call time, so the beta patch above works either way;
    # patch main's own bound name so we can also capture the dated output for inspection.
    import grid_resilience.main as main_mod
    main_mod.rolling_stress_betas = _capture_wrapper(main_mod.rolling_stress_betas)

    print("\nRunning full pipeline with orthogonalized stress beta...")
    result = main_mod.run(plot=False, save_dir="/tmp/gr_orthogonal_test")
    print("\nMETRICS:", result.get("metrics"))

    if _CAPTURED_ROLLING_BETAS:
        rolling_betas = _CAPTURED_ROLLING_BETAS[-1]
        rolling_betas.to_parquet("research/merchant_beta_orthogonalization/rolling_betas_orthogonal.parquet")
        print(f"Saved rolling betas: {rolling_betas.shape}")


if __name__ == "__main__":
    main()
