"""
Does the STEO forward spark-spread trend gate the ADOPTED screened-6 REIT short sleeve
(SBAC/WPC/EQIX/CCI/AMT/DLR, docs/interconnection-queue-signal-results.md Thread 1c) as
well as or better than the grid-equipment-momentum gate it was validated with?

Same rigor as that thread's Thread 1b OOS check: fit-period-only regression, then forecast
a genuinely later holdout using realized inputs, check OOS rank-correlation BEFORE trusting
any combined-book Sharpe number. STEO trend data only exists from 2021-01-19 onward (vs.
that thread's 2017-2022 fit period) -- fit window is necessarily much shorter here, flagged
explicitly, not smoothed over.

Throwaway. Findings -> reported inline to user, not a persistent doc unless asked.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from scipy import stats

from grid_equipment_basket.data.prices import fetch_prices
from electrification_strategy import backtest, config, universe, hedge_overlay
from electrification_strategy.fred import fetch_series
from electrification_strategy.backtest import compose, compute_metrics

SCREENED6 = ["SBAC", "WPC", "EQIX", "CCI", "AMT", "DLR"]
FIT_START, FIT_END = "2021-02-01", "2022-12-31"
HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2025-12-31"

def load_steo_trend():
    ercot = pd.read_parquet("research/steo_spark_spread/data/steo_trend_ERCOT.parquet")["steo_spark_trend_z"]
    pjm = pd.read_parquet("research/steo_spark_spread/data/steo_trend_PJM.parquet")["steo_spark_trend_z"]
    idx = ercot.index.union(pjm.index)
    avg = (ercot.reindex(idx).ffill() + pjm.reindex(idx).ffill()) / 2.0
    return avg.rename("steo_avg")

def monthly_ret(price_series):
    m = price_series.resample("ME").last()
    return m.pct_change().dropna()

def main():
    end = "2026-08-31"
    tickers = sorted(set(SCREENED6) | {"SPY"})
    prices = fetch_prices(tickers, "2020-06-01", end)

    basket_daily = prices[SCREENED6].pct_change().mean(axis=1)
    basket_m = (1 + basket_daily).resample("ME").prod() - 1
    spy_m = monthly_ret(prices["SPY"])

    steo_daily = load_steo_trend()
    steo_m = steo_daily.resample("ME").last()

    df = pd.concat([basket_m.rename("basket"), spy_m.rename("spy"), steo_m], axis=1).dropna()
    print(f"Monthly panel: {len(df)} obs, {df.index.min().date()} -> {df.index.max().date()}")

    fit = df.loc[FIT_START:FIT_END]
    hold = df.loc[HOLDOUT_START:HOLDOUT_END]
    print(f"Fit: n={len(fit)} [{FIT_START}->{FIT_END}]   Holdout: n={len(hold)} [{HOLDOUT_START}->{HOLDOUT_END}]")

    X = np.column_stack([np.ones(len(fit)), fit["steo_avg"].values, fit["spy"].values])
    y = fit["basket"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(fit) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstats = beta / se
    print(f"Fit-period coefficients (const, steo_avg, spy): {beta}")
    print(f"Fit-period t-stats:                              {tstats}  (n={len(fit)}, dof={dof})")

    Xh = np.column_stack([np.ones(len(hold)), hold["steo_avg"].values, hold["spy"].values])
    pred = Xh @ beta
    actual = hold["basket"].values
    rank_corr, rank_p = stats.spearmanr(pred, actual)
    lin_corr = np.corrcoef(pred, actual)[0, 1]
    print(f"\nOOS holdout: rank-corr={rank_corr:.3f} (p={rank_p:.3f})  linear-corr={lin_corr:.3f}")
    print(f"Cumulative predicted: {(np.prod(1+pred)-1)*100:.1f}%   Cumulative actual: {(np.prod(1+actual)-1)*100:.1f}%")
    print(f"Sign of steo_avg coefficient: {'positive' if beta[1] > 0 else 'negative'} (t={tstats[1]:.2f})")

if __name__ == "__main__":
    main()
