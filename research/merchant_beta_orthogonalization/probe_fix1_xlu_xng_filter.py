"""
XLU/XNG correlation filter probe (untried-tasks-log item 3, fix #1 of 5).

Diagnosis (docs/grid-stress-signal-evidence.md Sec5): VST/NRG drift between trading like a
regulated utility and trading like a merchant gas/power name, and the single-variable
stress beta can't tell which regime it's in. Fix #1 -- instead of trying to fix the beta
estimate, just EXCLUDE a name from that month's cross-section when it's demonstrably
trading like gas, not like utilities: corr(stock, ^XNG) > corr(stock, XLU) for 2
consecutive rebalance months.

Method: monkeypatch `grid_resilience.main.build_rolling_weights` to drop (date, ticker)
rows from `rolling_factors` before ranking, wherever the filter fires. Every other stage
(GSI, stress beta, factor score itself) is untouched -- this is purely a
universe-eligibility filter applied right before portfolio construction.

Pre-registered gate (same bar as fix #2, for a fair comparison):
  (a) IC@21d / IC@63d t-stat improves vs the single-variable baseline (t=1.352 / t=1.889).
  (b) full-sample Sharpe does not get worse.
  (c) sanity check: the filter should fire predominantly on VST/NRG/CEG/TLN (merchant
      names), rarely on regulated names -- if it fires broadly/randomly, the XNG proxy
      isn't doing what it's supposed to and the result isn't interpretable either way.
Anything less -> FAIL.

Throwaway. Re-run to regenerate. Findings -> docs/RESEARCH-LOG.md,
docs/untried-tasks-log.md item 3.
"""
from __future__ import annotations

import functools
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, ".")

import grid_resilience.main as main_mod
from grid_resilience.data.universe import UNIVERSE

CORR_WINDOW = 63  # trading days (~1 quarter) -- smooths noise, still regime-responsive


def build_exclusion_mask(rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by rebalance date with a boolean column per ticker:
    True = exclude this ticker at this rebalance (2-consecutive-month XNG>XLU rule).
    """
    tickers = list(UNIVERSE)
    px = yf.download(tickers + ["XLU", "^XNG"], start="2016-06-01", end="2026-08-01",
                      auto_adjust=True, progress=False)["Close"]
    ret = px.pct_change()

    corr_xlu = pd.DataFrame({t: ret[t].rolling(CORR_WINDOW).corr(ret["XLU"]) for t in tickers})
    corr_xng = pd.DataFrame({t: ret[t].rolling(CORR_WINDOW).corr(ret["^XNG"]) for t in tickers})

    gas_regime = (corr_xng > corr_xlu)  # daily boolean, per ticker

    # Sample at each rebalance date (nearest prior trading day)
    monthly_flag = pd.DataFrame(index=rebalance_dates, columns=tickers, dtype=bool)
    for d in rebalance_dates:
        avail = gas_regime.loc[:d]
        monthly_flag.loc[d] = avail.iloc[-1] if not avail.empty else False

    # 2-consecutive-month rule
    exclude = monthly_flag & monthly_flag.shift(1).fillna(False)
    return exclude


_last_exclude_mask: pd.DataFrame | None = None


def make_filtered_build_rolling_weights(original_fn, exclude_mask: pd.DataFrame):
    @functools.wraps(original_fn)
    def wrapped(rolling_factors, *args, **kwargs):
        if rolling_factors.empty:
            return original_fn(rolling_factors, *args, **kwargs)
        df = rolling_factors.copy()
        df["date"] = pd.to_datetime(df["date"])

        def _is_excluded(row):
            d, t = row["date"], row["ticker"]
            if t not in exclude_mask.columns:
                return False
            avail = exclude_mask.loc[:d, t] if d in exclude_mask.index or True else None
            sub = exclude_mask[exclude_mask.index <= d]
            if sub.empty or t not in sub.columns:
                return False
            return bool(sub[t].iloc[-1])

        mask = df.apply(_is_excluded, axis=1)
        n_excluded = int(mask.sum())
        print(f"  [xlu-xng filter] excluding {n_excluded} / {len(df)} (date, ticker) rows")
        if n_excluded:
            by_ticker = df.loc[mask, "ticker"].value_counts()
            print(f"  [xlu-xng filter] by ticker:\n{by_ticker.to_string()}")
        filtered = df.loc[~mask].reset_index(drop=True)
        return original_fn(filtered, *args, **kwargs)
    return wrapped


def main():
    from grid_resilience.config import BACKTEST_START, BACKTEST_END, REBALANCE_FREQ

    rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq=REBALANCE_FREQ)

    print("Building XLU/XNG exclusion mask...")
    exclude_mask = build_exclusion_mask(rebalance_dates)
    global _last_exclude_mask
    _last_exclude_mask = exclude_mask
    exclude_mask.to_parquet("research/merchant_beta_orthogonalization/xlu_xng_exclude_mask.parquet")

    total_excludable = exclude_mask.sum().sum()
    print(f"Total ticker-months flagged for exclusion: {total_excludable}")
    print("By ticker (top 10):")
    print(exclude_mask.sum().sort_values(ascending=False).head(10).to_string())

    main_mod.build_rolling_weights = make_filtered_build_rolling_weights(
        main_mod.build_rolling_weights, exclude_mask
    )

    print("\nRunning full pipeline with the XLU/XNG exclusion filter...")
    result = main_mod.run(plot=False, save_dir="/tmp/gr_xlu_xng_test")
    print("\nMETRICS:", result.get("metrics"))


if __name__ == "__main__":
    main()
