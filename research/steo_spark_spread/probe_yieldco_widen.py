"""
Does grid-equipment momentum help forecast FRVO/BEP/ENLT/CWEN/ORA/BEPC returns?
User request, following up on Thread 1/1b/1c in docs/interconnection-queue-signal-
results.md (grid-equipment 3-month momentum -> yieldco/REIT forward returns).

DATA CONSTRAINT FOUND FIRST (before running anything): FRVO IPO'd ~2026-05 (5 months
of price history), ENLT lists from 2023-01 (45 months), BEPC from 2020-07 (75 months)
-- none has enough history for Thread 1b/1c's 2017-09->2022-12 fit window. Only BEP,
CWEN, ORA (all 15+ years) support the full pre-registered methodology. FRVO/ENLT/BEPC
are reported as untestable under this methodology, not silently dropped.

RECONSTRUCTION CAVEAT: Thread 1/1b/1c's original scripts were scratchpad throwaways,
never committed to research/ (same gap found for Thread 6). This rebuilds the
"grid-equipment 3-month momentum" regressor from the documented description --
trailing 3-month return of the project's own grid-equipment supplier basket
(`screen` construction, the same basket used everywhere else in this project) --
and SANITY-CHECKS it against Thread 1b's reported numbers on the ORIGINAL basket
(CWEN/BEP/AES/AQN) before trusting it on the new tickers. Not guaranteed to be a
byte-exact reproduction; flagged, not hidden.

Two tests, per user request ("both"):
  (a) Thread 1c-style: per-name fit-period regression (ret_i ~ grid_mom + spy_ret,
      fit 2017-09->2022-12, n~64), |t|>=2.0 gate -- run individually on BEP/CWEN/ORA
      (the 3 testable names), no cherry-picking.
  (b) Thread 1/1b-style: pooled-basket OOS test -- fit on the original CWEN/BEP/AES/AQN
      basket (sanity check vs. reported +0.429 OOS rank-corr), then on a widened
      basket (BEP/CWEN/ORA -- the testable subset of the user's requested names,
      AES/AQN dropped to isolate the effect of the new names) over the SAME fit/
      holdout split.

Throwaway probe. Findings -> docs/interconnection-queue-signal-results.md,
docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, ".")

from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.basket import simulate_basket
from electrification_strategy import config, universe

FIT_START, FIT_END = "2017-09-01", "2022-12-31"
HOLDOUT_START, HOLDOUT_END = "2023-01-01", "2026-08-31"

ORIGINAL_YIELDCO = ["CWEN", "BEP", "AES", "AQN"]
TESTABLE_NEW = ["BEP", "CWEN", "ORA"]          # enough history for the fit window
UNTESTABLE_NEW = ["FRVO", "ENLT", "BEPC"]      # too little history, reported not run


def monthly_ret(price: pd.Series) -> pd.Series:
    m = price.resample("ME").last()
    return m.pct_change().dropna()


def grid_mom_series(cand_prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Trailing 3-month momentum of the project's own grid-equipment `screen` basket,
    sampled at each month-end (no lookahead: month-end t's momentum uses only prices
    through t)."""
    br = simulate_basket(cand_prices, start, end, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, universe.target_fn("screen", cand_prices))
    level = (1.0 + br.returns.fillna(0.0)).cumprod()
    month_end_level = level.resample("ME").last()
    mom = month_end_level.pct_change(3)
    return mom.dropna().rename("grid_mom")


def fit_ols(y: pd.Series, grid_mom: pd.Series, spy_ret: pd.Series, start, end):
    idx = y.index.intersection(grid_mom.index).intersection(spy_ret.index)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    d = pd.DataFrame({"y": y.reindex(idx), "grid_mom": grid_mom.reindex(idx),
                      "spy_ret": spy_ret.reindex(idx)}).dropna()
    if len(d) < 8:
        return None, len(d)
    X = sm.add_constant(d[["grid_mom", "spy_ret"]])
    return sm.OLS(d["y"], X).fit(), len(d)


def oos_check(model, grid_mom: pd.Series, spy_ret: pd.Series, y: pd.Series, start, end):
    idx = grid_mom.index.intersection(spy_ret.index).intersection(y.index)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    d = pd.DataFrame({"grid_mom": grid_mom.reindex(idx), "spy_ret": spy_ret.reindex(idx),
                      "y": y.reindex(idx)}).dropna()
    if len(d) < 6 or model is None:
        return None, len(d)
    X = sm.add_constant(d[["grid_mom", "spy_ret"]])
    pred = model.predict(X)
    rank_corr, _ = stats.spearmanr(pred, d["y"])
    return rank_corr, len(d)


def main():
    all_new_testable = sorted(set(TESTABLE_NEW))
    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    tickers = sorted(set(cand) | {"SPY"} | set(ORIGINAL_YIELDCO) | set(all_new_testable))
    print(f"Fetching prices for {len(tickers)} tickers, {FIT_START} -> {HOLDOUT_END}...")
    prices = fetch_prices(tickers, FIT_START, HOLDOUT_END)
    cand_prices = prices[[c for c in cand if c in prices.columns]]

    print(f"\nData depth check (already known, restated for the record):")
    for t in TESTABLE_NEW + UNTESTABLE_NEW:
        if t in prices.columns:
            s = prices[t].dropna()
            print(f"  {t}: {s.index.min().date() if len(s) else 'NO DATA'} -> "
                  f"{s.index.max().date() if len(s) else ''} ({len(s)} days)")
        else:
            print(f"  {t}: NOT FETCHED / NO DATA")
    print(f"  FRVO/ENLT/BEPC: insufficient history for the {FIT_START}->{FIT_END} fit "
          f"window (need 2017-09 onward) -- NOT run through either test.")

    grid_mom = grid_mom_series(cand_prices, FIT_START, HOLDOUT_END)
    spy_ret = monthly_ret(prices["SPY"])
    print(f"\ngrid_mom: {len(grid_mom)} months, {grid_mom.index.min().date()} -> {grid_mom.index.max().date()}")

    # ── Sanity check: reproduce Thread 1b's reported yieldco OOS rank-corr (+0.429) ──
    yc_orig = prices[[c for c in ORIGINAL_YIELDCO if c in prices.columns]].pct_change().mean(axis=1)
    yc_orig_m = (1 + yc_orig.fillna(0.0)).resample("ME").prod() - 1
    m_orig, n_fit_orig = fit_ols(yc_orig_m, grid_mom, spy_ret, FIT_START, FIT_END)
    if m_orig is not None:
        rc_orig, n_oos_orig = oos_check(m_orig, grid_mom, spy_ret, yc_orig_m, HOLDOUT_START, HOLDOUT_END)
        print(f"\nSanity check -- original yieldco basket (CWEN/BEP/AES/AQN):")
        print(f"  fit n={n_fit_orig}, OOS rank-corr={rc_orig:.3f} (n={n_oos_orig})  "
              f"[Thread 1b reported: +0.429]")

    # ── (a) Thread 1c-style: per-name fit-period regression, BEP/CWEN/ORA ──
    print(f"\n=== (a) Per-name fit-period regression (fit {FIT_START}->{FIT_END}), |t|>=2.0 gate ===")
    passers = []
    for t in TESTABLE_NEW:
        if t not in prices.columns:
            print(f"  {t}: no price data, skip")
            continue
        y = monthly_ret(prices[t])
        m, n = fit_ols(y, grid_mom, spy_ret, FIT_START, FIT_END)
        if m is None:
            print(f"  {t}: n={n}, too few obs")
            continue
        t_mom = m.tvalues["grid_mom"]
        passed = abs(t_mom) >= 2.0
        print(f"  {t}: n={n}, grid_mom t={t_mom:.3f}, spy_ret t={m.tvalues['spy_ret']:.3f}  "
              f"{'PASS' if passed else 'fail'}")
        if passed:
            passers.append((t, m))

    print(f"\n  {len(passers)}/{len(TESTABLE_NEW)} names pass the fit-period gate: "
          f"{[t for t, _ in passers]}")

    for t, m in passers:
        y = monthly_ret(prices[t])
        rc, n_oos = oos_check(m, grid_mom, spy_ret, y, HOLDOUT_START, HOLDOUT_END)
        print(f"  {t} OOS rank-corr (holdout {HOLDOUT_START}->{HOLDOUT_END}, n={n_oos}): {rc:.3f}")

    # ── (b) Thread 1/1b-style: widened pooled basket (testable new names only) ──
    print(f"\n=== (b) Pooled widened basket: BEP/CWEN/ORA (AES/AQN dropped) ===")
    widened = prices[[c for c in TESTABLE_NEW if c in prices.columns]].pct_change().mean(axis=1)
    widened_m = (1 + widened.fillna(0.0)).resample("ME").prod() - 1
    m_wide, n_fit_wide = fit_ols(widened_m, grid_mom, spy_ret, FIT_START, FIT_END)
    if m_wide is not None:
        print(f"  fit n={n_fit_wide}, grid_mom t={m_wide.tvalues['grid_mom']:.3f}, "
              f"spy_ret t={m_wide.tvalues['spy_ret']:.3f}")
        rc_wide, n_oos_wide = oos_check(m_wide, grid_mom, spy_ret, widened_m, HOLDOUT_START, HOLDOUT_END)
        print(f"  OOS rank-corr (holdout, n={n_oos_wide}): {rc_wide:.3f}")


if __name__ == "__main__":
    main()
