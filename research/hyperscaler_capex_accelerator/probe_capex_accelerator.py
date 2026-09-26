"""
Big-4 hyperscaler capex-reacceleration as an ENTRY ACCELERATOR on the SHIPPED
electrification_strategy book (`screen +val+sleeve`) -- the opposite framing
from `grid_equipment_basket/capex_signal.py`, which uses the same underlying
data as a one-directional exit DE-RISK trigger and FAILED (fires ~a year
late; lagging confirmation -- docs/RESEARCH-LOG.md line 137).

Open item #16 in docs/RESEARCH-LOG.md §4: "the demand-data classes all failed
as *timing* because they lag a 2-4-yr-backlog business, but an *entry
accelerator* is more tolerant of lag (you're confirming a trend, not calling
a turn)." This is the first test of that idea.

Construction (full detail + rationale: docs/hyperscaler-capex-accelerator-
preregistration.md, written and committed BEFORE this script was run):

  - Reuses `grid_equipment_basket.capex_signal.fetch_bigfour_capex()`
    UNCHANGED -- same cached point-in-time quarterly `yoy` (trailing-4q
    capex growth) series, same `known_date` (quarter end + 50d) lag
    convention. No new capex computation.
  - Re-acceleration rule: yoy > yoy.shift(1).rolling(4).mean() (yoy exceeds
    its own trailing 4-quarter average, excluding the current quarter) --
    a plain moving-average crossover on the growth rate, mirroring the
    trend_gate price-vs-MA convention already used elsewhere in this
    codebase. No new window parameter (4q matches yoy's own window).
  - Multiplier: REGIME_HI (1.25, `grid_equipment_basket.config.REGIME_HI`,
    an existing constant, not tuned for this test) when reaccelerating,
    else 1.0. One-way UP only (mirror image of capex_signal's one-way DOWN
    de-risk multiplier) -- never below 1.0.
  - Combined with the existing overlay stack exactly as
    `research/steo_spark_spread/probe_steo_electrification_sizing.py` (the
    direct template, see docs/steo-spark-spread-results.md §10):
        exposure = clip(capex_accel_mult * valuation_mult * vol_scalar,
                         upper=VOL_MAX_LEVERAGE)
    applied to the CORE equity leg only, before the 15% GLD/IEF sleeve
    blend. Selection of the 27 names is untouched.

Pre-registered two-part gate (docs/hyperscaler-capex-accelerator-
preregistration.md §5): full-sample (2017-06-01 -> today) Sharpe of the
capex-accelerated cell beats the unmodified `screen +val+sleeve` cell AND
max drawdown does not get worse. Both required for PASS.

Throwaway probe. Findings -> docs/hyperscaler-capex-accelerator-results.md,
docs/RESEARCH-LOG.md (consolidated separately, not by this script/session).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from grid_equipment_basket import config as geb_config
from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket.basket import simulate_basket
from grid_equipment_basket.capex_signal import fetch_bigfour_capex
from grid_equipment_basket.data.prices import fetch_prices

from electrification_strategy import config, hedge_overlay, universe, valuation_overlay

HERE = pathlib.Path(__file__).parent
CON = "screen"

REGIME_HI = geb_config.REGIME_HI  # 1.25, reused unchanged -- not tuned for this test


def capex_accel_multiplier(index: pd.DatetimeIndex, capex: pd.DataFrame,
                           hi_mult: float = REGIME_HI) -> pd.Series:
    """Daily exposure multiplier in {1.0, hi_mult}, aligned to `index`.
    `hi_mult` applies once the most recently *known* big-four capex print's
    yoy growth exceeds its own trailing 4-quarter average (prior 4 quarters,
    excluding the current one); back to 1.0 otherwise. One-way: never below
    1.0 -- the mirror image of capex_signal.capex_derisk_multiplier, which is
    never above 1.0."""
    idx = pd.DatetimeIndex(index)
    trailing_avg = capex["yoy"].shift(1).rolling(4, min_periods=4).mean()
    accel = capex["yoy"] > trailing_avg
    known = pd.Series(np.where(accel.to_numpy(), hi_mult, 1.0),
                      index=capex["known_date"].to_numpy())
    known = known[~known.index.duplicated(keep="last")].sort_index()
    daily = known.reindex(idx, method="ffill")
    return daily.fillna(1.0).rename("capex_accel_mult")


def _vol_scalar(base_r: pd.Series) -> pd.Series:
    rv = base_r.rolling(config.VOL_LOOKBACK).std() * np.sqrt(config.ANN)
    return (config.VOL_TARGET / rv).clip(upper=config.VOL_MAX_LEVERAGE).shift(1)


def compose_with_capex_accel(base_r: pd.Series, spy_ret: pd.Series, capex_mult: pd.Series,
                             use_valuation: bool = True, val_scale: float = 1.0) -> pd.Series:
    """Same as electrification_strategy.backtest.compose(), plus a third
    multiplicative exposure factor from the capex re-acceleration multiplier."""
    if base_r.empty:
        return base_r
    vscal = _vol_scalar(base_r)
    if use_valuation:
        level = (1.0 + base_r).cumprod()
        vmult = valuation_overlay.extension_multiplier(
            level, spy_ret.reindex(base_r.index), scale=val_scale)
    else:
        vmult = pd.Series(1.0, index=base_r.index)
    cmult = capex_mult.reindex(base_r.index).fillna(1.0)
    exposure = (cmult * vmult.reindex(base_r.index).fillna(1.0)
                * vscal.reindex(base_r.index)).clip(upper=config.VOL_MAX_LEVERAGE).fillna(1.0)
    return exposure * base_r.fillna(0.0) + (1.0 - exposure) * config.RF / config.ANN


def main():
    capex = fetch_bigfour_capex()
    print(f"Capex data: {len(capex)} quarters, "
          f"{capex.index.min()} -> {capex.index.max()}, "
          f"known_date range {capex['known_date'].min().date()} -> {capex['known_date'].max().date()}")

    start, end = config.START_DEFAULT, pd.Timestamp.today().strftime("%Y-%m-%d")

    capex_mult = capex_accel_multiplier(
        pd.date_range(start, end, freq="D"), capex)
    print(f"capex_accel_mult: mean={capex_mult.mean():.3f}, "
          f"pct at {REGIME_HI}x={100 * (capex_mult == REGIME_HI).mean():.1f}%, "
          f"pct at 1.0x={100 * (capex_mult == 1.0).mean():.1f}%")

    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    tickers = sorted(set(cand) | set(config.BENCHMARKS) | set(config.HEDGE_SLEEVE_TICKERS)
                     | {config.FEARED_PROXY})
    print(f"\nFetching prices for {len(tickers)} tickers, {start} -> {end}...")
    prices = fetch_prices(tickers, start, end)
    cand_prices = prices[[c for c in cand if c in prices.columns]]

    spy_ret = prices["SPY"].pct_change()
    sleeve_r = hedge_overlay.sleeve_returns(prices)

    br = simulate_basket(cand_prices, start, end, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, universe.target_fn(CON, cand_prices))
    base_r = br.returns
    print(f"Base basket returns: {len(base_r)} days")

    # Baseline: exact same construction as the shipped `screen +val+sleeve` cell
    # (capex_mult forced to 1.0 everywhere -- neutral / no-op).
    core_r_base = compose_with_capex_accel(
        base_r, spy_ret, capex_mult=pd.Series(1.0, index=base_r.index), use_valuation=True)
    cell_r_base = (1.0 - config.HEDGE_SLEEVE_WEIGHT) * core_r_base.fillna(0.0) + \
        config.HEDGE_SLEEVE_WEIGHT * sleeve_r.reindex(core_r_base.index).fillna(0.0)

    # Capex-accelerator variant.
    core_r_accel = compose_with_capex_accel(
        base_r, spy_ret, capex_mult=capex_mult, use_valuation=True)
    cell_r_accel = (1.0 - config.HEDGE_SLEEVE_WEIGHT) * core_r_accel.fillna(0.0) + \
        config.HEDGE_SLEEVE_WEIGHT * sleeve_r.reindex(core_r_accel.index).fillna(0.0)

    m_base = compute_metrics(cell_r_base, config.RF, config.ANN)
    m_accel = compute_metrics(cell_r_accel, config.RF, config.ANN)

    print("\n=== screen +val+sleeve: shipped baseline vs. capex-accelerator-sized ===")
    print(f"{'metric':20s} {'baseline':>15s} {'capex-accel':>15s}")
    for key in ("sharpe", "cagr", "max_dd", "ann_vol"):
        b = m_base.get(key, float("nan"))
        s = m_accel.get(key, float("nan"))
        print(f"{key:20s} {b:>15.4f} {s:>15.4f}")

    gate_a = m_accel["sharpe"] > m_base["sharpe"]
    gate_b = m_accel["max_dd"] >= m_base["max_dd"]
    print(f"\nGate (a) Sharpe improves: {gate_a}")
    print(f"Gate (b) MaxDD does not worsen: {gate_b}")
    print(f"VERDICT: {'PASS' if (gate_a and gate_b) else 'FAIL'}")

    (HERE / "data").mkdir(exist_ok=True)
    cell_r_base.to_frame("ret").to_csv(HERE / "data" / "electrification_screen_valsleeve_baseline.csv")
    cell_r_accel.to_frame("ret").to_csv(HERE / "data" / "electrification_screen_valsleeve_capex_accel.csv")
    capex_mult.to_frame("capex_accel_mult").to_csv(HERE / "data" / "capex_accel_mult.csv")


if __name__ == "__main__":
    main()
