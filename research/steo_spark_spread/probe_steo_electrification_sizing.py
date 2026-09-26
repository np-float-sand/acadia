"""
STEO forward-spark-spread as an exposure regime multiplier on the SHIPPED
electrification_strategy book (`screen +val+sleeve`), NOT the grid_resilience
merchant-beta book the rest of this session's STEO probes used.

Context: `probe_steo_sizing.py` tested STEO-as-sizing on the wrong book (the
legacy grid_resilience merchant stress-beta factor, Sharpe -0.007 baseline,
unrelated to anything shipped) and FAILED. This probe redirects the same
underlying idea -- "use STEO to raise/lower exposure, not to change which
names are held" -- onto the actual shipped 27-name equipment-supplier book,
using its ALREADY-EXISTING vol-target overlay (`backtest._vol_scalar`,
20%-target / 21d-trailing-vol / 1.5x cap) as the base to multiply into.

Construction: reuses `grid_equipment_basket.grid_regime.regime_multiplier`
UNCHANGED (same function, same default discrete thresholds/hi/lo/month-lookback
already calibrated for the grid-congestion Layer-2 overlay that failed its own
gate on this same book) -- no new parameter search, just a new input series.
Composite = mean of the PJM/ERCOT STEO spark-spread z-trend (already built,
`research/steo_spark_spread/data/steo_trend_{PJM,ERCOT}.parquet`).

exposure = clip(steo_mult * valuation_mult * vol_scalar, upper=VOL_MAX_LEVERAGE)

applied to the CORE equity leg only (mirrors how the existing vol_scalar and
valuation_mult already only touch the core, before the 15% GLD/IEF sleeve is
blended in at its fixed weight) -- selection of the 27 names is untouched.

Mechanism caveat (stated before running, not after): STEO's spark spread was
built to predict PJM/ERCOT MERCHANT GENERATOR profitability (VST/NRG/CEG/TLN).
This book is equipment SUPPLIERS (transformers/switchgear/etc) -- a more
indirect link. The two prior forward-power-market regime overlays tried on
this exact book (grid-congestion, FTR-bid-implied congestion) both failed
their gate already (docs/grid-regime-layer2-results.md, docs/ftr-bid-signal-
results.md), so the prior here is low.

Pre-registered gate (reuses the L2 regime ladder's own G1/G3 spirit, but
against the single shipped `screen +val+sleeve` cell rather than a full
ladder): full-sample (2017-06-01 -> today) Sharpe improves vs. the unmodified
shipped cell AND max drawdown does not get worse.

Throwaway probe. Findings -> docs/steo-spark-spread-results.md, docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket.basket import simulate_basket
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.grid_regime import regime_multiplier

from electrification_strategy import config, hedge_overlay, universe, valuation_overlay
from electrification_strategy.fred import fetch_series

HERE = pathlib.Path(__file__).parent
CON = "screen"


def load_steo_composite() -> pd.Series:
    pjm = pd.read_parquet(HERE / "data" / "steo_trend_PJM.parquet")["steo_spark_trend_z"]
    ercot = pd.read_parquet(HERE / "data" / "steo_trend_ERCOT.parquet")["steo_spark_trend_z"]
    df = pd.concat([pjm.rename("pjm"), ercot.rename("ercot")], axis=1)
    return df.mean(axis=1, skipna=True).dropna().rename("steo_composite")


def _vol_scalar(base_r: pd.Series) -> pd.Series:
    rv = base_r.rolling(config.VOL_LOOKBACK).std() * np.sqrt(config.ANN)
    return (config.VOL_TARGET / rv).clip(upper=config.VOL_MAX_LEVERAGE).shift(1)


def compose_with_steo(base_r: pd.Series, spy_ret: pd.Series, steo_mult: pd.Series,
                      use_valuation: bool = True, val_scale: float = 1.0) -> pd.Series:
    """Same as electrification_strategy.backtest.compose(), plus a third
    multiplicative exposure factor from the STEO regime multiplier."""
    if base_r.empty:
        return base_r
    vscal = _vol_scalar(base_r)
    if use_valuation:
        level = (1.0 + base_r).cumprod()
        vmult = valuation_overlay.extension_multiplier(
            level, spy_ret.reindex(base_r.index), scale=val_scale)
    else:
        vmult = pd.Series(1.0, index=base_r.index)
    smult = steo_mult.reindex(base_r.index).fillna(1.0)
    exposure = (smult * vmult.reindex(base_r.index).fillna(1.0)
                * vscal.reindex(base_r.index)).clip(upper=config.VOL_MAX_LEVERAGE).fillna(1.0)
    return exposure * base_r.fillna(0.0) + (1.0 - exposure) * config.RF / config.ANN


def main():
    steo_composite = load_steo_composite()
    print(f"STEO composite: {len(steo_composite)} days, "
          f"{steo_composite.index.min().date()} -> {steo_composite.index.max().date()}")

    steo_mult = regime_multiplier(steo_composite, mode="discrete")
    print(f"steo_mult: mean={steo_mult.mean():.3f}, "
          f"pct HI ({steo_mult.max():.2f})={100 * (steo_mult == steo_mult.max()).mean():.1f}%, "
          f"pct LO ({steo_mult.min():.2f})={100 * (steo_mult == steo_mult.min()).mean():.1f}%, "
          f"pct neutral (1.0)={100 * (steo_mult == 1.0).mean():.1f}%")

    start, end = config.START_DEFAULT, pd.Timestamp.today().strftime("%Y-%m-%d")

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

    # Baseline: exact same construction as the shipped `screen +val+sleeve` cell.
    core_r_base = compose_with_steo(base_r, spy_ret, steo_mult=pd.Series(1.0, index=base_r.index),
                                    use_valuation=True)
    cell_r_base = (1.0 - config.HEDGE_SLEEVE_WEIGHT) * core_r_base.fillna(0.0) + \
        config.HEDGE_SLEEVE_WEIGHT * sleeve_r.reindex(core_r_base.index).fillna(0.0)

    # STEO variant: same, with the regime multiplier layered into the core's exposure.
    core_r_steo = compose_with_steo(base_r, spy_ret, steo_mult=steo_mult, use_valuation=True)
    cell_r_steo = (1.0 - config.HEDGE_SLEEVE_WEIGHT) * core_r_steo.fillna(0.0) + \
        config.HEDGE_SLEEVE_WEIGHT * sleeve_r.reindex(core_r_steo.index).fillna(0.0)

    m_base = compute_metrics(cell_r_base, config.RF, config.ANN)
    m_steo = compute_metrics(cell_r_steo, config.RF, config.ANN)

    print("\n=== screen +val+sleeve: shipped baseline vs. STEO-regime-sized ===")
    print(f"{'metric':20s} {'baseline':>15s} {'STEO-sized':>15s}")
    for key in ("sharpe", "cagr", "max_dd", "ann_vol"):
        b = m_base.get(key, float("nan"))
        s = m_steo.get(key, float("nan"))
        print(f"{key:20s} {b:>15.4f} {s:>15.4f}")

    gate_a = m_steo["sharpe"] > m_base["sharpe"]
    gate_b = m_steo["max_dd"] >= m_base["max_dd"]
    print(f"\nGate (a) Sharpe improves: {gate_a}")
    print(f"Gate (b) MaxDD does not worsen: {gate_b}")
    print(f"VERDICT: {'PASS' if (gate_a and gate_b) else 'FAIL'}")

    cell_r_base.to_frame("ret").to_csv(HERE / "data" / "electrification_screen_valsleeve_baseline.csv")
    cell_r_steo.to_frame("ret").to_csv(HERE / "data" / "electrification_screen_valsleeve_steo_sized.csv")
    steo_mult.to_frame("steo_mult").to_csv(HERE / "data" / "steo_regime_mult.csv")


if __name__ == "__main__":
    main()
