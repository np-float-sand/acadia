"""
COMBINATION TEST — CEP as a second gate/scaler on the shipped DLR/EQIX short sleeve.

Pre-registered contingency (docs/datacenter-reit-cep-preregistration.md Decision 8) said this
only runs if the standalone CEP signal clears |t|>=2.0 in >=2 windows. It didn't
(docs/datacenter-reit-cep-signal-results.md: best t=-1.76). Run anyway per explicit user
instruction (t~1.75-1.76 accepted as "good enough" to test the downstream question directly,
and the user's own experience is that marginal signals often add value in combination) --
**this is a deviation from the pre-registration, done on direct instruction, not a change to the
project's own |t|>=2.0 gate convention used elsewhere.**

Question: does gating/scaling the existing real-yield-gated DLR+EQIX short sleeve on CEP change
the *combined* screen+val+sleeve+short book's Sharpe/Calmar/MaxDD/feared-scenario P&L, vs. the
real-yield-only gate already shipped? Tests (a) CEP alone, (b) CEP AND real-yield (both fire),
(c) CEP as a continuous 0-1x scaler -- each against the (d) real-yield-only baseline and (e) no
short at all.

CEP publication-lag caveat: EIA retail-sales monthly data is not available same-month; no
point-in-time archive exists in this repo (same limitation flagged for other EIA-derived
signals in this project's log), so a fixed 65-day publish lag is assumed (approximates EIA's
actual ~2-month release cadence) before the monthly CEP print can affect the mask -- documented
assumption, not fitted.

Throwaway. Findings -> docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from grid_equipment_basket.backtest import compute_metrics          # noqa: E402
from grid_equipment_basket.basket import simulate_basket            # noqa: E402
from grid_equipment_basket.data.prices import fetch_prices          # noqa: E402

from electrification_strategy import config, hedge_overlay, universe  # noqa: E402
from electrification_strategy._util import month_hold                # noqa: E402
from electrification_strategy.backtest import (                      # noqa: E402
    compose, episode_drawdowns, feared_pnl, calm_drag,
)
from electrification_strategy.fred import fetch_series               # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from probe import eia_national_cep                                  # noqa: E402

START, END = config.START_DEFAULT, "2026-08-31"
CONSTRUCTION = "screen"
CEP_PUBLISH_LAG_DAYS = 65


# ----------------------------- CEP gate construction -----------------------------
def cep_rising_mask(cep_monthly: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily boolean mask: True when 6-month CEP change is positive, publish-lagged and
    held one month -- structurally identical to hedge_overlay.real_yield_rising_mask, just
    swapping DFII10 (daily, 5-day lag) for CEP (monthly, 65-day publish lag)."""
    s = cep_monthly.astype(float).sort_index()
    publish_date = s.index.to_timestamp(how="end") + pd.Timedelta(days=CEP_PUBLISH_LAG_DAYS)
    daily = pd.Series(s.to_numpy(), index=publish_date).sort_index()
    daily = daily.reindex(index.union(daily.index)).ffill().reindex(index)
    rising = (daily - daily.shift(config.HEDGE_SHORT_LOOKBACK_DAYS)) > 0
    return month_hold(rising.fillna(False).astype(float), index, fill=0.0) > 0.5


def cep_scaler(cep_monthly: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Continuous 0-1x: 6-month CEP change, min-max normalized over its own trailing
    3-year history, clipped to [0,1]. Same publish lag as the binary gate."""
    s = cep_monthly.astype(float).sort_index()
    chg6m = s - s.shift(6)
    publish_date = chg6m.index.to_timestamp(how="end") + pd.Timedelta(days=CEP_PUBLISH_LAG_DAYS)
    daily = pd.Series(chg6m.to_numpy(), index=publish_date).sort_index()
    daily = daily.reindex(index.union(daily.index)).ffill().reindex(index)
    lo = daily.rolling(756, min_periods=252).quantile(0.10)
    hi = daily.rolling(756, min_periods=252).quantile(0.90)
    scaled = ((daily - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)
    return month_hold(scaled, index, fill=0.0)


# ----------------------------- book construction -----------------------------
def main():
    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    tickers = sorted(set(cand) | set(config.BENCHMARKS) | set(config.HEDGE_SLEEVE_TICKERS)
                     | set(config.HEDGE_SHORT_TICKERS) | {config.FEARED_PROXY})
    prices = fetch_prices(tickers, START, END)
    dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", END)
    cep = eia_national_cep()

    cand_prices = prices[[c for c in cand if c in prices.columns]]
    spy_ret = prices["SPY"].pct_change()
    tan_ret = prices[config.FEARED_PROXY].pct_change() if config.FEARED_PROXY in prices.columns else None

    br = simulate_basket(cand_prices, START, END, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, universe.target_fn(CONSTRUCTION, cand_prices))
    base_r = br.returns
    core_r = compose(base_r, spy_ret, use_valuation=True)   # "+val"
    plain_r = compose(base_r, spy_ret, use_valuation=False)

    sleeve_r = hedge_overlay.sleeve_returns(prices)
    short_r = hedge_overlay.short_returns(prices)

    idx = core_r.index
    ry_mask = hedge_overlay.real_yield_rising_mask(dfii10, idx)
    cep_mask = cep_rising_mask(cep, idx)
    both_mask = ry_mask & cep_mask
    scaler = cep_scaler(cep, idx)

    def apply_scaled_short(core, sleeve, short, weight_series, use_sleeve=True):
        w_s = config.HEDGE_SLEEVE_WEIGHT if use_sleeve else 0.0
        c = core.fillna(0.0)
        s = sleeve.reindex(core.index).fillna(0.0)
        h = short.reindex(core.index).fillna(0.0)
        w = weight_series.reindex(core.index).fillna(0.0)
        return (1.0 - w_s) * c + w_s * s - config.HEDGE_SHORT_WEIGHT * (h * w)

    variants = {
        "no_short (+val+sleeve)": hedge_overlay.apply_hedge(
            core_r, sleeve_r, short_r, ry_mask, use_sleeve=True, use_short=False),
        "real_yield_only (shipped)": hedge_overlay.apply_hedge(
            core_r, sleeve_r, short_r, ry_mask, use_sleeve=True, use_short=True),
        "cep_only": apply_scaled_short(core_r, sleeve_r, short_r, cep_mask.astype(float)),
        "cep_AND_real_yield": apply_scaled_short(core_r, sleeve_r, short_r, both_mask.astype(float)),
        "cep_continuous_scaler": apply_scaled_short(core_r, sleeve_r, short_r, scaler),
    }

    rows = []
    for name, r in variants.items():
        m = compute_metrics(r, config.RF, config.ANN)
        calmar = (m["cagr"] / abs(m["max_dd"])) if m["max_dd"] not in (0, None) and m["max_dd"] == m["max_dd"] else float("nan")
        fp = feared_pnl(r, plain_r, tan_ret)
        ep = episode_drawdowns(r, config.EPISODES)
        rows.append({
            "variant": name, "sharpe": m["sharpe"], "cagr": m["cagr"], "max_dd": m["max_dd"],
            "calmar": round(calmar, 3) if calmar == calmar else float("nan"),
            "feared_pnl": round(fp, 4) if fp == fp else float("nan"),
            **{f"dd_{k}": round(v, 3) if v == v else float("nan") for k, v in ep.items()},
        })
        for win_name, (a, b) in config.SUBWINDOWS.items():
            m2 = compute_metrics(r.loc[a:b], config.RF, config.ANN)
            print(f"    [{name}] {win_name} Sharpe={m2['sharpe']} MaxDD={m2['max_dd']}")

    df = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 200)
    print("\n=== Full-window (%s..%s) combined-book comparison, construction=%s ===" % (START, END, CONSTRUCTION))
    print(df.to_string())

    print("\n--- mask activity (fraction of days each gate is ON) ---")
    print(f"real_yield: {ry_mask.mean():.2%}   cep: {cep_mask.mean():.2%}   "
          f"both: {both_mask.mean():.2%}   scaler mean: {scaler.mean():.3f}")


if __name__ == "__main__":
    main()
