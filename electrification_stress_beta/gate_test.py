# electrification_stress_beta/gate_test.py
"""Stress-beta gate test for the Screened Electrification supplier universe.
Spec: docs/superpowers/specs/2026-09-15-screened-electrification-sow-plan-design.md §3.4.

Pre-registered gate, fixed before this script is run:
  Gate 1: pooled monthly tercile-spread t-stat |t| >= 2 on the 2021-01-01-onward window
          (the tight-grid era -- see the spec's regime-dating note). The pre-2021 window
          is reported for context only, not required to match sign.

Run:
    poetry run python -m electrification_stress_beta.gate_test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grid_equipment_basket.data.prices import fetch_prices
from grid_resilience.config import SUPPORTED_ISOS
from grid_resilience.signals.conditional_beta import rolling_stress_betas

from electrification_strategy import universe
from electrification_stress_beta import national_gsi

START, END = "2017-06-01", "2026-08-31"
PRIMARY_WINDOW_START = "2021-01-01"
GATE_T_THRESHOLD = 2.0
OUT_PATH = Path(__file__).parent / "gate_result.json"


def _tercile_spread(betas: pd.DataFrame) -> pd.Series:
    rows = []
    for date, grp in betas.groupby(level="date"):
        g = grp.dropna(subset=["stress_beta", "fwd_ret"])
        if len(g) < 6:
            continue
        n = len(g)
        top = g.nlargest(max(1, n // 3), "stress_beta")["fwd_ret"].mean()
        bot = g.nsmallest(max(1, n // 3), "stress_beta")["fwd_ret"].mean()
        rows.append({"date": date, "spread": top - bot})
    return pd.DataFrame(rows).set_index("date")["spread"] if rows else pd.Series(dtype=float)


def _gate1_tstat(spread: pd.Series) -> tuple[float, int]:
    s = spread.dropna()
    if len(s) < 3:
        return float("nan"), len(s)
    t = float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))
    return t, len(s)


def main() -> dict:
    tickers = sorted(universe.load_seed().index)
    prices = fetch_prices(tickers + ["SPY"], START, END)
    rets = prices[tickers].pct_change()
    sector_r = rets.mean(axis=1).rename("sector_return")

    national = national_gsi.fetch_and_build_national_gsi(START, END, isos=SUPPORTED_ISOS)
    gsi_by_iso = {national_gsi.NATIONAL_KEY: national}
    ticker_iso_map = national_gsi.uniform_ticker_iso_map(tickers)

    rebalance_dates = pd.date_range(START, END, freq="MS")
    betas = rolling_stress_betas(rets, gsi_by_iso, ticker_iso_map, sector_r, rebalance_dates)

    fwd_ret = (rets.resample("MS").apply(lambda x: (1 + x).prod() - 1)).shift(-1)
    # rolling_stress_betas() names its MultiIndex levels ['date', 'ticker'];
    # fetch_prices()'s DataFrame carries index/columns names 'Date'/'Ticker'
    # (or none, for synthetic frames), which .stack() would otherwise carry
    # through verbatim -- pandas .join() then raises "cannot join with no
    # overlapping index names". Normalize names before stacking so the join
    # aligns on values, not naming happenstance.
    fwd_ret.index.name = "date"
    fwd_ret.columns.name = "ticker"
    betas = betas.join(fwd_ret.stack().rename("fwd_ret"), how="left")

    spread = _tercile_spread(betas)
    primary = spread.loc[PRIMARY_WINDOW_START:]
    pre = spread.loc[:PRIMARY_WINDOW_START]

    t_primary, n_primary = _gate1_tstat(primary)
    t_pre, n_pre = _gate1_tstat(pre)

    result = {
        "t_primary_2021on": t_primary, "n_primary": n_primary,
        "t_pre_2021": t_pre, "n_pre": n_pre,
        "gate1_pass": bool(abs(t_primary) >= GATE_T_THRESHOLD) if t_primary == t_primary else False,
        "sign_primary": "positive" if (t_primary == t_primary and t_primary > 0) else "negative",
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
