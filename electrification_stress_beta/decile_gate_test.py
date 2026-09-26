# electrification_stress_beta/decile_gate_test.py
"""Decile follow-up to the Task 5 stress-beta gate (tercile spread, FAILED, t=0.144).

Pre-registration (written before running, see
docs/electrification-stress-beta-decile-preregistration.md):
  - Cut: 10th decile minus 1st decile, fixed top-3/bottom-3 by raw ticker count.
  - Minimum monthly cross-section: 15 valid (stress_beta, fwd_ret) names.
  - Test statistic: same pooled monthly-spread t-test as gate_test.py (Task 5).
  - Window / threshold: unchanged (2021-01-01-onward primary, |t| >= 2.0 gate;
    pre-2021 reported for context only, no sign-match requirement).

Run:
    poetry run python -m electrification_stress_beta.decile_gate_test
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
DECILE_BUCKET = 3
MIN_CROSS_SECTION = 15
OUT_PATH = Path(__file__).parent / "decile_gate_result.json"


def _decile_spread(betas: pd.DataFrame) -> tuple[pd.Series, pd.Series, int, int]:
    rows = []
    n_months_seen = 0
    n_months_dropped = 0
    for date, grp in betas.groupby(level="date"):
        g = grp.dropna(subset=["stress_beta", "fwd_ret"])
        n_months_seen += 1
        if len(g) < MIN_CROSS_SECTION:
            n_months_dropped += 1
            continue
        top = g.nlargest(DECILE_BUCKET, "stress_beta")["fwd_ret"].mean()
        bot = g.nsmallest(DECILE_BUCKET, "stress_beta")["fwd_ret"].mean()
        rows.append({"date": date, "spread": top - bot, "n": len(g)})
    if not rows:
        return pd.Series(dtype=float), pd.Series(dtype=float), n_months_seen, n_months_dropped
    df = pd.DataFrame(rows).set_index("date")
    return df["spread"], df["n"], n_months_seen, n_months_dropped


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
    fwd_ret.index.name = "date"
    fwd_ret.columns.name = "ticker"
    betas = betas.join(fwd_ret.stack().rename("fwd_ret"), how="left")

    spread, n_names, n_months_seen, n_months_dropped = _decile_spread(betas)
    primary = spread.loc[PRIMARY_WINDOW_START:]
    pre = spread.loc[:PRIMARY_WINDOW_START]

    t_primary, n_primary = _gate1_tstat(primary)
    t_pre, n_pre = _gate1_tstat(pre)

    result = {
        "cut": "10th-minus-1st decile (top-3/bottom-3 by count)",
        "min_cross_section": MIN_CROSS_SECTION,
        "t_primary_2021on": t_primary, "n_primary": n_primary,
        "t_pre_2021": t_pre, "n_pre": n_pre,
        "gate1_pass": bool(abs(t_primary) >= GATE_T_THRESHOLD) if t_primary == t_primary else False,
        "sign_primary": "positive" if (t_primary == t_primary and t_primary > 0) else "negative",
        "avg_monthly_cross_section_n": float(n_names.mean()) if len(n_names) else float("nan"),
        "months_seen_total": n_months_seen,
        "months_dropped_below_min_n": n_months_dropped,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
