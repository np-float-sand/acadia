"""
Rank-IC test on the length/size-residualized raw_count panel, per
docs/backlog-pricing-power-length-control-preregistration.md §3-4.

Imports rank_ic_series / summarize / WINDOWS / HORIZONS / get_monthly_prices
directly from ic_test.py -- same method, same windows, same gate as the
original pre-registered test, applied here to residualize.py's output
instead of a raw score column. Not reimplemented.

Not production code. Throwaway research script. Run:
    python3 research/backlog_pricing_power/ic_test_residual.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from ic_test import HORIZONS, WINDOWS, get_monthly_prices, rank_ic_series, summarize
from residualize import monthly_residual_panel

_CACHE = Path(__file__).parent / "_cache"


def run(controls: tuple[str, ...]) -> pd.DataFrame:
    resid_panel = monthly_residual_panel(controls=controls)
    monthly_px = get_monthly_prices("2018-01-01", "2026-08-31")

    rows = []
    for h in HORIZONS:
        ic_full = rank_ic_series(resid_panel, monthly_px, h)
        for wname, (wstart, wend) in WINDOWS.items():
            ic_w = ic_full[(ic_full.index >= wstart) & (ic_full.index <= wend)]
            stats = summarize(ic_w)
            rows.append({"horizon_months": h, "window": wname, **stats})
    return pd.DataFrame(rows)


def gate_check(res_pooled_1m: dict, res_pooled_3m: dict, res_sub: dict) -> None:
    pass  # gate is applied manually when writing up results, not automated here


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    for name, ctrls in [
        ("two-control residual: raw_count ~ n_words + log_market_cap", ("n_words", "log_market_cap")),
        ("one-control residual (secondary): raw_count ~ n_words", ("n_words",)),
    ]:
        print(f"\n=== {name} ===")
        res = run(controls=ctrls)
        print(res.to_string(index=False))
