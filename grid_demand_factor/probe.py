"""
End-to-end Probe D: does a grid-demand-sensitivity sort carry cross-sectional
information in a broad equity universe?

Run (against the cached industrial universe — no network):
    .venv/bin/python -m grid_demand_factor.probe \
        --start 2019-01-01 --end 2025-12-31 \
        --price-glob 'backlog_factor/data/cache/prices_*.parquet'

Or, if yfinance is reachable, drop --price-glob to pull the S&P 100 fresh.

Prints the pre-registered pass/fail verdict and writes CSVs to
``output_grid_demand/``. This is a triage probe — a FAIL means "not worth a full
gated build", not "the code is broken".

Result (2026-08-31): GATE FAILED — rank-IC t = −0.47, Q5−Q1 Sharpe 0.12,
non-monotone; null holds across 6 nowcast/window variants. See
``docs/triage_2026-08-31-proposals-bda.md``.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from grid_resilience.data.equity_prices import fetch_prices
from grid_demand_factor.nowcast import (
    build_monthly_nowcast,
    load_daily_gsi,
    seasonality_strength,
)
from grid_demand_factor.sensitivity import (
    evaluate_gate,
    monthly_total_returns,
    quintile_spread,
    rank_ic,
    rolling_sensitivity,
)

# S&P 100 (OEX) constituents — a broad, liquid, sector-diverse cross-section.
# Static list is fine for a triage probe (survivorship is a known caveat, noted
# in the writeup). Diagnostic ETFs appended for the contemporaneous check.
SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DHR", "DIS", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
    "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT",
    "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE",
    "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX",
    "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP",
    "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]
DIAG_ETFS = ["SPY", "XLU", "XLI", "XLK"]


def _load_price_cache(pattern: str, start: str, end: str) -> pd.DataFrame:
    """Load a wide monthly-parquet price cache directly from disk (no network).

    ``pattern`` is a glob over ``<stem>_YYYY-MM.parquet`` chunk files, e.g.
    ``backlog_factor/data/cache/prices_*.parquet``.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no price chunks match {pattern!r}")
    px = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    return px.loc[start:end]


def _contemporaneous_corr(monthly_rets: pd.DataFrame, nowcast_change: pd.Series) -> pd.Series:
    """corr(monthly return, contemporaneous nowcast change) for diagnostic tickers."""
    out = {}
    for col in monthly_rets.columns:
        s, n = monthly_rets[col].align(nowcast_change, join="inner")
        pair = pd.DataFrame({"r": s, "n": n}).dropna()
        if len(pair) >= 24:
            out[col] = pair["r"].corr(pair["n"])
    return pd.Series(out).sort_values()


def run_probe(
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    window: int = 36,
    min_periods: int = 24,
    min_months: int = 60,
    output_dir: str | Path = "output_grid_demand",
    gsi_dir: str | Path = "output",
    price_glob: str | None = None,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if price_glob:
        print(f"\n[probe-D] price cache={price_glob}  span={start}..{end}")
        prices = _load_price_cache(price_glob, start, end)
    else:
        print(f"\n[probe-D] universe={len(SP100)} names (yfinance)  span={start}..{end}")
        prices = fetch_prices(SP100 + DIAG_ETFS, start, end)

    monthly_rets_all = monthly_total_returns(prices)
    # keep only names with enough monthly history to estimate a rolling beta
    dense = monthly_rets_all.columns[monthly_rets_all.notna().sum() >= min_months].tolist()
    diag_present = [t for t in DIAG_ETFS if t in dense]
    have = [t for t in dense if t not in DIAG_ETFS]
    print(f"[probe-D] {len(have)} names with >={min_months} monthly obs "
          f"(of {monthly_rets_all.shape[1]} columns); diagnostic ETFs: {diag_present}")

    monthly_rets = monthly_rets_all[have]

    daily_gsi = load_daily_gsi(gsi_dir)
    print(f"[probe-D] GSI ISOs={list(daily_gsi.columns)}  "
          f"{daily_gsi.index.min().date()}..{daily_gsi.index.max().date()}")

    nowcast_level = build_monthly_nowcast(daily_gsi, how="level", standardize=False)
    nowcast_chg = build_monthly_nowcast(daily_gsi, how="change", standardize=True)

    seas = seasonality_strength(nowcast_level)
    diag_cols = [t for t in (diag_present + ["ITA", "PAVE", "SOXX", "XHB"])
                 if t in monthly_rets_all.columns]
    diag = _contemporaneous_corr(monthly_rets_all[diag_cols], nowcast_chg) if diag_cols else pd.Series(dtype=float)

    betas = rolling_sensitivity(monthly_rets, nowcast_chg, window=window, min_periods=min_periods)
    spread = quintile_spread(betas, monthly_rets)
    ic = rank_ic(betas, monthly_rets)
    gate = evaluate_gate(ic, spread)

    # ── persist ──
    nowcast_chg.to_csv(out / "nowcast_change.csv")
    betas.to_csv(out / "sensitivity_betas.csv")
    ic["ic"].rename("rank_ic").to_csv(out / "monthly_rank_ic.csv")
    spread["ls_returns"].rename("q5_minus_q1").to_csv(out / "q5_minus_q1_returns.csv")

    # ── report ──
    print("\n" + "=" * 64)
    print("  PROBE D — grid-demand-sensitivity cross-sectional factor")
    print("=" * 64)
    print(f"  nowcast seasonality : month_r2={seas['month_r2']:.3f}  "
          f"resid_ac1={seas['resid_ac1']:.3f}")
    print(f"  contemp. corr(Δnowcast, monthly ret):")
    for k, v in diag.items():
        print(f"      {k:<5} {v:+.3f}")
    print(f"\n  quintile mean monthly returns (next-month, EW):")
    for k, v in spread["quantile_means"].items():
        print(f"      {k}  {v*100:+.2f}%")
    print(f"  Q5-Q1: ann Sharpe={spread['ls_sharpe_ann']:.2f}  t={spread['ls_t']:.2f}  "
          f"monotone={spread['monotonic']}  months={spread['n_months']}")
    print(f"\n  rank-IC: mean={ic['mean_ic']:+.4f}  t={ic['t_stat']:+.2f}  "
          f"IR={ic['ir']:+.3f}  pct_pos={ic['pct_pos']:.2f}  months={ic['n_months']}")
    print("\n  " + "\n  ".join(gate["reasons"]))
    print("=" * 64 + "\n")

    return {"gate": gate, "ic": ic, "spread": spread, "seasonality": seas, "diag": diag}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--window", type=int, default=36)
    ap.add_argument("--min-periods", type=int, default=24)
    ap.add_argument("--min-months", type=int, default=60,
                    help="drop tickers with fewer monthly observations than this")
    ap.add_argument("--output-dir", default="output_grid_demand")
    ap.add_argument("--gsi-dir", default="output")
    ap.add_argument("--price-glob", default=None,
                    help="glob over wide monthly price parquet chunks; bypasses yfinance "
                         "(e.g. 'backlog_factor/data/cache/prices_*.parquet')")
    args = ap.parse_args()
    run_probe(
        start=args.start, end=args.end, window=args.window,
        min_periods=args.min_periods, min_months=args.min_months,
        output_dir=args.output_dir, gsi_dir=args.gsi_dir, price_glob=args.price_glob,
    )


if __name__ == "__main__":
    main()
