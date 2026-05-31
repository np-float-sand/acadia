"""
Grid Resilience Strategy — full pipeline runner.

Steps:
  1. Fetch equity prices for the utility universe
  2. Fetch grid LMP + load data for each ISO
  3. Detect stress events (named + algo-derived)
  4. Build Grid Stress Index per ISO
  5. Estimate conditional stress betas per ticker
  6. Construct Grid Resilience factor scores
  7. Build long/short portfolio weights
  8. Run backtest and report performance

Run:
    python -m grid_resilience.main

    # ERCOT-only quick demo (faster, no gridstatus for other ISOs):
    python -m grid_resilience.main --iso ERCOT --start 2020-01-01

Environment variables (set before running):
    ERCOT_API_TOKEN       : ERCOT public API token (free at developer.ercot.com)
    EIA_API_KEY           : EIA Open Data API key  (free at eia.gov/opendata)
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from grid_resilience.config import (
    BACKTEST_START, BACKTEST_END,
    SUPPORTED_ISOS, REBALANCE_FREQ,
    PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N,
    CONGESTION_SPREAD_ISOS, ISO_ZONE_LOCATION_TYPE,
    XLU_HEDGE,
)
from grid_resilience.data.universe import (
    UNIVERSE, LMP_MAPPED, get_ticker_iso,
)
from grid_resilience.data.equity_prices import (
    fetch_prices, fetch_returns, fetch_sector_return,
)
from grid_resilience.data.grid_data import (
    fetch_lmp, fetch_load, daily_lmp_summary, daily_load_summary,
    daily_spread_summary, fill_congestion_from_spread,
)
from grid_resilience.signals.stress_events import build_full_event_calendar
from grid_resilience.signals.grid_stress_index import build_multi_iso_gsi
from grid_resilience.signals.conditional_beta import (
    rolling_stress_betas, compute_stress_betas,
)
from grid_resilience.factor.resilience_score import (
    build_factor, build_rolling_factor,
)
from grid_resilience.portfolio.construction import (
    build_rolling_weights, weights_to_matrix, portfolio_summary,
)
from grid_resilience.portfolio.backtest import (
    run_backtest, plot_performance, print_metrics,
)


def run(
    isos:      list[str] = SUPPORTED_ISOS,
    start:     str       = BACKTEST_START,
    end:       str       = BACKTEST_END,
    n_long:    int       = PORTFOLIO_LONG_N,
    n_short:   int       = PORTFOLIO_SHORT_N,
    xlu_hedge: bool      = XLU_HEDGE,
    plot:      bool      = True,
    save_dir:  str       = "output",
) -> dict:
    """
    Execute the full Grid Resilience pipeline.

    Returns a dict with keys: pnl, metrics, factor_scores, event_calendar.
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)

    # ── 1. Equity prices ──────────────────────────────────────────────────────
    print("\n[1/7] Fetching equity prices…")
    tickers = [t for t in UNIVERSE if get_ticker_iso(t) in isos or get_ticker_iso(t) is not None]
    prices  = fetch_prices(tickers, start, end)
    returns = fetch_returns(tickers, start, end)

    # Restrict to tickers that actually downloaded
    tickers  = [t for t in tickers if t in returns.columns]
    returns  = returns[tickers]

    sector_r = fetch_sector_return(tickers, start, end)

    # ── XLU hedge returns (fetched separately, not factor-scored) ────────────
    if xlu_hedge:
        xlu_prices = fetch_prices(["XLU"], start, end)
        if not xlu_prices.empty and "XLU" in xlu_prices.columns:
            xlu_ret = np.log(xlu_prices["XLU"] / xlu_prices["XLU"].shift(1)).dropna()
            xlu_ret.name = "XLU"
            returns = returns.join(xlu_ret, how="left")
            tickers = list(returns.columns)

    ticker_iso_map = {t: get_ticker_iso(t) for t in tickers}

    # ── 2. Grid data ──────────────────────────────────────────────────────────
    print("\n[2/7] Fetching grid LMP and load data…")
    daily_lmp_by_iso:  dict[str, pd.DataFrame] = {}
    daily_load_by_iso: dict[str, pd.DataFrame] = {}

    def _fetch_iso_data(iso: str) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        lmp_raw  = fetch_lmp(iso, start, end)
        load_raw = fetch_load(iso, start, end)
        zone_raw = pd.DataFrame()
        if iso in CONGESTION_SPREAD_ISOS:
            zone_loc = ISO_ZONE_LOCATION_TYPE.get(iso)
            zone_raw = fetch_lmp(iso, start, end, location_type=zone_loc) if zone_loc else lmp_raw
        return iso, lmp_raw, load_raw, zone_raw

    with ThreadPoolExecutor(max_workers=len(isos)) as ex:
        futures = {ex.submit(_fetch_iso_data, iso): iso for iso in isos}
        for future in as_completed(futures):
            iso = futures[future]
            try:
                _, lmp_raw, load_raw, zone_raw = future.result()
            except Exception as exc:
                print(f"  [warn] {iso} data fetch failed: {exc}")
                continue

            if not lmp_raw.empty:
                daily_lmp = daily_lmp_summary(lmp_raw, iso)
                if iso in CONGESTION_SPREAD_ISOS and not zone_raw.empty:
                    spread_df = daily_spread_summary(zone_raw)
                    daily_lmp = fill_congestion_from_spread(daily_lmp, spread_df)
                daily_lmp_by_iso[iso] = daily_lmp
            else:
                print(f"  [warn] No LMP data for {iso} — skipping.")

            if not load_raw.empty:
                daily_load_by_iso[iso] = daily_load_summary(load_raw)

    if not daily_lmp_by_iso:
        print("[ERROR] No grid data retrieved.  Check gridstatus installation and ISO connectivity.")
        sys.exit(1)

    # ── 3. Stress events ──────────────────────────────────────────────────────
    print("\n[3/7] Building stress event calendar…")
    events_df = build_full_event_calendar(daily_lmp_by_iso, isos=isos)
    print(f"  {len(events_df)} events identified "
          f"({events_df['type'].value_counts().to_dict()})")
    events_df.to_csv(output_dir / "stress_events.csv", index=False)

    # ── 4. Grid Stress Index ──────────────────────────────────────────────────
    print("\n[4/7] Computing Grid Stress Index…")
    gsi_by_iso = build_multi_iso_gsi(daily_lmp_by_iso, daily_load_by_iso, events_df)

    for iso, gsi_df in gsi_by_iso.items():
        if not gsi_df.empty:
            gsi_df.to_csv(output_dir / f"gsi_{iso.lower()}.csv")
            high_stress = (gsi_df["gsi"] >= 0.6).sum()
            print(f"  {iso}: {high_stress} high-stress days (GSI ≥ 0.6) in period")

    # ── 5. Stress betas ───────────────────────────────────────────────────────
    print("\n[5/7] Estimating stress betas…")
    rebalance_dates = pd.date_range(start, end, freq=REBALANCE_FREQ)
    # Snap each calendar month-end to the nearest prior trading day so that
    # non-trading-day month-ends (e.g. Sat/Sun) still produce valid weights.
    def _snap(d):
        if d in returns.index:
            return d
        pos = returns.index.searchsorted(d, side="right") - 1
        return returns.index[pos] if pos >= 0 else None
    rebalance_dates = pd.DatetimeIndex(
        [_snap(d) for d in rebalance_dates if _snap(d) is not None]
    ).unique()

    rolling_betas = rolling_stress_betas(
        returns          = returns,
        gsi_by_iso       = gsi_by_iso,
        ticker_iso_map   = ticker_iso_map,
        sector_return    = sector_r,
        rebalance_dates  = rebalance_dates,
    )

    if rolling_betas.empty:
        print("  [warn] Could not estimate stress betas — insufficient overlapping data.")
        print("  Tip: ensure grid data and equity data cover the same date range.")
        sys.exit(1)

    # ── 6. Factor scores ──────────────────────────────────────────────────────
    print("\n[6/7] Building Grid Resilience factor scores…")
    rolling_factors = build_rolling_factor(rolling_betas)
    rolling_factors.to_csv(output_dir / "factor_scores.csv", index=False)

    if not rolling_factors.empty:
        latest = rolling_factors[rolling_factors["date"] == rolling_factors["date"].max()]
        print("\n  Latest factor scores:")
        for _, row in latest.sort_values("factor_score", ascending=False).iterrows():
            bar = "█" * int(abs(row["factor_score"]) * 5)
            sign = "▲" if row["factor_score"] > 0 else "▼"
            print(f"    {row['ticker']:5s} {sign} {row['factor_score']:+.3f}  {bar}")

    # ── 7. Portfolio + backtest ───────────────────────────────────────────────
    print("\n[7/7] Constructing portfolio and running backtest…")
    weights_df = build_rolling_weights(
        rolling_factors,
        rebalance_dates=pd.DatetimeIndex(rolling_factors["date"].unique()),
        n_long=n_long,
        n_short=n_short,
        xlu_hedge=xlu_hedge,
    )

    weights_matrix = weights_to_matrix(weights_df, returns.index, list(returns.columns))
    pnl, metrics   = run_backtest(weights_matrix, returns, events_df)

    pnl.to_csv(output_dir / "pnl.csv")
    print_metrics(metrics)

    if plot:
        plot_performance(
            pnl, events_df, metrics,
            save_path=output_dir / "backtest_performance.png",
        )

    return {
        "pnl":            pnl,
        "metrics":        metrics,
        "factor_scores":  rolling_factors,
        "event_calendar": events_df,
        "gsi":            gsi_by_iso,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid Resilience Strategy pipeline")
    p.add_argument("--iso",    nargs="+", default=SUPPORTED_ISOS,
                   help="ISOs to include (default: all 5)")
    p.add_argument("--start",  default=BACKTEST_START, help="Start date YYYY-MM-DD")
    p.add_argument("--end",    default=BACKTEST_END,   help="End date YYYY-MM-DD")
    p.add_argument("--long",   type=int, default=PORTFOLIO_LONG_N,  help="Long book size")
    p.add_argument("--short",  type=int, default=PORTFOLIO_SHORT_N, help="Short book size")
    p.add_argument(
        "--xlu-hedge", dest="xlu_hedge",
        action=argparse.BooleanOptionalAction,
        default=XLU_HEDGE,
        help="Replace short book with -0.5 XLU hedge (default: on)",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib charts")
    p.add_argument("--output", default="output", help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        isos      = args.iso,
        start     = args.start,
        end       = args.end,
        n_long    = args.long,
        n_short   = args.short,
        xlu_hedge = args.xlu_hedge,
        plot      = not args.no_plot,
        save_dir  = args.output,
    )
