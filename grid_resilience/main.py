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
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


class _Tee:
    """Write to both an underlying stream and a log file simultaneously."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def isatty(self):
        return self._stream.isatty()


def _setup_logging(output_dir: Path) -> None:
    """Clear run.log and wire up stdout/stderr + root logger to write into it."""
    log_path = output_dir / "run.log"
    log_file = open(log_path, "w", buffering=1)   # 'w' clears the file each run

    # Tee stdout/stderr so all print() calls land in both terminal and the log
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    # Root logger: drop any handlers already registered (e.g. from gridstatus
    # import-time basicConfig), then add a file handler + a console handler.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a")  # 'a': file already opened with 'w' above
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler writes to the real stderr (before Tee) to avoid double-
    # printing library logging lines in the terminal.
    sh = logging.StreamHandler(sys.__stderr__)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    print(f"[log] Run log → {log_path.resolve()}")

from grid_resilience.config import (
    BACKTEST_START, BACKTEST_END,
    SUPPORTED_ISOS, REBALANCE_FREQ,
    PORTFOLIO_LONG_N, PORTFOLIO_SHORT_N,
    CONGESTION_SPREAD_ISOS, ISO_ZONE_LOCATION_TYPE,
    XLU_HEDGE, USE_ICR,
)
from grid_resilience.data.universe import (
    UNIVERSE, LMP_MAPPED, get_ticker_iso,
)
from grid_resilience.data.equity_prices import fetch_returns, fetch_icr
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
    compute_ic, print_ic_metrics,
)


def run(
    isos:      list[str] = SUPPORTED_ISOS,
    start:     str       = BACKTEST_START,
    end:       str       = BACKTEST_END,
    n_long:    int       = PORTFOLIO_LONG_N,
    n_short:   int       = PORTFOLIO_SHORT_N,
    xlu_hedge: bool      = XLU_HEDGE,
    use_icr:   bool      = USE_ICR,
    plot:      bool      = True,
    save_dir:  str       = "output",
) -> dict:
    """
    Execute the full Grid Resilience pipeline.

    Returns a dict with keys: pnl, metrics, factor_scores, event_calendar.
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)
    _setup_logging(output_dir)

    # ── 1. Equity prices ──────────────────────────────────────────────────────
    print("\n[1/7] Fetching equity prices…")
    universe_tickers = [t for t in UNIVERSE if get_ticker_iso(t) in isos or get_ticker_iso(t) is not None]
    # XLU is included in the batch so it lands in the same monthly parquet as the
    # universe — avoids the NaN gaps that arise from narrow single-ticker downloads.
    fetch_tickers = list(dict.fromkeys(universe_tickers + ["XLU"]))
    all_returns = fetch_returns(fetch_tickers, start, end)

    # Factor universe: restrict to universe tickers that downloaded; XLU excluded
    # from scoring (build_weights already drops it, but keep returns separate here)
    tickers = [t for t in universe_tickers if t in all_returns.columns]
    returns = all_returns[tickers]

    # ── XLU: extract from the already-fetched batch ──────────────────────────
    xlu_ret: pd.Series | None = None
    if "XLU" in all_returns.columns and not all_returns["XLU"].isna().all():
        xlu_ret = all_returns["XLU"].dropna()
        xlu_ret.name = "XLU"
        returns = returns.join(xlu_ret, how="left")
        tickers = list(returns.columns)
    else:
        print("  [warn] XLU price data unavailable — disabling xlu_hedge for this run")
        xlu_hedge = False

    # Sector benchmark: XLU is a more stable proxy for the utilities sector than
    # equal-weighting 3–5 universe names (small N makes excess returns circular).
    # Fall back to equal-weight if XLU is unavailable.
    sector_r = (
        xlu_ret.rename("sector_return")
        if xlu_ret is not None
        else all_returns[tickers].mean(axis=1).rename("sector_return")
    )

    # Equal-weight universe basket (excludes XLU so it's pure stock selection)
    universe_in_returns = [t for t in universe_tickers if t in all_returns.columns]
    ew_ret = all_returns[universe_in_returns].mean(axis=1).rename("ew_universe")

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
    icr_history = None
    if use_icr:
        print("  Fetching interest coverage ratios…")
        icr_history = fetch_icr(universe_tickers)
        if not icr_history.empty:
            print(f"  ICR data: {len(icr_history)} quarters, {icr_history.notna().any().sum()} tickers")
        else:
            print("  [warn] ICR data unavailable — factor will use beta + renewables only")

    rolling_factors = build_rolling_factor(rolling_betas, icr_history=icr_history)
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
    pnl, metrics   = run_backtest(
        weights_matrix, returns, events_df,
        benchmark_returns=xlu_ret,
        ew_returns=ew_ret,
    )

    ic_series, ic_metrics = compute_ic(rolling_factors, returns, gsi_by_iso=gsi_by_iso)

    pnl.to_csv(output_dir / "pnl.csv")
    print_metrics(metrics)
    print_ic_metrics(ic_metrics, perf_metrics=metrics)

    if plot:
        plot_performance(
            pnl, events_df, metrics,
            benchmark_returns=xlu_ret,
            ew_returns=ew_ret,
            ic_series=ic_series,
            ic_metrics=ic_metrics,
            save_path=output_dir / "backtest_performance.png",
        )

    return {
        "pnl":            pnl,
        "metrics":        metrics,
        "factor_scores":  rolling_factors,
        "event_calendar": events_df,
        "gsi":            gsi_by_iso,
        "ic_metrics":     ic_metrics,
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
    p.add_argument(
        "--icr", dest="use_icr",
        action=argparse.BooleanOptionalAction,
        default=USE_ICR,
        help="Include interest coverage ratio as a factor component (default: off)",
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
        use_icr   = args.use_icr,
        plot      = not args.no_plot,
        save_dir  = args.output,
    )
