#!/usr/bin/env python
"""
Grid search over key Grid Resilience strategy parameters.

Edit SEARCH_GRID at the top to change which values are explored.

Parallelism model
─────────────────
  Outer loop  (stress detection params → events → GSI → betas → factor scores)
              is CPU-bound.  Each combination runs in its own process via
              ProcessPoolExecutor, using all available cores.

  Inner loop  (portfolio params → weights → backtest)
              is cheap matrix math; runs sequentially inside each worker.

  Data fetch  (equity prices + grid LMP/load) runs once on the main process
              using threaded IO, then the immutable DataFrames are pickled
              to each worker.  On macOS the spawn start-method is used
              automatically; fork is unsafe with Objective-C runtimes.

Usage:
    python grid_search.py                              # all ISOs, full history
    python grid_search.py --iso ERCOT --start 2020-01-01  # fast smoke-test
    python grid_search.py --workers 4                  # cap parallelism

Results are printed ranked by Sharpe and saved to output/grid_search_results.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Search grid — edit these ──────────────────────────────────────────────────
# Each key maps to a list of values to try.  Comment out values (or reduce to
# a single-element list) to hold a parameter fixed.

SEARCH_GRID: dict[str, list] = {
    # ── Stress detection ──────────────────────────────────────────────────────
    # Percentile of lmp_max that defines a "spike day".
    # Higher = fewer, more extreme events.  Original design intent: 0.99.
    "stress_spike_pct": [0.95, 0.97, 0.99],

    # Congestion fraction threshold: fraction of LMP that must be congestion
    # component for a day to count as a congestion day.
    "stress_cong_threshold": [0.30, 0.40, 0.50],

    # Minimum consecutive high-congestion days to form an event.
    "stress_cong_min_days": [3, 7, 10],

    # ── Portfolio construction ────────────────────────────────────────────────
    # Long-book size.
    "n_long": [3, 5, 7],

    # XLU hedge: True → short book = -0.5 XLU (sector hedge, not individual names).
    # Reduces idiosyncratic short risk; good when utilities have a sector tailwind.
    "xlu_hedge": [True, False],

    # Short-book size.  Ignored when xlu_hedge=True.
    "n_short": [3, 5],
}

# ── Fixed parameters (held constant across all runs) ──────────────────────────
FIXED: dict[str, int | float] = {
    "stress_spike_min_days":  3,  # min consecutive spike days to form an event
    "stress_spike_merge_gap": 1,  # merge events separated by fewer days
}

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
RESULTS_CSV = OUTPUT_DIR / "grid_search_results.csv"


# ══════════════════════════════════════════════════════════════════════════════
# Worker function — must be a module-level def so it can be pickled by spawn
# ══════════════════════════════════════════════════════════════════════════════

def _outer_worker(payload: dict) -> dict:
    """
    Run one outer (stress-param) combination inside a worker process.

    Receives a payload dict with:
      - outer_params : {stress_spike_pct, stress_cong_threshold, stress_cong_min_days}
      - fixed        : FIXED dict
      - inner_combos : list of (n_long, xlu_hedge, n_short) tuples
      - data         : {returns, sector_r, ew_ret, ticker_iso_map,
                        daily_lmp_by_iso, daily_load_by_iso,
                        rebalance_dates, isos, xlu_ret_available}

    Returns a dict with:
      - outer_params
      - results       : list of metric dicts (one per inner combo)
      - stress_day_pct, n_events, elapsed
      - error         : str or None
    """
    # Lazy imports — each spawned process re-imports from scratch.
    from grid_resilience.signals.stress_events import (
        get_named_events, detect_lmp_spike_events, detect_congestion_events,
    )
    from grid_resilience.signals.grid_stress_index import build_multi_iso_gsi
    from grid_resilience.signals.conditional_beta import rolling_stress_betas
    from grid_resilience.factor.resilience_score import build_rolling_factor
    from grid_resilience.portfolio.construction import (
        build_rolling_weights, weights_to_matrix,
    )
    from grid_resilience.portfolio.backtest import run_backtest

    t0 = time.perf_counter()

    outer = payload["outer_params"]
    fixed = payload["fixed"]
    inner_combos = payload["inner_combos"]
    d = payload["data"]

    returns          = d["returns"]
    sector_r         = d["sector_r"]
    ew_ret           = d["ew_ret"]
    ticker_iso_map   = d["ticker_iso_map"]
    daily_lmp_by_iso = d["daily_lmp_by_iso"]
    daily_load_by_iso = d["daily_load_by_iso"]
    rebalance_dates  = d["rebalance_dates"]
    isos             = d["isos"]
    xlu_ret          = d.get("xlu_ret")  # may be None

    spike_pct      = outer["stress_spike_pct"]
    cong_threshold = outer["stress_cong_threshold"]
    cong_min_days  = outer["stress_cong_min_days"]

    try:
        # Build stress event calendar with these thresholds
        frames = [get_named_events(isos=isos)]
        for iso, df in daily_lmp_by_iso.items():
            frames.append(detect_lmp_spike_events(
                df, iso,
                spike_pct=spike_pct,
                min_duration_days=fixed["stress_spike_min_days"],
                merge_gap_days=fixed["stress_spike_merge_gap"],
            ))
            frames.append(detect_congestion_events(
                df, iso,
                congestion_threshold=cong_threshold,
                min_duration_days=cong_min_days,
            ))
        non_empty = [f for f in frames if not f.empty]
        events_df = (
            pd.concat(non_empty, ignore_index=True)
            .sort_values("window_start")
            .reset_index(drop=True)
            if non_empty else pd.DataFrame()
        )

        # Stress-day fraction
        if not events_df.empty:
            mask = pd.Series(False, index=returns.index)
            for _, row in events_df.iterrows():
                mask |= (returns.index >= row["window_start"]) & (returns.index <= row["window_end"])
            stress_day_pct = float(mask.mean())
        else:
            stress_day_pct = 0.0

        # GSI and rolling betas (expensive)
        gsi_by_iso = build_multi_iso_gsi(daily_lmp_by_iso, daily_load_by_iso, events_df)
        rolling_betas = rolling_stress_betas(
            returns=returns,
            gsi_by_iso=gsi_by_iso,
            ticker_iso_map=ticker_iso_map,
            sector_return=sector_r,
            rebalance_dates=rebalance_dates,
        )

        if rolling_betas.empty:
            return {
                "outer_params": outer, "results": [],
                "stress_day_pct": stress_day_pct, "n_events": len(events_df),
                "elapsed": time.perf_counter() - t0, "error": "no stress betas",
            }

        rolling_factors = build_rolling_factor(rolling_betas)
        factor_dates = pd.DatetimeIndex(rolling_factors["date"].unique())

        # Inner loop: portfolio param combinations
        results = []
        seen_xlu_n_short: set = set()  # de-duplicate xlu_hedge=True × n_short variants

        for (n_long, xlu_hedge, n_short) in inner_combos:
            # When XLU hedge is on, n_short is irrelevant — only run once per n_long
            if xlu_hedge:
                key = ("xlu", n_long)
                if key in seen_xlu_n_short:
                    continue
                seen_xlu_n_short.add(key)

            if xlu_hedge and xlu_ret is None:
                continue

            weights_df = build_rolling_weights(
                rolling_factors,
                rebalance_dates=factor_dates,
                n_long=n_long,
                n_short=n_short,
                xlu_hedge=xlu_hedge,
            )
            if weights_df.empty:
                continue

            weights_matrix = weights_to_matrix(
                weights_df, returns.index, list(returns.columns)
            )
            _, metrics = run_backtest(
                weights_matrix, returns, events_df,
                benchmark_returns=xlu_ret,
                ew_returns=ew_ret,
            )

            results.append({
                **outer,
                "n_long":    n_long,
                "xlu_hedge": xlu_hedge,
                "n_short":   n_short if not xlu_hedge else None,
                "stress_day_pct": round(stress_day_pct * 100, 1),
                "n_events":  len(events_df),
                "sharpe":    metrics.get("full_sharpe"),
                "ann_return": metrics.get("full_ann_return"),
                "ann_vol":   metrics.get("full_ann_vol"),
                "max_dd":    metrics.get("full_max_dd"),
                "hit_rate":  metrics.get("full_hit_rate"),
                "sortino":   metrics.get("full_sortino"),
                "stress_sharpe": metrics.get("stress_sharpe"),
                "calm_sharpe":   metrics.get("calm_sharpe"),
            })

    except Exception as exc:
        return {
            "outer_params": outer, "results": [],
            "stress_day_pct": 0.0, "n_events": 0,
            "elapsed": time.perf_counter() - t0, "error": str(exc),
        }

    return {
        "outer_params": outer,
        "results": results,
        "stress_day_pct": round(stress_day_pct * 100, 1),
        "n_events": len(events_df),
        "elapsed": time.perf_counter() - t0,
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_grid_search(
    isos: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor as _ThreadPool
    from grid_resilience.config import (
        BACKTEST_START, BACKTEST_END, SUPPORTED_ISOS, REBALANCE_FREQ,
        CONGESTION_SPREAD_ISOS, ISO_ZONE_LOCATION_TYPE,
    )
    from grid_resilience.data.universe import UNIVERSE, get_ticker_iso
    from grid_resilience.data.equity_prices import fetch_returns
    from grid_resilience.data.grid_data import (
        fetch_lmp, fetch_load, daily_lmp_summary, daily_load_summary,
        daily_spread_summary, fill_congestion_from_spread,
    )

    isos  = isos  or SUPPORTED_ISOS
    start = start or BACKTEST_START
    end   = end   or BACKTEST_END

    n_cpus = os.cpu_count() or 2
    # Leave one core for the OS; cap at 8 to avoid diminishing returns
    if max_workers is None:
        max_workers = min(n_cpus - 1, 8)
    max_workers = max(1, max_workers)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 65)
    print(f"GRID SEARCH  |  CPUs: {n_cpus}  |  workers: {max_workers}")
    print("=" * 65)

    # ── 1. Fetch data once (main process) ─────────────────────────────────────
    print("\n[data] Fetching equity prices…")
    universe_tickers = [t for t in UNIVERSE if get_ticker_iso(t) is not None]
    fetch_tickers = list(dict.fromkeys(universe_tickers + ["XLU"]))
    all_returns = fetch_returns(fetch_tickers, start, end)

    tickers = [t for t in universe_tickers if t in all_returns.columns]
    returns = all_returns[tickers]

    xlu_ret: pd.Series | None = None
    if "XLU" in all_returns.columns and not all_returns["XLU"].isna().all():
        xlu_ret = all_returns["XLU"].dropna()
        returns = returns.join(xlu_ret, how="left")
        tickers = list(returns.columns)

    sector_r = (
        xlu_ret.rename("sector_return")
        if xlu_ret is not None
        else all_returns[tickers].mean(axis=1).rename("sector_return")
    )
    universe_in_returns = [t for t in universe_tickers if t in all_returns.columns]
    ew_ret = all_returns[universe_in_returns].mean(axis=1).rename("ew_universe")
    ticker_iso_map = {t: get_ticker_iso(t) for t in tickers}

    # Snap month-end rebalance dates to nearest prior trading day
    raw_dates = pd.date_range(start, end, freq=REBALANCE_FREQ)
    def _snap(d):
        if d in returns.index:
            return d
        pos = returns.index.searchsorted(d, side="right") - 1
        return returns.index[pos] if pos >= 0 else None
    rebalance_dates = pd.DatetimeIndex(
        [_snap(d) for d in raw_dates if _snap(d) is not None]
    ).unique()

    print(f"[data] Universe: {len(universe_in_returns)} tickers  "
          f"Dates: {returns.index[0].date()} → {returns.index[-1].date()}  "
          f"Rebalances: {len(rebalance_dates)}")

    print("[data] Fetching grid LMP and load data (parallel by ISO)…")
    daily_lmp_by_iso:  dict[str, pd.DataFrame] = {}
    daily_load_by_iso: dict[str, pd.DataFrame] = {}

    def _fetch_iso(iso: str):
        lmp_raw  = fetch_lmp(iso, start, end)
        load_raw = fetch_load(iso, start, end)
        zone_raw = pd.DataFrame()
        if iso in CONGESTION_SPREAD_ISOS:
            zone_loc = ISO_ZONE_LOCATION_TYPE.get(iso)
            zone_raw = fetch_lmp(iso, start, end, location_type=zone_loc) if zone_loc else lmp_raw
        return iso, lmp_raw, load_raw, zone_raw

    with _ThreadPool(max_workers=len(isos)) as ex:
        futures = {ex.submit(_fetch_iso, iso): iso for iso in isos}
        for fut in as_completed(futures):
            iso = futures[fut]
            try:
                _, lmp_raw, load_raw, zone_raw = fut.result()
            except Exception as exc:
                print(f"  [warn] {iso}: {exc}")
                continue
            if not lmp_raw.empty:
                daily_lmp = daily_lmp_summary(lmp_raw, iso)
                if iso in CONGESTION_SPREAD_ISOS and not zone_raw.empty:
                    spread_df = daily_spread_summary(zone_raw)
                    daily_lmp = fill_congestion_from_spread(daily_lmp, spread_df)
                daily_lmp_by_iso[iso] = daily_lmp
            if not load_raw.empty:
                daily_load_by_iso[iso] = daily_load_summary(load_raw)

    if not daily_lmp_by_iso:
        raise RuntimeError("No grid data retrieved.")

    # Reference Sharpe values (XLU, EW basket)
    rf_daily = 0.04 / 252
    xlu_sharpe = ew_sharpe = float("nan")
    if xlu_ret is not None:
        exc = xlu_ret - rf_daily
        xlu_sharpe = round(exc.mean() / exc.std() * np.sqrt(252), 3)
    exc_ew = ew_ret - rf_daily
    ew_sharpe = round(exc_ew.mean() / exc_ew.std() * np.sqrt(252), 3)

    print(f"\n  Reference — XLU sharpe: {xlu_sharpe:.3f}")
    print(f"  Reference — EW  sharpe: {ew_sharpe:.3f}")

    # ── 2. Build all outer combinations ───────────────────────────────────────
    outer_keys = ["stress_spike_pct", "stress_cong_threshold", "stress_cong_min_days"]
    inner_keys = ["n_long", "xlu_hedge", "n_short"]

    outer_combos = [
        dict(zip(outer_keys, vals))
        for vals in itertools.product(*[SEARCH_GRID[k] for k in outer_keys])
    ]
    inner_combos = list(itertools.product(*[SEARCH_GRID[k] for k in inner_keys]))

    total_outer = len(outer_combos)
    print(f"\n  {total_outer} stress configs × ~{len(inner_combos)} portfolio configs"
          f"  →  dispatching to {max_workers} workers\n")

    # Shared data payload (pickled once per worker on macOS spawn)
    shared_data = {
        "returns":           returns,
        "sector_r":          sector_r,
        "ew_ret":            ew_ret,
        "ticker_iso_map":    ticker_iso_map,
        "daily_lmp_by_iso":  daily_lmp_by_iso,
        "daily_load_by_iso": daily_load_by_iso,
        "rebalance_dates":   rebalance_dates,
        "isos":              isos,
        "xlu_ret":           xlu_ret,
    }

    payloads = [
        {"outer_params": outer, "fixed": FIXED, "inner_combos": inner_combos, "data": shared_data}
        for outer in outer_combos
    ]

    # ── 3. Parallel dispatch ───────────────────────────────────────────────────
    all_results: list[dict] = []
    t_wall = time.perf_counter()
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_outer_worker, p): p["outer_params"] for p in payloads}

        for fut in as_completed(future_map):
            outer = future_map[fut]
            completed += 1
            try:
                res = fut.result()
            except Exception as exc:
                print(f"  [{completed:02d}/{total_outer}] ERROR: {exc}")
                continue

            label = (f"spike={outer['stress_spike_pct']:.2f}"
                     f" cong_t={outer['stress_cong_threshold']:.2f}"
                     f" c_days={outer['stress_cong_min_days']}")

            if res["error"]:
                print(f"  [{completed:02d}/{total_outer}] {label}  → skip ({res['error']})")
                continue

            n_results   = len(res["results"])
            best_sharpe = max((r["sharpe"] for r in res["results"] if r["sharpe"] is not None), default=float("nan"))
            print(f"  [{completed:02d}/{total_outer}] {label}"
                  f"  str%={res['stress_day_pct']:.0f}%"
                  f"  runs={n_results}"
                  f"  best_sharpe={best_sharpe:.3f}"
                  f"  ({res['elapsed']:.1f}s)")

            all_results.extend(res["results"])

    wall_elapsed = time.perf_counter() - t_wall
    print(f"\n[done] Total wall-clock: {wall_elapsed:.0f}s across {max_workers} workers")

    # ── 4. Collate and display ─────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("\n[warn] No valid results produced.")
        return results_df

    results_df = results_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    results_df.index += 1

    results_df.to_csv(RESULTS_CSV)
    print(f"Full results saved → {RESULTS_CSV}")

    # Print ranked summary table
    print("\n" + "=" * 95)
    print(f"  GRID SEARCH RESULTS — ranked by Sharpe"
          f"    Reference: XLU={xlu_sharpe:.3f}  EW={ew_sharpe:.3f}")
    print("=" * 95)
    print(f"{'Rk':>3}  {'spike':>5} {'cong_t':>6} {'c_d':>3}"
          f"  {'lng':>3} {'XLU':>3} {'sht':>3}"
          f"  {'str%':>4}  {'Sharpe':>7}  {'AnnRet':>6}  {'MaxDD':>6}  {'Hit%':>5}")
    print("-" * 95)

    for rank, row in results_df.head(30).iterrows():
        sharpe_val = row["sharpe"] if row["sharpe"] is not None else float("nan")
        beat_xlu   = "✓" if sharpe_val > xlu_sharpe else " "
        xlu_flag   = "Y" if row["xlu_hedge"] else "N"
        short_str  = "-" if row["xlu_hedge"] else str(int(row["n_short"]))
        ann_ret    = row["ann_return"] or float("nan")
        max_dd     = row["max_dd"] or float("nan")
        hit_rate   = row["hit_rate"] or float("nan")
        print(
            f"{rank:>3}  "
            f"{row['stress_spike_pct']:.2f}  "
            f"{row['stress_cong_threshold']:.2f}  "
            f"{int(row['stress_cong_min_days']):>3}  "
            f"{int(row['n_long']):>3}  "
            f"{xlu_flag:>3}  "
            f"{short_str:>3}  "
            f"{row['stress_day_pct']:>4.0f}%  "
            f"{sharpe_val:>6.3f}{beat_xlu}  "
            f"{ann_ret*100:>5.1f}%  "
            f"{max_dd*100:>5.1f}%  "
            f"{hit_rate*100:>4.1f}%"
        )

    print("=" * 95)
    best = results_df.iloc[0]
    print(f"  ✓  beats XLU Sharpe ({xlu_sharpe:.3f})")
    print(f"  Top config:  spike={best['stress_spike_pct']:.2f}"
          f"  cong_thr={best['stress_cong_threshold']:.2f}"
          f"  cong_days={int(best['stress_cong_min_days'])}"
          f"  n_long={int(best['n_long'])}"
          f"  xlu_hedge={'Y' if best['xlu_hedge'] else 'N'}"
          + (f"  n_short={int(best['n_short'])}" if not best["xlu_hedge"] else ""))
    print("=" * 95)

    return results_df


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # macOS/Windows: multiprocessing requires the spawn guard
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    from grid_resilience.config import BACKTEST_START, BACKTEST_END, SUPPORTED_ISOS

    p = argparse.ArgumentParser(description="Parallelised grid search over strategy parameters")
    p.add_argument("--iso",     nargs="+", default=SUPPORTED_ISOS)
    p.add_argument("--start",   default=BACKTEST_START)
    p.add_argument("--end",     default=BACKTEST_END)
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel worker processes (default: cpu_count - 1, max 8)")
    args = p.parse_args()

    run_grid_search(isos=args.iso, start=args.start, end=args.end, max_workers=args.workers)
