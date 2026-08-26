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
    XLU_HEDGE, USE_ICR, BUSINESS_MODEL_ARCH, REGULATED_SIGNAL,
    PEER_GROUP_CONSTRUCTION, GSI_WEIGHTS, GSI_WEIGHTS_PRICE_ONLY,
)
from grid_resilience.data.universe import (
    UNIVERSE, LMP_MAPPED, get_ticker_iso,
)
from grid_resilience.data.utility_node_map import TICKER_NODE_MAP, ERCOT_TSP_MAP
from grid_resilience.data.equity_prices import fetch_returns, fetch_icr
from grid_resilience.data.grid_data import (
    fetch_lmp, fetch_load, fetch_zonal_load, daily_lmp_summary, daily_load_summary,
    daily_spread_summary, fill_congestion_from_spread,
)
from grid_resilience.data.dc_load_data import (
    fetch_interconnection_queue, compute_dc_load_signal, fill_with_icr,
)
from grid_resilience.data.pjm_large_load_data import fetch_large_load_adjustment, cross_check_dc_signal
from grid_resilience.data.ercot_large_load_data import load_ercot_large_load_seed, compute_ercot_signal
from grid_resilience.data.hyperscaler_deals import load_hyperscaler_deals, compute_hyperscaler_signal
from grid_resilience.data.dc_demand_combined import combine_dc_demand_layers, apply_layer_precedence
from grid_resilience.signals.stress_events import build_full_event_calendar
from grid_resilience.signals.grid_stress_index import build_multi_iso_gsi
from grid_resilience.signals.conditional_beta import (
    rolling_stress_betas, compute_stress_betas,
)
from grid_resilience.factor.resilience_score import (
    build_factor, build_rolling_factor, _ICR_LAG_DAYS,
)
from grid_resilience.portfolio.construction import (
    build_rolling_weights, weights_to_matrix, portfolio_summary,
)
from grid_resilience.portfolio.backtest import (
    run_backtest, plot_performance, print_metrics,
    compute_ic, print_ic_metrics,
)


def run(
    isos:         list[str] = SUPPORTED_ISOS,
    start:        str       = BACKTEST_START,
    end:          str       = BACKTEST_END,
    n_long:       int       = PORTFOLIO_LONG_N,
    n_short:      int       = PORTFOLIO_SHORT_N,
    xlu_hedge:    bool      = XLU_HEDGE,
    use_icr:      bool      = USE_ICR,
    arch:         str | None = BUSINESS_MODEL_ARCH,
    regulated_signal: str   = REGULATED_SIGNAL,
    zone_gsi:     bool      = True,
    peer_group:   bool      = PEER_GROUP_CONSTRUCTION,
    price_only_gsi: bool    = False,
    plot:         bool      = True,
    save_dir:     str       = "output",
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

    # ── Business-model pass_through map ──────────────────────────────────────
    pass_through: pd.Series | None = None
    if arch is not None:
        pass_through = pd.Series({
            t: TICKER_NODE_MAP[t].get("pass_through", 1.0)
            for t in tickers
            if t in TICKER_NODE_MAP
        })
        use_icr = True   # regulated path requires ICR; override caller setting
        print(f"\n  [arch] Business-model arch: {arch!r} — ICR auto-enabled for regulated path")

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

    # ── PJM per-ticker zone GSI ───────────────────────────────────────────────
    # Fetch zone-level LMPs (type=ZONE) and build a separate GSI per ticker so
    # each company is scored against its home territory rather than the shared
    # system hub signal, which clusters betas and kills cross-sectional dispersion.
    if zone_gsi and "PJM" in daily_lmp_by_iso:
        print("\n  [grid] Fetching PJM zone LMPs for per-ticker GSI…")
        pjm_zone_raw = fetch_lmp("PJM", start, end, location_type="ZONE")
        if not pjm_zone_raw.empty:
            loc_col = next((c for c in pjm_zone_raw.columns if c == "location"), None)
            if loc_col:
                for ticker, info in TICKER_NODE_MAP.items():
                    if info.get("iso") != "PJM":
                        continue
                    zones = info.get("nodes", [])
                    if not zones:
                        continue
                    ticker_raw = pjm_zone_raw[pjm_zone_raw[loc_col].isin(zones)]
                    if not ticker_raw.empty:
                        daily_lmp_by_iso[f"PJM:{ticker}"] = daily_lmp_summary(ticker_raw, "PJM")
                        print(f"    [grid] PJM:{ticker} zone GSI data built from {zones}")
                    else:
                        print(f"    [grid] PJM:{ticker} — no zone data matched {zones}, using system hub GSI")
        else:
            print("  [warn] PJM zone LMP fetch returned empty — all PJM tickers use system hub GSI")

    # ── 3. Stress events ──────────────────────────────────────────────────────
    print("\n[3/7] Building stress event calendar…")
    events_df = build_full_event_calendar(daily_lmp_by_iso, isos=isos)
    print(f"  {len(events_df)} events identified "
          f"({events_df['type'].value_counts().to_dict()})")
    events_df.to_csv(output_dir / "stress_events.csv", index=False)

    # ── 4. Grid Stress Index ──────────────────────────────────────────────────
    print("\n[4/7] Computing Grid Stress Index…")
    gsi_weights = GSI_WEIGHTS_PRICE_ONLY if price_only_gsi else GSI_WEIGHTS
    gsi_by_iso = build_multi_iso_gsi(daily_lmp_by_iso, daily_load_by_iso, events_df, weights=gsi_weights)

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
    if regulated_signal not in ("icr", "dc_queue", "dc_multi"):
        raise ValueError(
            f"Unrecognized regulated_signal={regulated_signal!r}. "
            "Expected 'icr', 'dc_queue', or 'dc_multi' (note: underscore form, not the "
            "CLI's hyphenated 'dc-multi')."
        )

    icr_history = None
    regulated_signal_lag_days = _ICR_LAG_DAYS
    if use_icr:
        print(f"  Fetching regulated-path signal ({regulated_signal})…")
        icr_history = fetch_icr(universe_tickers)
        if not icr_history.empty:
            print(f"  ICR data: {len(icr_history)} quarters, {icr_history.notna().any().sum()} tickers")
        else:
            print("  [warn] ICR data unavailable — factor will use beta + renewables only")

        if regulated_signal == "dc_queue":
            queue = fetch_interconnection_queue()
            # zone_size() looks back 365 days from each as_of, so the zonal-load
            # fetch must start a year before the backtest start — otherwise the
            # first backtest year sees a partial, seasonally-biased window.
            zonal_load_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
            zonal_load = fetch_zonal_load(zonal_load_start, end)

            if queue.empty or zonal_load.empty:
                print(
                    "  [WARNING] DC load signal unavailable — "
                    f"{'interconnection queue' if queue.empty else ''}"
                    f"{' and ' if queue.empty and zonal_load.empty else ''}"
                    f"{'zonal load' if zonal_load.empty else ''} fetch came back "
                    "empty (missing PJM_API_KEY, PJM outage, or schema drift). "
                    "Falling back to regulated_signal='icr' for this run."
                )
            else:
                dc_history = compute_dc_load_signal(
                    tickers=universe_tickers,
                    node_map=TICKER_NODE_MAP,
                    queue=queue,
                    zonal_load=zonal_load,
                    as_of_dates=list(rebalance_dates),
                )
                merged_history = fill_with_icr(dc_history, icr_history, lag_days=_ICR_LAG_DAYS)
                n_real   = int(dc_history.notna().any().sum())
                n_filled = int((merged_history.notna().any() & dc_history.isna().all()).sum())
                print(f"  DC load signal: {n_real} PJM tickers with real data, "
                      f"{n_filled} filled from ICR")
                icr_history = merged_history
                regulated_signal_lag_days = 0

        elif regulated_signal == "dc_multi":
            queue = fetch_interconnection_queue()
            zonal_load_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
            zonal_load = fetch_zonal_load(zonal_load_start, end)

            if queue.empty or zonal_load.empty:
                # Mirror the dc_queue branch above: the PJM generation-queue layer
                # is dc-multi's only long, dispersed history (ERCOT yields no
                # per-ticker MW at all, and the hyperscaler layer only starts in
                # 2024), so losing it degrades dc-multi to something far weaker
                # than the name implies. Warn loudly and fall back rather than
                # silently substituting an empty DataFrame and reporting apparent
                # success (whole-branch review finding #5).
                print(
                    "  [WARNING] Multi-source DC demand signal unavailable — "
                    f"{'interconnection queue' if queue.empty else ''}"
                    f"{' and ' if queue.empty and zonal_load.empty else ''}"
                    f"{'zonal load' if zonal_load.empty else ''} fetch came back "
                    "empty (missing PJM_API_KEY, PJM outage, or schema drift). "
                    "The PJM generation-queue layer is dc-multi's only long-history "
                    "layer, so the remaining layers (ERCOT, hyperscaler deals) are "
                    "not a usable substitute. Falling back to regulated_signal='icr' "
                    "for this run."
                )
            else:
                pjm_dc_history = compute_dc_load_signal(
                    tickers=universe_tickers, node_map=TICKER_NODE_MAP,
                    queue=queue, zonal_load=zonal_load, as_of_dates=list(rebalance_dates),
                )

                ercot_seed = load_ercot_large_load_seed()
                ercot_history = compute_ercot_signal(
                    tickers=universe_tickers, tsp_map=ERCOT_TSP_MAP,
                    ercot_history=ercot_seed, as_of_dates=list(rebalance_dates),
                )

                deals = load_hyperscaler_deals()
                hyperscaler_history = compute_hyperscaler_signal(
                    tickers=universe_tickers, deals=deals, as_of_dates=list(rebalance_dates),
                )

                # Overlap resolution: compute_dc_load_signal / compute_ercot_signal /
                # compute_hyperscaler_signal each return a column for EVERY universe
                # ticker (NaN where uncovered), not just the tickers they actually
                # cover, and CEG/TLN are covered with real data by two layers at once
                # (both are iso="PJM", so pjm_dc_history's generation-queue proxy has
                # real values for them that collide with hyperscaler_history's
                # disclosed-deal coverage). apply_layer_precedence() resolves this
                # PER DATE — hyperscaler (disclosed deals) > PJM generation-queue
                # proxy > ERCOT TSP level — so the hyperscaler layer only wins the
                # dates on which it actually has post-disclosure information and the
                # PJM proxy keeps CEG/TLN's pre-2024 history. Only the *_for_combine
                # copies are trimmed — pjm_dc_history itself is left intact below
                # for the Layer 2 cross-check, which diagnoses the generation-queue
                # signal's own tagging and should still see every ticker it
                # actually computed a value for.
                hyperscaler_for_combine, pjm_dc_for_combine, ercot_for_combine = apply_layer_precedence(
                    [hyperscaler_history, pjm_dc_history, ercot_history]
                )

                combined = combine_dc_demand_layers(pjm_dc_for_combine, ercot_for_combine, hyperscaler_for_combine)
                n_real = int(combined.notna().any().sum()) if not combined.empty else 0
                # apply_layer_precedence() returns only the columns a layer actually
                # won, so tickers no layer covers are absent from `combined` entirely
                # — and fill_with_icr() fills NaN *cells*, so it never sees them and
                # they end up on build_factor's flat cross-sectional-mean fallback
                # instead of their own ICR reading. Reindex to the full universe
                # first so dc-multi honours the same "falling back to ICR for tickers
                # none of the layers cover" contract the dc_queue branch gets for
                # free (compute_dc_load_signal returns a column per universe ticker).
                combined = combined.reindex(index=list(rebalance_dates), columns=universe_tickers).astype(float)
                merged_history = fill_with_icr(combined, icr_history, lag_days=_ICR_LAG_DAYS)
                print(f"  Multi-source DC demand signal: {n_real} tickers with real layer data "
                      f"(PJM/ERCOT/hyperscaler), rest filled from ICR")
                icr_history = merged_history
                regulated_signal_lag_days = 0

                # Layer 2's cross-check is a diagnostic, not blended into the score
                # (see spec) — still run it and write the output so a reviewer can
                # actually see where the generation-queue signal and PJM's own
                # industry tags disagree, rather than importing it unused.
                if not pjm_dc_history.empty:
                    large_load = fetch_large_load_adjustment()
                    cross_check = cross_check_dc_signal(pjm_dc_history, large_load, TICKER_NODE_MAP)
                    cross_check.to_csv(output_dir / "dc_signal_cross_check.csv", index=False)
                    n_disagree = int((~cross_check["agrees"]).sum())
                    if n_disagree:
                        print(f"  [WARNING] {n_disagree} ticker(s) disagree between the generation-queue "
                              f"DC signal and PJM's own industry tags — see output/dc_signal_cross_check.csv")

    rolling_factors = build_rolling_factor(
        rolling_betas,
        icr_history=icr_history,
        arch=arch or "hard_switch",
        pass_through=pass_through,
        regulated_signal_lag_days=regulated_signal_lag_days,
    )
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
        grouped=peer_group,
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
    p.add_argument(
        "--zone-gsi", dest="zone_gsi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-ticker PJM zone GSI (default: on)",
    )
    p.add_argument(
        "--peer-group", dest="peer_group",
        action=argparse.BooleanOptionalAction,
        default=PEER_GROUP_CONSTRUCTION,
        help="Build long/short baskets within business-model peer groups "
             "(merchant/mixed/regulated) instead of ranking the whole universe. "
             "Cancels sector-beta exposure that whole-universe ranking carries on "
             "both legs (default: off).",
    )
    p.add_argument(
        "--arch",
        default=None,
        choices=["hard-switch", "revenue-mix", "dual-track"],
        help="Business-model signal architecture. When set, ICR is auto-enabled. "
             "Default: None (original stress-beta-only behaviour).",
    )
    p.add_argument(
        "--regulated-signal",
        dest="regulated_signal",
        default=REGULATED_SIGNAL.replace("_", "-"),
        choices=["icr", "dc-queue", "dc-multi"],
        help="Signal used for the regulated path of --arch hard-switch "
             f"(default: {REGULATED_SIGNAL.replace('_', '-')})",
    )
    p.add_argument(
        "--price-only-gsi", dest="price_only_gsi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build GSI from lmp_zscore alone (100%%), dropping congestion/reserve/"
             "event — isolates the energy-price leg of the resilience thesis from "
             "the congestion leg (default: off, uses the blended GSI_WEIGHTS)",
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
        arch      = args.arch.replace("-", "_") if args.arch else None,
        regulated_signal = args.regulated_signal.replace("-", "_"),
        zone_gsi  = args.zone_gsi,
        peer_group = args.peer_group,
        price_only_gsi = args.price_only_gsi,
        plot      = not args.no_plot,
        save_dir  = args.output,
    )
