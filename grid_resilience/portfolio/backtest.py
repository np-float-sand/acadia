"""
Strategy backtest engine and performance analytics.

Runs the full long/short portfolio through history and computes:
  - Daily and cumulative P&L
  - Sharpe ratio, Sortino ratio, max drawdown, hit rate
  - Stress-period vs. calm-period performance attribution
  - Event-study overlay chart

Usage:
    from grid_resilience.portfolio.backtest import run_backtest, plot_performance
    pnl, metrics = run_backtest(weights_matrix, returns_df, events_df)
    plot_performance(pnl, events_df, save_path="output/backtest.png")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from grid_resilience.signals.stress_events import event_date_mask


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    weights_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    events_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    benchmark_returns: pd.Series | None = None,
    ew_returns: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute daily strategy P&L and summary performance metrics.

    Parameters
    ----------
    weights_matrix  : (date × ticker) weight matrix from construction.weights_to_matrix()
                      Forward-filled between rebalance dates.
    returns         : (date × ticker) daily log-return DataFrame
    events_df       : stress event calendar (for attribution)
    risk_free_rate  : annualised risk-free rate for Sharpe/Sortino (default 4%)

    Returns
    -------
    pnl_df  : DataFrame with columns [strategy, long_book, short_book, cum_strategy]
    metrics : dict of performance statistics
    """
    # Align on common dates and tickers
    common_dates   = weights_matrix.index.intersection(returns.index)
    common_tickers = [t for t in weights_matrix.columns if t in returns.columns]

    w = weights_matrix.loc[common_dates, common_tickers]
    r = returns.loc[common_dates, common_tickers]

    # Daily strategy return = sum(weight * return)
    strategy_ret  = (w * r).sum(axis=1)
    long_ret  = (w.clip(lower=0) * r).sum(axis=1)
    short_ret = (w.clip(upper=0) * r).sum(axis=1)

    cum = np.exp(strategy_ret.cumsum()) - 1  # convert log returns to cumulative

    pnl = pd.DataFrame({
        "strategy":     strategy_ret,
        "long_book":    long_ret,
        "short_book":   short_ret,
        "cum_strategy": cum,
    })

    # ── Attribution: stress vs. calm periods ──────────────────────────────────
    stress_mask = event_date_mask(pd.DatetimeIndex(common_dates), events_df)
    stress_ret  = strategy_ret[stress_mask]
    calm_ret    = strategy_ret[~stress_mask]

    metrics = {
        **_compute_metrics(strategy_ret, risk_free_rate, label="full"),
        **_compute_metrics(stress_ret,   risk_free_rate, label="stress"),
        **_compute_metrics(calm_ret,     risk_free_rate, label="calm"),
        **_significance_tests(stress_ret, calm_ret),
        "stress_days": int(stress_mask.sum()),
        "total_days":  len(common_dates),
    }

    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(common_dates).fillna(0)
        metrics.update(_compute_metrics(bench, risk_free_rate, label="bench"))

    if ew_returns is not None:
        ew = ew_returns.reindex(common_dates).fillna(0)
        metrics.update(_compute_metrics(ew, risk_free_rate, label="ew"))

    return pnl, metrics


def _compute_metrics(ret: pd.Series, rf_annual: float, label: str) -> dict:
    """Compute annualised performance stats for a return series."""
    if ret.empty or ret.std() == 0:
        return {f"{label}_sharpe": np.nan, f"{label}_sortino": np.nan,
                f"{label}_ann_return": np.nan, f"{label}_max_dd": np.nan,
                f"{label}_hit_rate": np.nan}

    ann_factor = 252
    rf_daily   = rf_annual / ann_factor

    excess      = ret - rf_daily
    ann_ret     = ret.mean() * ann_factor
    ann_vol     = ret.std() * np.sqrt(ann_factor)
    sharpe      = excess.mean() / excess.std() * np.sqrt(ann_factor) if excess.std() > 0 else np.nan

    downside    = ret[ret < 0].std() * np.sqrt(ann_factor)
    sortino     = (ann_ret - rf_annual) / downside if downside > 0 else np.nan

    cum         = np.exp(ret.cumsum())
    rolling_max = cum.cummax()
    drawdown    = (cum / rolling_max) - 1
    max_dd      = drawdown.min()

    hit_rate    = (ret > 0).mean()

    return {
        f"{label}_ann_return": round(ann_ret, 4),
        f"{label}_ann_vol":    round(ann_vol, 4),
        f"{label}_sharpe":     round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        f"{label}_sortino":    round(sortino, 3) if not np.isnan(sortino) else np.nan,
        f"{label}_max_dd":     round(max_dd, 4),
        f"{label}_hit_rate":   round(hit_rate, 3),
    }


def _significance_tests(stress_ret: pd.Series, calm_ret: pd.Series) -> dict:
    """
    Compare stress-period vs calm-period daily returns with three tests:
      - Welch t-test       : mean daily return different? (unequal-variance)
      - Levene test        : variance (volatility) different?
      - Mann-Whitney U     : non-parametric, distribution-free comparison

    Returns p-values and a star annotation (* p<0.10, ** p<0.05, *** p<0.01).
    """
    out: dict = {}
    if len(stress_ret) < 5 or len(calm_ret) < 5:
        for k in ("sig_mean_pval", "sig_vol_pval", "sig_mw_pval"):
            out[k] = np.nan
        return out

    s = stress_ret.dropna().values
    c = calm_ret.dropna().values

    _, mean_p = stats.ttest_ind(s, c, equal_var=False)
    _, vol_p  = stats.levene(s, c)
    _, mw_p   = stats.mannwhitneyu(s, c, alternative="two-sided")

    out["sig_mean_pval"] = round(float(mean_p), 4)
    out["sig_vol_pval"]  = round(float(vol_p),  4)
    out["sig_mw_pval"]   = round(float(mw_p),   4)
    return out


def _stars(p: float) -> str:
    if np.isnan(p):  return "   "
    if p < 0.01:     return "***"
    if p < 0.05:     return " **"
    if p < 0.10:     return "  *"
    return "   "


# ── Information Coefficient ───────────────────────────────────────────────────

def compute_ic(
    factor_scores: pd.DataFrame,
    returns: pd.DataFrame,
    horizons: list[int] | None = None,
    gsi_by_iso: dict | None = None,
    gsi_threshold: float = 0.5,
) -> tuple[dict[int, pd.Series], dict]:
    """
    Cross-sectional IC: Pearson correlation between factor score and N-day
    forward cumulative return, computed at each rebalance date.

    Parameters
    ----------
    factor_scores  : DataFrame with columns [date, ticker, factor_score]
    returns        : (date × ticker) daily log-return DataFrame
    horizons       : forward-return windows in trading days (default [5, 10, 21, 63])
    gsi_by_iso     : {iso: gsi_df} from build_multi_iso_gsi — enables conditional IC
    gsi_threshold  : GSI level separating stressed from calm regimes (default 0.5)

    Returns
    -------
    ic_series : {horizon: pd.Series(date → IC)}
    ic_metrics: {horizon: {mean_ic, std_ic, t_stat, pct_pos, n_periods,
                            stressed_mean_ic, stressed_t_stat, stressed_n,
                            calm_mean_ic, calm_t_stat, calm_n}}
                plus ic_metrics["stress_dates"] = set of stressed rebalance dates
    """
    if horizons is None:
        horizons = [5, 10, 21, 63]

    ic_raw: dict[int, dict] = {h: {} for h in horizons}

    # Build max-GSI series across all ISOs (stressed if ANY ISO is stressed)
    max_gsi: pd.Series | None = None
    if gsi_by_iso:
        frames = [df["gsi"] for df in gsi_by_iso.values()
                  if isinstance(df, pd.DataFrame) and "gsi" in df.columns and not df.empty]
        if frames:
            max_gsi = pd.concat(frames, axis=1).max(axis=1).sort_index()

    date_gsi: dict = {}

    for date_val in sorted(factor_scores["date"].unique()):
        scores_at = (
            factor_scores[factor_scores["date"] == date_val]
            .set_index("ticker")["factor_score"]
        )
        tickers = [t for t in scores_at.index if t in returns.columns]
        if len(tickers) < 3:
            continue

        date_ts = pd.Timestamp(date_val)
        # first trading day strictly after the score date
        idx = returns.index.searchsorted(date_ts, side="right")

        # GSI level: average over the 21 trading days ending on the score date
        if max_gsi is not None:
            pos = max_gsi.index.searchsorted(date_ts, side="right")
            window = max_gsi.iloc[max(0, pos - 21): pos]
            if not window.empty:
                date_gsi[date_ts] = float(window.mean())

        for h in horizons:
            end_idx = idx + h
            if end_idx > len(returns):
                continue
            fwd_ret = returns.iloc[idx:end_idx][tickers].sum()
            aligned = pd.DataFrame(
                {"score": scores_at[tickers], "fwd_ret": fwd_ret}
            ).dropna()
            if len(aligned) < 3:
                continue
            ic_raw[h][date_ts] = aligned["score"].corr(aligned["fwd_ret"])

    ic_series = {h: pd.Series(v).sort_index() for h, v in ic_raw.items()}
    print(f"[ic] dates: {len(sorted(factor_scores['date'].unique()))}  |  "
          + "  |  ".join(f"{h}d: {len(s)} pts" for h, s in ic_series.items()))

    gsi_at_dates = pd.Series(date_gsi).sort_index() if date_gsi else pd.Series(dtype=float)
    stress_dates: set = set(gsi_at_dates.index[gsi_at_dates >= gsi_threshold]) if not gsi_at_dates.empty else set()

    ic_metrics: dict = {}
    for h, series in ic_series.items():
        if series.empty:
            continue
        mean_ic = series.mean()
        std_ic  = series.std()
        n       = len(series)
        t_stat  = mean_ic / std_ic * np.sqrt(n) if std_ic > 0 else np.nan
        entry: dict = {
            "mean_ic":   round(float(mean_ic), 4),
            "std_ic":    round(float(std_ic),  4),
            "t_stat":    round(float(t_stat),  3) if not np.isnan(t_stat) else np.nan,
            "pct_pos":   round(float((series > 0).mean()), 3),
            "n_periods": n,
        }

        # Conditional IC
        if stress_dates:
            for regime, mask in [
                ("stressed", series.index.isin(stress_dates)),
                ("calm",     ~series.index.isin(stress_dates)),
            ]:
                sub = series[mask]
                if len(sub) >= 3:
                    m = float(sub.mean())
                    s = float(sub.std())
                    t = m / s * np.sqrt(len(sub)) if s > 0 else np.nan
                    entry[f"{regime}_mean_ic"] = round(m, 4)
                    entry[f"{regime}_t_stat"]  = round(t, 3) if not np.isnan(t) else np.nan
                    entry[f"{regime}_n"]       = len(sub)
                else:
                    entry[f"{regime}_mean_ic"] = np.nan
                    entry[f"{regime}_t_stat"]  = np.nan
                    entry[f"{regime}_n"]       = len(sub)

        ic_metrics[h] = entry

    if stress_dates:
        ic_metrics["stress_dates"] = stress_dates

    return ic_series, ic_metrics


# ── Performance charting ──────────────────────────────────────────────────────

def plot_performance(
    pnl: pd.DataFrame,
    events_df: pd.DataFrame,
    metrics: dict | None = None,
    benchmark_returns: pd.Series | None = None,
    ew_returns: pd.Series | None = None,
    ic_series: dict | None = None,
    ic_metrics: dict | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot cumulative strategy P&L with stress event shading.

    Three-panel layout (four if IC data provided):
      Top    : cumulative return of strategy vs. long book vs. short book
      Middle : rolling 63-day Sharpe (strategy + benchmark)
      3rd    : drawdown
      Bottom : cross-sectional IC (10d forward returns) — if ic_series provided
    """
    has_ic = ic_series is not None and any(
        h in ic_series and not ic_series[h].empty for h in [21, 10]
    )
    n_panels      = 4 if has_ic else 3
    height_ratios = [3, 1.5, 1.5, 1.5][:n_panels]
    fig_height    = 13 if has_ic else 10

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, fig_height), sharex=True,
                              gridspec_kw={"height_ratios": height_ratios})
    fig.suptitle("Grid Resilience Strategy — Backtest Performance", fontsize=14, fontweight="bold")

    dates = pnl.index

    # ── Panel 1: Cumulative return ─────────────────────────────────────────────
    ax = axes[0]
    cum_strat = np.exp(pnl["strategy"].cumsum()) - 1
    cum_long  = np.exp(pnl["long_book"].cumsum()) - 1
    cum_short = np.exp(pnl["short_book"].cumsum()) - 1

    ax.plot(dates, cum_strat * 100, color="#1a3a5c", linewidth=2.0, label="Strategy (L/S)")
    ax.plot(dates, cum_long  * 100, color="#2e8b57", linewidth=1.2, linestyle="--", label="Long book")
    ax.plot(dates, cum_short * 100, color="#8b0000", linewidth=1.2, linestyle="--", label="Short book")
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(dates).fillna(0)
        cum_bench = np.exp(bench.cumsum()) - 1
        ax.plot(dates, cum_bench * 100, color="#888888", linewidth=1.0, linestyle=":", label="Long XLU")
    if ew_returns is not None:
        ew = ew_returns.reindex(dates).fillna(0)
        cum_ew = np.exp(ew.cumsum()) - 1
        ax.plot(dates, cum_ew * 100, color="#cc8800", linewidth=1.0, linestyle="-.", label="EW Universe")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=9)

    _shade_events(ax, dates, events_df)

    if metrics:
        bench_sharpe = metrics.get("bench_sharpe")
        bench_str = f"  |  Bench Sharpe: {bench_sharpe}" if bench_sharpe is not None else ""
        info = (f"Sharpe: {metrics.get('full_sharpe', '—')}  |  "
                f"Ann. Ret: {metrics.get('full_ann_return', 0)*100:.1f}%  |  "
                f"Max DD: {metrics.get('full_max_dd', 0)*100:.1f}%"
                + bench_str)
        ax.set_title(info, fontsize=9, color="#555555", loc="right")

    # ── Panel 2: Rolling Sharpe ────────────────────────────────────────────────
    ax2 = axes[1]
    roll_sharpe = (
        pnl["strategy"].rolling(63).mean()
        / pnl["strategy"].rolling(63).std()
        * np.sqrt(252)
    )
    ax2.plot(dates, roll_sharpe, color="#1a3a5c", linewidth=1.0, label="Strategy")
    if benchmark_returns is not None:
        bench_aligned = benchmark_returns.reindex(dates).fillna(0)
        roll_bench = (
            bench_aligned.rolling(63).mean()
            / bench_aligned.rolling(63).std()
            * np.sqrt(252)
        )
        ax2.plot(dates, roll_bench, color="#888888", linewidth=0.8,
                 linestyle=":", label="Benchmark (XLU)")
        ax2.legend(loc="upper left", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(1, color="grey", linewidth=0.5, linestyle=":")
    ax2.set_ylabel("Rolling Sharpe\n(63d)", fontsize=9)
    _shade_events(ax2, dates, events_df)

    # ── Panel 3: Drawdown ──────────────────────────────────────────────────────
    ax3 = axes[2]
    cum = np.exp(pnl["strategy"].cumsum())
    dd  = (cum / cum.cummax()) - 1
    ax3.fill_between(dates, dd * 100, 0, color="#8b0000", alpha=0.4)
    ax3.set_ylabel("Drawdown (%)", fontsize=9)
    ax3.set_xlabel("Date")
    _shade_events(ax3, dates, events_df)

    # ── Panel 4: Cross-sectional IC (21d) ────────────────────────────────────
    ic_horizon = 21 if (ic_series and 21 in ic_series and not ic_series[21].empty) else 10
    has_ic_panel = ic_series is not None and ic_horizon in ic_series and not ic_series[ic_horizon].empty
    if has_ic_panel:
        ax4 = axes[3]
        ic_s = ic_series[ic_horizon]
        stress_dates = (ic_metrics or {}).get("stress_dates", set())
        bar_colors = []
        for d, v in zip(ic_s.index, ic_s.values):
            is_stressed = d in stress_dates
            if is_stressed:
                bar_colors.append("#cc4400")   # dark orange = stressed (pos & neg)
            else:
                bar_colors.append("#2e8b57")   # green = calm (pos & neg)
        alphas = [0.75 if v >= 0 else 0.45 for v in ic_s.values]
        for xi, (d, v, c, a) in enumerate(zip(ic_s.index, ic_s.values, bar_colors, alphas)):
            ax4.bar(d, v, color=c, alpha=a, width=20)
        ax4.axhline(0, color="black", linewidth=0.5)
        mean_ic = ic_s.mean()
        ax4.axhline(mean_ic, color="#1a3a5c", linewidth=1.2, linestyle="--",
                    label=f"Mean IC: {mean_ic:.3f}")
        if stress_dates:
            legend_patches = [
                mpatches.Patch(color="#cc4400", alpha=0.75, label="Stressed regime (+)"),
                mpatches.Patch(color="#cc4400", alpha=0.45, label="Stressed regime (−)"),
                mpatches.Patch(color="#2e8b57", alpha=0.75, label="Calm regime (+)"),
                mpatches.Patch(color="#2e8b57", alpha=0.45, label="Calm regime (−)"),
            ]
            ax4.legend(handles=legend_patches + [
                plt.Line2D([0], [0], color="#1a3a5c", linestyle="--",
                           label=f"Mean IC: {mean_ic:.3f}")
            ], loc="upper right", fontsize=8)
        else:
            ax4.legend(loc="upper right", fontsize=8)
        ax4.set_ylabel(f"IC ({ic_horizon}d fwd)", fontsize=9)
        ax4.set_xlabel("Date")
        _shade_events(ax4, dates, events_df)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[backtest] Saved chart to {save_path}")

    plt.show()


def print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dict to stdout."""
    print("\n" + "=" * 55)
    print("  GRID RESILIENCE STRATEGY — PERFORMANCE SUMMARY")
    print("=" * 55)
    labels = [
        ("full",   "Full Period"),
        ("bench",  "Benchmark (XLU)"),
        ("ew",     "EW Universe"),
        ("stress", "Stress Periods"),
        ("calm",   "Calm Periods"),
    ]
    for key, label in labels:
        print(f"\n  {label}:")
        for stat in ["ann_return", "ann_vol", "sharpe", "sortino", "max_dd", "hit_rate"]:
            val = metrics.get(f"{key}_{stat}", "—")
            if isinstance(val, float) and not np.isnan(val):
                if stat in ("ann_return", "ann_vol", "max_dd"):
                    print(f"    {stat:<15}: {val*100:>7.2f}%")
                elif stat == "hit_rate":
                    print(f"    {stat:<15}: {val*100:>7.1f}%")
                else:
                    print(f"    {stat:<15}: {val:>7.3f}")
            else:
                print(f"    {stat:<15}:      —")
    stress_days = metrics.get("stress_days", 0)
    total_days  = metrics.get("total_days", 0)
    print(f"\n  Stress days: {stress_days} / {total_days} ({stress_days/max(total_days,1)*100:.1f}%)")

    # ── Significance tests: stress vs calm ────────────────────────────────────
    mean_p = metrics.get("sig_mean_pval", np.nan)
    vol_p  = metrics.get("sig_vol_pval",  np.nan)
    mw_p   = metrics.get("sig_mw_pval",   np.nan)
    print(f"\n  Stress vs Calm — significance tests:")
    print(f"    {'Test':<26}  {'p-value':>8}  Sig")
    print(f"    {'-'*26}  {'-'*8}  ---")

    def _fmt(p):
        return f"{p:8.4f}" if not np.isnan(p) else "      —"

    print(f"    {'Welch t (mean return)':<26}  {_fmt(mean_p)}  {_stars(mean_p)}")
    print(f"    {'Levene  (volatility)':<26}  {_fmt(vol_p)}   {_stars(vol_p)}")
    print(f"    {'Mann-Whitney U (distrib)':<26}  {_fmt(mw_p)}   {_stars(mw_p)}")
    print(f"    Significance: * p<0.10  ** p<0.05  *** p<0.01")
    print("=" * 55 + "\n")


def print_ic_metrics(ic_metrics: dict, perf_metrics: dict | None = None) -> None:
    """
    Pretty-print IC summary (overall + conditional) and a signal comparison
    table showing IC vs Sharpe for the factor, EW basket, and benchmark.
    """
    horizon_keys = sorted(k for k in ic_metrics if isinstance(k, int))
    if not horizon_keys:
        return

    def _t(v) -> str:
        return f"{v:8.3f}" if isinstance(v, float) and not np.isnan(v) else "      —"
    def _ic(v) -> str:
        return f"{v:8.4f}" if isinstance(v, float) and not np.isnan(v) else "      —"
    def _pct(v) -> str:
        return f"{v*100:7.2f}%" if isinstance(v, float) and not np.isnan(v) else "      —"

    print("\n" + "=" * 62)
    print("  FACTOR SCORE IC (vs. Forward Returns)")
    print("=" * 62)
    print(f"  {'Horizon':<10} {'Mean IC':>8} {'Std IC':>8} {'t-stat':>8} {'% Pos':>7} {'N':>5}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*5}")
    for h in horizon_keys:
        m = ic_metrics[h]
        print(f"  {f'{h}d':<10} {_ic(m['mean_ic'])} {_ic(m['std_ic'])}"
              f" {_t(m['t_stat'])} {m['pct_pos']:>7.1%} {m['n_periods']:>5}")

    has_cond = any("stressed_mean_ic" in ic_metrics[h] for h in horizon_keys)
    if has_cond:
        print(f"\n  Conditional IC by GSI regime:")
        print(f"  {'Horizon':<10} {'Regime':<10} {'Mean IC':>8} {'t-stat':>8} {'N':>5}")
        print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*5}")
        for h in horizon_keys:
            m = ic_metrics[h]
            for regime in ("stressed", "calm"):
                key_ic = f"{regime}_mean_ic"
                key_t  = f"{regime}_t_stat"
                key_n  = f"{regime}_n"
                if key_ic in m:
                    print(f"  {f'{h}d':<10} {regime:<10} {_ic(m[key_ic])} {_t(m[key_t])} {m[key_n]:>5}")

    # ── Signal comparison: IC + Sharpe + Ann Return across signals ────────────
    if perf_metrics:
        cmp_h = next((h for h in [21, 10, 63, 5] if h in ic_metrics), None)
        print(f"\n  Signal comparison (IC @ {cmp_h}d fwd  |  EW/Bench IC = 0 by construction):")
        print(f"  {'Signal':<22} {'IC':>8} {'t-stat':>8} {'Sharpe':>8} {'Ann Ret':>9} {'Max DD':>8}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
        rows = [
            ("Factor (L/S)",   ic_metrics.get(cmp_h, {}), "full"),
            ("Benchmark (XLU)", None,                      "bench"),
            ("EW Universe",    None,                       "ew"),
        ]
        for label, ic_m, pfx in rows:
            if ic_m is not None:
                ic_val = _ic(ic_m.get("mean_ic"))
                t_val  = _t(ic_m.get("t_stat"))
            else:
                ic_val, t_val = "   0.0000", "      —"
            sharpe  = perf_metrics.get(f"{pfx}_sharpe",     np.nan)
            ann_ret = perf_metrics.get(f"{pfx}_ann_return", np.nan)
            max_dd  = perf_metrics.get(f"{pfx}_max_dd",     np.nan)
            s_str  = f"{sharpe:8.3f}"  if isinstance(sharpe,  float) and not np.isnan(sharpe)  else "      —"
            r_str  = _pct(ann_ret)
            d_str  = _pct(max_dd)
            print(f"  {label:<22} {ic_val} {t_val} {s_str} {r_str} {d_str}")

    print("=" * 62 + "\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shade_events(
    ax,
    dates: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    alpha: float = 0.12,
) -> None:
    """Shade stress event windows on a matplotlib axis."""
    for _, ev in events_df.iterrows():
        ax.axvspan(ev["window_start"], ev["window_end"],
                   alpha=alpha, color="#cc8800", zorder=0)
