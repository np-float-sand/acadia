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

from grid_resilience.signals.stress_events import event_date_mask


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    weights_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    events_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
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
        "stress_days": int(stress_mask.sum()),
        "total_days":  len(common_dates),
    }

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


# ── Performance charting ──────────────────────────────────────────────────────

def plot_performance(
    pnl: pd.DataFrame,
    events_df: pd.DataFrame,
    metrics: dict | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot cumulative strategy P&L with stress event shading.

    Three-panel layout:
      Top    : cumulative return of strategy vs. long book vs. short book
      Middle : rolling 63-day Sharpe
      Bottom : drawdown
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
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
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=9)

    _shade_events(ax, dates, events_df)

    if metrics:
        info = (f"Sharpe: {metrics.get('full_sharpe', '—')}  |  "
                f"Ann. Ret: {metrics.get('full_ann_return', 0)*100:.1f}%  |  "
                f"Max DD: {metrics.get('full_max_dd', 0)*100:.1f}%")
        ax.set_title(info, fontsize=9, color="#555555", loc="right")

    # ── Panel 2: Rolling Sharpe ────────────────────────────────────────────────
    ax2 = axes[1]
    roll_sharpe = (
        pnl["strategy"].rolling(63).mean()
        / pnl["strategy"].rolling(63).std()
        * np.sqrt(252)
    )
    ax2.plot(dates, roll_sharpe, color="#1a3a5c", linewidth=1.0)
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
    labels = [("full", "Full Period"), ("stress", "Stress Periods"), ("calm", "Calm Periods")]
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
    print("=" * 55 + "\n")


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
