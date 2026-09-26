"""Portfolio construction and evaluation for the capex-intensity rule."""
from __future__ import annotations

import numpy as np
import pandas as pd

from grid_equipment_basket.backtest import compute_metrics

from capex_intensity_strategy import config


def target_weights(score: pd.DataFrame, mask: pd.DataFrame,
                   quantile: float | None = None,
                   short_weight: float | None = None) -> pd.DataFrame:
    """Monthly target weights: long the lightest quantile, short the heaviest.

    The long leg is fully invested and the short leg is scaled by
    `short_weight`, so the book runs net long by design -- the short exists to
    fund and hedge, not to be dollar-neutral.
    """
    quantile = config.QUANTILE if quantile is None else quantile
    short_weight = config.SHORT_WEIGHT if short_weight is None else short_weight
    eligible = score.where(mask.reindex_like(score).fillna(False))
    # Select by COUNT, not by percentile. Percentile ranking with `average` tie
    # handling can leave a bucket empty when many names share a score -- eight
    # tied names in a 13-name pool all land at pct 0.35 and none clear a 0.25
    # cut. Counting guarantees both legs are populated whenever the pool has at
    # least two scored names. `method="first"` makes the ordering deterministic.
    ranks = eligible.rank(axis=1, method="first")
    n_scored = eligible.notna().sum(axis=1)
    n_pick = np.maximum(1, np.round(n_scored * quantile)).where(n_scored >= 2, 0)
    longs = ranks.le(n_pick, axis=0) & eligible.notna()
    shorts = ranks.gt(n_scored - n_pick, axis=0) & eligible.notna()
    long_w = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    short_w = shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return long_w - short_w * short_weight


def run(weights: pd.DataFrame, returns: pd.DataFrame,
        cost: float | None = None, vol_target: float | None = None) -> pd.Series:
    """Daily net returns from monthly target weights.

    Weights are held between rebalances, applied with a one-day lag, charged on
    traded notional, then scaled to a trailing volatility target. The leverage
    change is itself charged, so vol-targeting is not free.

    `cost` and `vol_target` both take None to mean "use the config default".
    To switch either OFF, pass 0 -- `vol_target=0` returns the unscaled book.
    """
    cost = config.COST_PER_TURNOVER if cost is None else cost
    vol_target = config.VOL_TARGET if vol_target is None else vol_target
    cols = [c for c in weights.columns if c in returns.columns]
    held = weights[cols].reindex(returns.index).ffill().fillna(0.0)
    gross = (held.shift(1) * returns[cols]).sum(axis=1)
    turnover = (held - held.shift(1)).abs().sum(axis=1)
    net = gross - turnover * cost
    if not vol_target:
        return net
    realised = net.rolling(config.VOL_LOOKBACK_DAYS).std() * np.sqrt(config.ANN)
    leverage = (vol_target / realised).clip(upper=config.MAX_LEVERAGE).shift(1)
    return net * leverage - leverage.diff().abs().fillna(0.0) * cost


def diagnostics(weights: pd.DataFrame, returns: pd.DataFrame) -> dict[str, float]:
    """Breadth and trading intensity -- the things that decide if it is real."""
    cols = [c for c in weights.columns if c in returns.columns]
    held = weights[cols].reindex(returns.index).ffill().fillna(0.0)
    turnover = (held - held.shift(1)).abs().sum(axis=1)
    years = max(len(held) / config.ANN, 1e-9)
    def avg_count(counts: pd.Series) -> float:
        # Months with no position at all (before the first scored date) are not
        # evidence of low breadth, so they are excluded -- but a genuinely
        # long-only book reports 0 shorts rather than NaN.
        active = counts.replace(0, np.nan).mean()
        return 0.0 if pd.isna(active) else float(active)

    return {
        "avg_long_names": avg_count((weights[cols] > 0).sum(axis=1)),
        "avg_short_names": avg_count((weights[cols] < 0).sum(axis=1)),
        "annual_turnover": float(turnover.sum() / years),
    }


def summarise(net: pd.Series, windows: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Headline metrics per window, using the repo-standard metric function."""
    windows = config.WINDOWS if windows is None else windows
    rows = []
    for label, (start, end) in windows.items():
        sliced = net.loc[start:end].dropna()
        if len(sliced) < 60:
            continue
        m = compute_metrics(sliced, config.RF, config.ANN)
        max_dd = m.get("max_dd", np.nan)
        rows.append({
            "window": label,
            "cagr": m.get("cagr", np.nan),
            "ann_vol": m.get("ann_vol", np.nan),
            "sharpe": m.get("sharpe", np.nan),
            "max_dd": max_dd,
            "calmar": m.get("cagr", np.nan) / abs(max_dd) if max_dd else np.nan,
            "n_obs": m.get("n_obs", 0),
        })
    return pd.DataFrame(rows)


def format_summary(summary: pd.DataFrame, title: str) -> str:
    lines = [title, "-" * len(title),
             f"{'window':18s} {'CAGR':>8} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}"]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['window']:18s} {r['cagr'] * 100:7.1f}% {r['ann_vol'] * 100:6.1f}% "
            f"{r['sharpe']:7.2f} {r['max_dd'] * 100:7.1f}% {r['calmar']:7.2f}"
        )
    return "\n".join(lines)
