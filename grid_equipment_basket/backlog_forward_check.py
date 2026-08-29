from __future__ import annotations

"""Exploratory: does as-of backlog growth line up with subsequent stock return?
Descriptive only — informs whether a future cross-sectional factor (spec §8) is
worth a separate spec. Not a backtest, not an IC claim."""

import numpy as np
import pandas as pd

from grid_equipment_basket.backlog_data import backlog_growth_signal


def _asof_growth_series(backlog_df: pd.DataFrame, ticker: str) -> pd.Series:
    g = backlog_df[backlog_df["ticker"] == ticker].sort_values("availability_date")
    out = {}
    for _, row in g.iterrows():
        asof = row["availability_date"]
        sig = backlog_growth_signal(backlog_df[backlog_df["ticker"] == ticker], asof)
        if ticker in sig.index and pd.notna(sig[ticker]):
            out[asof] = float(sig[ticker])
    return pd.Series(out, dtype=float)


def forward_return_table(backlog_df: pd.DataFrame, prices: pd.DataFrame,
                         horizons=(63, 126)) -> pd.DataFrame:
    rows = {}
    for t in sorted(backlog_df["ticker"].unique()):
        if t not in prices.columns:
            continue
        growth = _asof_growth_series(backlog_df, t)
        if len(growth) < 3:
            continue
        px = prices[t].dropna()
        rec: dict = {}
        for h in horizons:
            xs, ys = [], []
            for asof, gval in growth.items():
                pos = px.index.get_indexer([asof], method="bfill")[0]
                if pos == -1 or pos + h >= len(px):
                    continue
                fwd = px.iloc[pos + h] / px.iloc[pos] - 1.0
                xs.append(gval)
                ys.append(fwd)
            rec[f"corr_{h}"] = (float(np.corrcoef(xs, ys)[0, 1])
                                if len(xs) >= 3 else np.nan)
            rec["n"] = len(xs)
        rows[t] = rec
    return pd.DataFrame(rows).T


def plot_backlog_forward(backlog_df, prices, save_path: str, horizon: int = 63) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for t in sorted(backlog_df["ticker"].unique()):
        if t not in prices.columns:
            continue
        growth = _asof_growth_series(backlog_df, t)
        px = prices[t].dropna()
        xs, ys = [], []
        for asof, gval in growth.items():
            pos = px.index.get_indexer([asof], method="bfill")[0]
            if pos == -1 or pos + horizon >= len(px):
                continue
            xs.append(gval)
            ys.append(px.iloc[pos + horizon] / px.iloc[pos] - 1.0)
        ax.scatter(xs, ys, label=t, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("as-of backlog growth (YoY)")
    ax.set_ylabel(f"subsequent {horizon}-day return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
