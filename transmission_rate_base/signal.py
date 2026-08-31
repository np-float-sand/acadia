from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from grid_resilience.factor.neutralize import cross_section_zscore

from transmission_rate_base import config

_SEGMENT_MIX_PATH = Path(__file__).parent / "data" / "segment_mix.csv"


def build_parent_panel(net_tx_filer: pd.DataFrame, totals_filer: pd.DataFrame,
                       filer_map: dict[str, list[int]]) -> pd.DataFrame:
    """Sum filer-level transmission plant up to parent tickers.

    Columns: ticker, year, net_tx, net_total, tx_share, n_filers. All reporting
    filers are summed each year (composition drift is handled downstream by the
    structural-break / filer-count guard in primary_signal()).
    """
    id_to_ticker = {fid: tkr for tkr, ids in filer_map.items() for fid in ids}

    tx = net_tx_filer.copy()
    tx["ticker"] = tx["utility_id_ferc1"].map(id_to_ticker)
    tx = tx.dropna(subset=["ticker"])

    tot = totals_filer[["utility_id_ferc1", "report_year", "net_total"]].copy()

    merged = tx.merge(tot, on=["utility_id_ferc1", "report_year"], how="inner")
    agg = (merged.groupby(["ticker", "report_year"])
           .agg(net_tx=("net_tx", "sum"), net_total=("net_total", "sum"),
                n_filers=("utility_id_ferc1", "nunique"))
           .reset_index()
           .rename(columns={"report_year": "year"}))
    agg["tx_share"] = agg["net_tx"] / agg["net_total"]
    return (agg[["ticker", "year", "net_tx", "net_total", "tx_share", "n_filers"]]
            .sort_values(["ticker", "year"]).reset_index(drop=True))


def _one_ticker_components(g: pd.DataFrame) -> pd.DataFrame:
    """One row per year present for this ticker; component values are NaN where
    the 3-yr window is incomplete, has non-positive plant, or a guard trips."""
    g = g.sort_values("year").set_index("year")
    n = config.SIGNAL_CAGR_YEARS
    rows = []
    for y in g.index:
        years = list(range(y - n, y + 1))
        g3 = d3 = np.nan
        if all(yr in g.index for yr in years):
            sub = g.loc[years]
            if not ((sub["net_tx"] <= 0).any() or (sub["net_total"] <= 0).any()):
                steps = np.log(sub["net_tx"].to_numpy()[1:] / sub["net_tx"].to_numpy()[:-1])
                broke = bool(np.any(np.abs(steps) > config.STRUCTURAL_BREAK_LOG))
                broke = broke or (sub["n_filers"].nunique() > 1)  # filer-set change over window
                if not broke:
                    g3 = (sub["net_tx"].iloc[-1] / sub["net_tx"].iloc[0]) ** (1 / n) - 1
                    d3 = sub["tx_share"].iloc[-1] - sub["tx_share"].iloc[0]
        rows.append(dict(year=y, g3_net_tx=g3, d3_tx_share=d3))
    return pd.DataFrame(rows)


def primary_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """Per ticker-year: 3-yr net-transmission-plant CAGR and 3-yr change in
    transmission share of rate base, combined as the mean of their within-year
    cross-sectional percentile ranks. NaN where guards trip."""
    parts = []
    for ticker, g in panel.groupby("ticker"):
        comp = _one_ticker_components(g)
        comp["ticker"] = ticker
        parts.append(comp)
    df = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=["ticker", "year", "g3_net_tx", "d3_tx_share"]))

    df["raw_signal"] = np.nan
    for _, grp in df.groupby("year"):
        ok = grp["g3_net_tx"].notna() & grp["d3_tx_share"].notna()
        if ok.sum() == 0:
            continue
        r1 = grp.loc[ok, "g3_net_tx"].rank(pct=True)
        r2 = grp.loc[ok, "d3_tx_share"].rank(pct=True)
        df.loc[r1.index, "raw_signal"] = (r1 + r2) / 2
    return (df[["ticker", "year", "g3_net_tx", "d3_tx_share", "raw_signal"]]
            .sort_values(["year", "ticker"]).reset_index(drop=True))
