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
