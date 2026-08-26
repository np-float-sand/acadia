from __future__ import annotations

import pandas as pd


def _zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across this layer's own tickers (its subpopulation),
    never mixed with another layer's raw scale before scoring."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, pd.NA)
    return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def combine_dc_demand_layers(
    pjm_history: pd.DataFrame,
    ercot_history: pd.DataFrame,
    hyperscaler_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the three DC-demand layers into one dates x tickers score.
    Each layer is z-scored within its own covered tickers before merging —
    concatenating raw values across layers before a single z-score is the
    bug already found and fixed once in fill_with_icr() (see spec).
    """
    layers = [df for df in (pjm_history, ercot_history, hyperscaler_history) if not df.empty and not df.columns.empty]
    if not layers:
        return pd.DataFrame()
    zscored = [_zscore_columns(df) for df in layers]
    return pd.concat(zscored, axis=1)
