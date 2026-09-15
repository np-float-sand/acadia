"""Adapter for Client-provided (Addenda) risk-factor exposures (spec 3.3).

No real file exists yet -- ships against a placeholder schema (long format:
date, ticker, factor, exposure) with a documented swap-in point. See
electrification_strategy/data/risk_exposures_sample.csv for the schema
contract and docs/technical-reference.md for what's needed from the Client.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from electrification_strategy import config

_COLUMNS = ["date", "ticker", "factor", "exposure"]
DEFAULT_PATH = config.DATA_DIR / "risk_exposures.csv"


def load_risk_exposures(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_PATH
    if not Path(path).exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(path, comment="#")
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    return df[_COLUMNS]


def align_to_universe(
    exposures: pd.DataFrame, tickers: list[str], asof: pd.Timestamp,
    lag_days: int = config.REBALANCE_LAG_DAYS,
) -> pd.DataFrame:
    effective_date = asof - pd.Timedelta(days=lag_days)
    eligible = exposures[exposures["date"] <= effective_date]
    if eligible.empty:
        factors = sorted(exposures["factor"].unique()) if not exposures.empty else []
        return pd.DataFrame(index=tickers, columns=factors, dtype=float)
    latest = eligible.sort_values("date").groupby(["ticker", "factor"]).last()["exposure"]
    wide = latest.unstack("factor")
    return wide.reindex(tickers)


def portfolio_exposure_report(
    exposures: pd.DataFrame, weights: pd.Series, asof: pd.Timestamp,
    lag_days: int = config.REBALANCE_LAG_DAYS,
) -> pd.Series:
    aligned = align_to_universe(exposures, list(weights.index), asof, lag_days)
    w = weights.reindex(aligned.index).fillna(0.0)
    return aligned.mul(w, axis=0).sum(axis=0)
