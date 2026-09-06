"""FRED series fetch with a parquet cache and a committed-CSV offline fallback.

Only DFII10 (10-year TIPS yield) is used in v1. DFII10 publishes next business
day with negligible revisions; the strategy applies a 5-day signal lag on top.
"""
from __future__ import annotations

import io
import warnings

import pandas as pd
import requests

from electrification_strategy import config

_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2000-01-01"


def _http_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def _clean(raw: pd.DataFrame, series_id: str) -> pd.Series:
    raw = raw.copy()
    raw.columns = ["date", "value"][: raw.shape[1]]
    idx = pd.to_datetime(raw["date"])
    val = pd.to_numeric(raw["value"], errors="coerce")  # "." -> NaN
    return pd.Series(val.to_numpy(), index=idx, name=series_id).sort_index().dropna()


def fetch_series(series_id: str, start: str, end: str, use_cache: bool = True) -> pd.Series:
    cache = config.CACHE_DIR / f"fred_{series_id}.parquet"
    if use_cache and cache.exists():
        s = pd.read_parquet(cache).iloc[:, 0]
        s.name = series_id
        return s.loc[start:end]

    try:
        s = _clean(_http_csv(_URL.format(sid=series_id)), series_id)
        if use_cache:
            s.to_frame().to_parquet(cache)
    except Exception as exc:  # noqa: BLE001 - any network/parse failure -> offline file
        fallback = config.DATA_DIR / f"fred_{series_id}.csv"
        warnings.warn(
            f"FRED fetch for {series_id} failed ({exc!r}); using committed {fallback.name}",
            RuntimeWarning, stacklevel=2,
        )
        s = _clean(pd.read_csv(fallback), series_id)
    return s.loc[start:end]
