from __future__ import annotations

from pathlib import Path

import pandas as pd

from transmission_rate_base import config

CACHE_DIR = Path(__file__).parent / "cache"


def _read_remote(url: str) -> pd.DataFrame:
    """Isolated network read so tests can monkeypatch it."""
    return pd.read_parquet(url)


def _cache_path(table_name: str) -> Path:
    return CACHE_DIR / f"{table_name}.parquet"


def fetch_table(key: str, *, refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    """Return a PUDL FERC Form 1 table, cached locally as parquet.

    ``key`` is a ``config.FERC_TABLES`` key. Reads the local cache when present
    (unless ``refresh``); otherwise downloads from PUDL's public parquet mirror
    and writes the cache. ``offline=True`` forbids the network: use the cache or
    raise ``FileNotFoundError``.
    """
    table_name = config.FERC_TABLES[key]  # KeyError on unknown key
    path = _cache_path(table_name)

    if path.exists() and not refresh:
        return pd.read_parquet(path)

    if offline:
        raise FileNotFoundError(
            f"offline=True and no cache at {path}; run once online or with --refresh-ferc"
        )

    df = _read_remote(config.PUDL_BASE + table_name + ".parquet")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
