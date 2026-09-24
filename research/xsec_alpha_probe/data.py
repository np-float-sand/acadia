"""Data access for the cross-sectional alpha probe.

Reads existing caches only (no network). Caches are untracked, so when this runs
from a git worktree they are read from the main checkout: set ACADIA_DATA_ROOT to
override. The eligible universe mirrors `capex_intensity_strategy.universe`
(SEC SIC ELEC/POWER candidates + fuel-cycle lexicon names, trailing-1y median
dollar volume > $10m, >= 400 days of history), evaluated at each month-end.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "_cache"
CACHE.mkdir(exist_ok=True)

LIQ_FLOOR = 10e6
LIQ_WINDOW = 252
LIQ_MIN_PERIODS = 120
MIN_HISTORY_DAYS = 400
NON_EQUITY = {"SPY", "GRID", "PAVE", "XLU", "XLI", "VOLT", "BTAL", "GLD", "IEF", "TLT", "TAN", "QQQ"}


@lru_cache
def data_root() -> Path:
    env = os.environ.get("ACADIA_DATA_ROOT")
    if env:
        return Path(env)
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=HERE,
                            capture_output=True, text=True, check=True).stdout.strip()
    return (HERE / common).resolve().parent


def _cis() -> Path:
    return data_root() / "capex_intensity_strategy" / "data" / "cache"


def import_capex_fundamentals():
    """`capex_intensity_strategy.fundamentals` (point-in-time TTM machinery)."""
    root = str(data_root())
    if root not in sys.path:
        sys.path.append(root)
    from capex_intensity_strategy import fundamentals  # noqa: WPS433
    return fundamentals


def prices(control: bool = False) -> pd.DataFrame:
    name = "broad_px.parquet" if control else "prices.parquet"
    px = pd.read_parquet(_cis() / name).sort_index()
    return px


def benchmarks() -> pd.DataFrame:
    px = prices()
    return px[[c for c in ["SPY", "GRID", "PAVE", "XLU", "XLI"] if c in px]]


def dollar_volume(control: bool = False) -> pd.DataFrame:
    name = "broad_dv.parquet" if control else "dollar_volume.parquet"
    return pd.read_parquet(_cis() / name).sort_index()


def candidate_tickers() -> list[str]:
    cand = pd.read_parquet(_cis() / "candidates.parquet")["ticker"].tolist()
    lex_path = _cis() / "lexicon_members.json"
    if lex_path.exists():
        cand += list(json.loads(lex_path.read_text()).keys())
    return sorted(set(cand) - NON_EQUITY)


def universe(control: bool = False) -> pd.DataFrame:
    """Month-end boolean eligibility (date x ticker)."""
    px = prices(control)
    dv = dollar_volume(control)
    tickers = sorted(set(px.columns) - NON_EQUITY) if control else \
        [t for t in candidate_tickers() if t in px.columns]
    tickers = [t for t in tickers if t in dv.columns]
    liq = (dv[tickers].rolling(LIQ_WINDOW, min_periods=LIQ_MIN_PERIODS).median() > LIQ_FLOOR)
    hist = px[tickers].notna().cumsum() >= MIN_HISTORY_DAYS
    live = px[tickers].notna()
    ok = (liq & hist.reindex_like(liq).fillna(False) & live.reindex_like(liq).fillna(False))
    return ok.resample("ME").last().fillna(False)


def facts(control: bool = False) -> pd.DataFrame:
    name = "edgar_facts_broad.parquet" if control else "edgar_facts.parquet"
    return pd.read_parquet(_cis() / name)


def monthly_prices(px: pd.DataFrame) -> pd.DataFrame:
    return px.resample("ME").last()


def iso_cache() -> Path:
    return data_root() / "grid_resilience" / "data" / "cache"
