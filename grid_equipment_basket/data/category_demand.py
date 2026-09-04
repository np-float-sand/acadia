"""National equipment-category demand series (FRED / Census M3 / BLS PPI / Fed IP).

Monthly, ~5-week publication lag (M3), pulled from the keyless FRED CSV endpoint
and parquet-cached.

**Status: the maker-rotation signal built on this FAILED** -- scoped in
``docs/handoff_2026-09-03-category-demand-rotation.md``, result in
``docs/category-demand-rotation-results.md``. A category-demand -> per-name tilt
(exposure matrix x category momentum) has ~zero cross-sectional forecast power in
the 2023-26 AI regime (fwd-1m rank-IC t=0.78; fwd-3m IC is 100% from the 2019-22
sub-window) and ~26% of the 9-name book -- 45-70% of GEV/VRT/FLNC -- maps to no
public category series. Keep this module as a *dashboard* series ("is national
grid-equipment demand still expanding?"), not a signal.

Granularity note: the only monthly *volume* detail is NAICS-335 aggregate
(``A34S*``) plus NAICS-3353 industrial production (``IPG3353S``). The
transformer / switchgear / wire split is available **only as PPI (price)**.
There is no public monthly series for data-center power & cooling or grid storage.
"""
from __future__ import annotations

import io
import time
import urllib.request

import pandas as pd

from grid_equipment_basket.config import CACHE_DIR

_CACHE = CACHE_DIR / "category_demand.parquet"
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=1990-01-01"

# series id -> short column name
SERIES: dict[str, str] = {
    # Census M3, Electrical Equipment, Appliances & Components (NAICS 335), $M SA
    "A34SNO": "ee_new_orders",
    "A34SVS": "ee_shipments",
    "A34SUO": "ee_unfilled_orders",
    "A34STI": "ee_inventories",
    # Fed G.17 industrial production, index SA
    "IPG3353S": "ip_naics3353",          # Electrical equipment (transformers+switchgear+motors+relays)
    "IPG335S": "ip_naics335",
    # BLS PPI, index NSA -- the only monthly category split (price, not volume)
    "PCU335311335311": "ppi_transformers",
    "PCU335313335313": "ppi_switchgear",
    "PCU335929335929": "ppi_energy_wire",
    # macro capex context
    "NEWORDER": "core_capex_new_orders",
}


def _fetch_one(sid: str) -> pd.Series:
    raw = urllib.request.urlopen(_FRED_CSV.format(sid=sid), timeout=45).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["date", "val"]
    s = pd.Series(
        pd.to_numeric(df["val"], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["date"]),
        name=SERIES[sid],
    ).dropna()
    s.index = s.index.to_period("M").to_timestamp("M")   # month-end stamp
    return s


def fetch_category_demand(use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
    """Monthly wide frame, one column per :data:`SERIES` value, month-end index.

    Also adds derived diagnostics:
      * ``ee_book_to_bill``      = new orders / shipments
      * ``ee_backlog_months``    = unfilled orders / shipments
      * ``ee_inv_to_ship``       = inventories / shipments
      * ``ppi_tx_vs_sg``         = ppi_transformers / ppi_switchgear (relative scarcity)
    """
    if use_cache and not force_refresh and _CACHE.exists():
        return pd.read_parquet(_CACHE)

    cols = []
    for sid in SERIES:
        for attempt in range(3):
            try:
                cols.append(_fetch_one(sid))
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    df = pd.concat(cols, axis=1).sort_index()

    df["ee_book_to_bill"] = df["ee_new_orders"] / df["ee_shipments"]
    df["ee_backlog_months"] = df["ee_unfilled_orders"] / df["ee_shipments"]
    df["ee_inv_to_ship"] = df["ee_inventories"] / df["ee_shipments"]
    df["ppi_tx_vs_sg"] = df["ppi_transformers"] / df["ppi_switchgear"]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_CACHE)
    return df


def momentum(df: pd.DataFrame, col: str, months: int = 6, *, log: bool = True) -> pd.Series:
    """Trailing `months`-month change of `col` (log-difference by default)."""
    s = df[col].astype(float)
    return (s.apply("log").diff(months) if log else s.pct_change(months)).rename(f"{col}_mom{months}")
