from __future__ import annotations

"""
Unified grid data fetcher using the `gridstatus` library.

Covers ERCOT, PJM, MISO, CAISO, and SPP with a consistent interface.
Results are cached to parquet files keyed by (iso, dataset, start, end).

Data sources (all publicly available, no key required for basic access):
  ERCOT  : https://api.ercot.com  (B2C token auth — set ERCOT_PASSWORD + ERCOT_SUBSCRIPTION_KEY)
             Registration: developer.ercot.com
  PJM    : https://dataminer2.pjm.com  (free PJM account for bulk pulls)
  MISO   : https://api.misoenergy.org  (public, no registration)
  CAISO  : http://oasis.caiso.com  (public, no registration)
  SPP    : https://marketplace.spp.org  (free registration)

The gridstatus library wraps all of the above:
  pip install gridstatus

For historical data beyond what the live API returns, bulk archive files
can be downloaded from each ISO's market data portal and loaded via
`load_lmp_from_csv()` below.
"""

import os
import hashlib
from pathlib import Path

import pandas as pd

from grid_resilience.config import (
    CACHE_DIR,
    ISO_CLASS_MAP,
    ISO_LOCATION_TYPE,
    ISO_BENCHMARK_NODES,
    LMP_SPIKE_THRESHOLD,
)

# Lazily instantiated gridstatus ISO objects (one per ISO, shared)
_ISO_INSTANCES: dict = {}


def _get_iso(iso: str):
    if iso not in _ISO_INSTANCES:
        try:
            import gridstatus
        except ImportError:
            raise ImportError("gridstatus is not installed. Run: pip install gridstatus")
        cls_name = ISO_CLASS_MAP.get(iso)
        if not cls_name:
            raise ValueError(f"Unsupported ISO: {iso}. Choose from {list(ISO_CLASS_MAP)}")

        if iso == "ERCOT":
            # B2C token auth — see ercot_auth.py
            _inject_ercot_token()
            _ISO_INSTANCES[iso] = getattr(gridstatus, cls_name)()
        elif iso == "PJM":
            # PJM requires a free API key: register at pjm.com/api
            # Set env var: PJM_API_KEY=your_key
            api_key = os.environ.get("PJM_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "PJM_API_KEY not set. Register free at pjm.com/api "
                    "then: export PJM_API_KEY=your_key"
                )
            _ISO_INSTANCES[iso] = getattr(gridstatus, cls_name)(api_key=api_key)
        else:
            # MISO, CAISO, SPP, NYISO, ISO-NE — no credentials needed
            _ISO_INSTANCES[iso] = getattr(gridstatus, cls_name)()

    return _ISO_INSTANCES[iso]


def _inject_ercot_token() -> None:
    """
    Fetch a fresh ERCOT B2C id_token and expose it as ERCOT_API_KEY so
    gridstatus can make authenticated ERCOT API calls.
    Falls back silently if credentials are not configured.
    """
    try:
        from grid_resilience.data.ercot_auth import get_id_token
        token = get_id_token()
        os.environ["ERCOT_API_KEY"] = token
    except EnvironmentError as exc:
        print(f"  [grid] ERCOT auth skipped — {exc}")
    except Exception as exc:
        print(f"  [grid] ERCOT token fetch failed: {exc}")


def _cache_path(iso: str, dataset: str, start: str, end: str) -> Path:
    key = hashlib.md5(f"{iso}_{dataset}_{start}_{end}".encode()).hexdigest()[:10]
    return CACHE_DIR / f"{iso.lower()}_{dataset}_{key}.parquet"


def fetch_lmp(
    iso: str,
    start: str,
    end: str,
    location_type: str | None = None,
    locations: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly LMP data for an ISO over a date range.

    Returns a DataFrame with at minimum:
        time        : UTC timestamp
        location    : settlement point / zone / hub name
        lmp         : total LMP ($/MWh)
        energy      : energy component (where available)
        congestion  : congestion component (where available)
        loss        : loss component (where available)

    Parameters
    ----------
    iso           : "ERCOT" | "PJM" | "MISO" | "CAISO" | "SPP"
    start / end   : "YYYY-MM-DD" strings
    location_type : overrides ISO default (e.g. "hub", "zone", "node")
    locations     : filter to these location names after fetching
    use_cache     : read/write parquet cache
    """
    loc_type = location_type or ISO_LOCATION_TYPE.get(iso, "hub")
    cache_file = _cache_path(iso, f"lmp_{loc_type}", start, end)

    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        if locations:
            loc_col = _location_col(df)
            df = df[df[loc_col].isin(locations)]
        return df

    iso_obj = _get_iso(iso)
    print(f"[grid] Fetching {iso} LMP ({loc_type}) {start} → {end}…")

    # SPP uses dedicated DA/RT methods instead of a generic get_lmp()
    if iso == "SPP":
        return _fetch_spp_lmp(iso_obj, start, end, cache_file, use_cache)

    try:
        df = iso_obj.get_lmp(
            date=start,
            end=end,
            location_type=loc_type,
            verbose=False,
        )
    except Exception as exc:
        print(f"  [grid] {iso} LMP fetch failed: {exc}")
        return pd.DataFrame()

    df = _normalise_lmp_columns(df)

    if use_cache and not df.empty:
        df.to_parquet(cache_file, index=False)

    if locations:
        loc_col = _location_col(df)
        df = df[df[loc_col].isin(locations)]

    return df


def fetch_load(
    iso: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly total load (demand) for an ISO.

    Returns DataFrame with columns: time, load_mw
    """
    cache_file = _cache_path(iso, "load", start, end)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    iso_obj = _get_iso(iso)
    print(f"[grid] Fetching {iso} load {start} → {end}…")
    try:
        df = iso_obj.get_load(date=start, end=end, verbose=False)
    except Exception as exc:
        print(f"  [grid] {iso} load fetch failed: {exc}")
        return pd.DataFrame()

    df = _normalise_load_columns(df)

    if use_cache and not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


def fetch_fuel_mix(
    iso: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly generation fuel mix (MW by fuel type) for an ISO.
    Used as input to the reserve-tightness sub-signal.
    """
    cache_file = _cache_path(iso, "fuel_mix", start, end)
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    iso_obj = _get_iso(iso)
    print(f"[grid] Fetching {iso} fuel mix {start} → {end}…")
    try:
        df = iso_obj.get_fuel_mix(date=start, end=end, verbose=False)
    except Exception as exc:
        print(f"  [grid] {iso} fuel mix fetch failed: {exc}")
        return pd.DataFrame()

    if use_cache and not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


# ── SPP LMP helper ────────────────────────────────────────────────────────────

def _fetch_spp_lmp(
    spp_obj,
    start: str,
    end: str,
    cache_file: Path,
    use_cache: bool,
) -> pd.DataFrame:
    """
    SPP does not implement a generic get_lmp().
    Use get_lmp_day_ahead_hourly() as the primary price signal.
    Falls back to get_lmp_real_time_5_min_by_location() if DA isn't available.
    """
    frames = []
    dates = pd.date_range(start, end, freq="D")

    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        try:
            df = spp_obj.get_lmp_day_ahead_hourly(date=ds, verbose=False)
            if not df.empty:
                frames.append(df)
        except Exception:
            try:
                df = spp_obj.get_lmp_real_time_5_min_by_location(date=ds, verbose=False)
                if not df.empty:
                    frames.append(df)
            except Exception:
                pass

    if not frames:
        print("  [grid] SPP LMP: no data retrieved.")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = _normalise_lmp_columns(result)

    if use_cache and not result.empty:
        result.to_parquet(cache_file, index=False)
    return result


# ── Daily aggregate helpers ───────────────────────────────────────────────────

def daily_lmp_summary(lmp_df: pd.DataFrame, iso: str) -> pd.DataFrame:
    """
    Aggregate hourly LMP data to daily statistics used by the stress index.

    Returns DataFrame indexed by date with columns:
        lmp_max          : daily maximum total LMP across benchmark nodes
        lmp_mean         : daily mean total LMP
        spike_hours      : hours where LMP > iso-specific threshold
        congestion_frac  : mean(|congestion| / |lmp|), 0 if not available
    """
    if lmp_df.empty:
        return pd.DataFrame()

    time_col = _time_col(lmp_df)
    lmp_df = lmp_df.copy()
    lmp_df["_date"] = pd.to_datetime(lmp_df[time_col]).dt.date

    threshold = LMP_SPIKE_THRESHOLD.get(iso, 200.0)

    agg: dict = {
        "lmp_max":   ("lmp", "max"),
        "lmp_mean":  ("lmp", "mean"),
        "spike_hours": ("lmp", lambda x: (x > threshold).sum()),
    }

    if "congestion" in lmp_df.columns:
        lmp_df["_cong_frac"] = lmp_df["congestion"].abs() / (lmp_df["lmp"].abs() + 1e-6)
        agg["congestion_frac"] = ("_cong_frac", "mean")

    daily = lmp_df.groupby("_date").agg(**agg)
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"

    if "congestion_frac" not in daily.columns:
        daily["congestion_frac"] = float("nan")

    return daily


def daily_load_summary(load_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly load to daily max — used for the reserve-tightness signal.
    Returns DataFrame indexed by date with column: load_max_mw
    """
    if load_df.empty:
        return pd.DataFrame()

    time_col = _time_col(load_df)
    load_df = load_df.copy()
    load_df["_date"] = pd.to_datetime(load_df[time_col]).dt.date

    mw_col = next((c for c in load_df.columns if "load" in c.lower() or "mw" in c.lower()), None)
    if mw_col is None:
        return pd.DataFrame()

    daily = load_df.groupby("_date")[mw_col].max().rename("load_max_mw")
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    return daily.to_frame()


# ── CSV loader for bulk historical downloads ──────────────────────────────────

def load_lmp_from_csv(path: str | Path, iso: str) -> pd.DataFrame:
    """
    Load LMP data from a manually downloaded CSV/ZIP from an ISO data portal.
    Normalises columns to the standard schema expected by daily_lmp_summary().

    Supported format hints are detected from the filename or column names.
    """
    path = Path(path)
    print(f"[grid] Loading {iso} LMP from {path.name}…")
    df = pd.read_csv(path, low_memory=False)
    df = _normalise_lmp_columns(df)
    return df


# ── Internal column normalisation ─────────────────────────────────────────────

def _normalise_lmp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common ISO-specific column names to standard schema."""
    rename: dict[str, str] = {}

    # Time column
    for c in df.columns:
        cl = c.lower()
        if cl in ("time", "datetime", "interval_ending", "delivery_date",
                  "operatingday", "mktintervalstart", "timestamp"):
            rename[c] = "time"
            break

    # Location column
    for c in df.columns:
        cl = c.lower()
        if cl in ("location", "settlement_point", "pnode_name", "node",
                  "locationname", "pricingnode", "busname"):
            rename[c] = "location"
            break

    # LMP / price column
    for c in df.columns:
        cl = c.lower()
        if cl in ("lmp", "price", "settlementpointprice", "lmp_price",
                  "da_lmp", "rt_lmp", "lmpprice"):
            rename[c] = "lmp"
            break

    # Congestion component
    for c in df.columns:
        cl = c.lower()
        if "congestion" in cl:
            rename[c] = "congestion"
            break

    # Energy component
    for c in df.columns:
        cl = c.lower()
        if cl in ("energy", "lmpenergy", "energy_component"):
            rename[c] = "energy"
            break

    # Loss component
    for c in df.columns:
        cl = c.lower()
        if cl in ("loss", "lmploss", "loss_component"):
            rename[c] = "loss"
            break

    return df.rename(columns=rename)


def _normalise_load_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("time", "datetime", "interval_ending", "timestamp"):
            rename[c] = "time"
            break
    for c in df.columns:
        cl = c.lower()
        if "load" in cl and "mw" in cl:
            rename[c] = "load_mw"
            break
        elif cl in ("load", "demand", "actual_load"):
            rename[c] = "load_mw"
            break
    return df.rename(columns=rename)


def _location_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in ("location", "settlement_point", "node"):
            return c
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]


def _time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in ("time", "datetime", "interval_ending", "timestamp"):
            return c
    return df.columns[0]
