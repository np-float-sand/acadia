from __future__ import annotations

"""
Unified grid data fetcher using the `gridstatus` library.

Covers ERCOT, PJM, MISO, CAISO, and SPP with a consistent interface.
Results are cached to parquet files keyed by (iso, dataset); gap-filling
fetches only the date ranges not already on disk.

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

import inspect
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from grid_resilience.config import (
    CACHE_DIR,
    ISO_CLASS_MAP,
    ISO_LOCATION_TYPE,
    ISO_BENCHMARK_NODES,
    LMP_SPIKE_THRESHOLD,
)
from grid_resilience.data.cache_utils import (
    read_cached, compute_date_gaps, merge_and_save,
    find_missing_months, read_monthly_cache, save_monthly_chunks, month_bounds,
    chunk_path,
)

# Lazily instantiated gridstatus ISO objects (one per ISO, shared)
_ISO_INSTANCES: dict = {}
_ISO_INIT_LOCK = threading.Lock()  # guards concurrent ISO instantiation


def _get_iso(iso: str):
    if iso not in _ISO_INSTANCES:
        with _ISO_INIT_LOCK:
            if iso not in _ISO_INSTANCES:  # re-check after acquiring lock
                try:
                    import gridstatus
                except ImportError:
                    raise ImportError("gridstatus is not installed. Run: pip install gridstatus")
                cls_name = ISO_CLASS_MAP.get(iso)
                if not cls_name:
                    raise ValueError(f"Unsupported ISO: {iso}. Choose from {list(ISO_CLASS_MAP)}")

                if iso == "ERCOT":
                    _inject_ercot_token()
                    _ISO_INSTANCES[iso] = getattr(gridstatus, cls_name)()
                elif iso == "PJM":
                    api_key = os.environ.get("PJM_API_KEY")
                    if not api_key:
                        raise EnvironmentError(
                            "PJM_API_KEY not set. Register free at pjm.com/api "
                            "then: export PJM_API_KEY=your_key"
                        )
                    _ISO_INSTANCES[iso] = getattr(gridstatus, cls_name)(api_key=api_key)
                else:
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


def _cache_path(iso: str, dataset: str) -> Path:
    return CACHE_DIR / f"{iso.lower()}_{dataset}.parquet"


# ── Raw fetch helpers (no caching) ───────────────────────────────────────────

def _pjm_get(url: str, params: dict, api_key: str, retries: int = 6) -> dict:
    """GET a PJM DataMiner 2 endpoint with exponential-backoff retry on 429."""
    import requests
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    delay = 15
    for attempt in range(retries):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            print(f"  [grid] PJM 429 — retrying in {delay}s (attempt {attempt + 1}/{retries})")
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def _fetch_pjm_lmp_direct(api_key: str, start: str, end: str, loc_type: str = "HUB") -> pd.DataFrame:
    """
    Fetch PJM DA hourly LMPs directly from DataMiner 2.

    Bypasses gridstatus to avoid its client-side location filtering, which
    downloads all 337k nodes per day before filtering — exhausting the rate
    limit across multiple pages.  Filtering server-side with type=HUB or
    type=ZONE reduces each monthly pull to ~9k rows (1 page).

    loc_type: "HUB" for system benchmark hubs, "ZONE" for settlement zones.
    """
    start_ept = pd.Timestamp(start).strftime("%m/%d/%Y %H:%M")
    end_ept   = pd.Timestamp(end).strftime("%m/%d/%Y %H:%M")
    url       = "https://api.pjm.com/api/v1/da_hrl_lmps"
    row_count = 50000
    start_row = 1
    all_items: list = []

    while True:
        params = {
            "row_is_current": "TRUE",
            "startRow":       start_row,
            "rowCount":       row_count,
            "type":           loc_type.upper(),
            "datetime_beginning_ept": f"{start_ept}to{end_ept}",
        }
        data  = _pjm_get(url, params, api_key)
        items = data.get("items", [])
        all_items.extend(items)
        total = data.get("totalRows", 0)
        if start_row + row_count - 1 >= total or not items:
            break
        start_row += row_count
        time.sleep(1)

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items).rename(columns={
        "datetime_beginning_utc": "time",
        "pnode_name":             "location",
        "total_lmp_da":           "lmp",
        "congestion_price_da":    "congestion",
        "system_energy_price_da": "energy",
        "marginal_loss_price_da": "loss",
    })
    return df


def _fetch_pjm_load_direct(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch PJM metered hourly load directly from DataMiner 2 (hrl_load_metered).

    inst_load has no historical depth; hrl_load_metered has full history.
    Aggregates all load areas to total system MW per hour.
    """
    start_ept = pd.Timestamp(start).strftime("%m/%d/%Y %H:%M")
    end_ept   = pd.Timestamp(end).strftime("%m/%d/%Y %H:%M")
    url       = "https://api.pjm.com/api/v1/hrl_load_metered"
    row_count = 50000
    start_row = 1
    all_items: list = []

    while True:
        params = {
            "startRow":  start_row,
            "rowCount":  row_count,
            "datetime_beginning_ept": f"{start_ept}to{end_ept}",
        }
        data  = _pjm_get(url, params, api_key)
        items = data.get("items", [])
        all_items.extend(items)
        total = data.get("totalRows", 0)
        if start_row + row_count - 1 >= total or not items:
            break
        start_row += row_count
        time.sleep(1)

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    df["time"] = pd.to_datetime(df["datetime_beginning_utc"])
    total_load = df.groupby("time")["mw"].sum().rename("load_mw").reset_index()
    return total_load


def _fetch_pjm_load_by_zone_direct(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch PJM metered hourly load directly from DataMiner 2, keeping the
    per-zone breakdown (unlike _fetch_pjm_load_direct, which sums to system total).
    Used for the DC load signal's zone-size denominator.
    """
    start_ept = pd.Timestamp(start).strftime("%m/%d/%Y %H:%M")
    end_ept   = pd.Timestamp(end).strftime("%m/%d/%Y %H:%M")
    url       = "https://api.pjm.com/api/v1/hrl_load_metered"
    row_count = 50000
    start_row = 1
    all_items: list = []

    while True:
        params = {
            "startRow":  start_row,
            "rowCount":  row_count,
            "datetime_beginning_ept": f"{start_ept}to{end_ept}",
        }
        data  = _pjm_get(url, params, api_key)
        items = data.get("items", [])
        all_items.extend(items)
        total = data.get("totalRows", 0)
        if start_row + row_count - 1 >= total or not items:
            break
        start_row += row_count
        time.sleep(1)

    if not all_items:
        return pd.DataFrame(columns=["time", "zone", "load_mw"])

    df = pd.DataFrame(all_items)
    df["time"] = pd.to_datetime(df["datetime_beginning_utc"])
    by_zone = df.groupby(["time", "zone"])["mw"].sum().rename("load_mw").reset_index()
    return by_zone


def _fetch_lmp_raw(iso_obj, iso: str, start: str, end: str, loc_type: str) -> pd.DataFrame:
    print(f"[grid] Fetching {iso} LMP ({loc_type}) {start} → {end}…")
    if iso == "SPP":
        return _fetch_spp_lmp_raw(iso_obj, start, end)
    if iso == "PJM":
        api_key = os.environ.get("PJM_API_KEY", "")
        # loc_type here is the gridstatus market string (e.g. "DAY_AHEAD_HOURLY").
        # Map to PJM DataMiner API type param: ZONE only when explicitly requested,
        # otherwise fall back to HUB (the default for all regular hub fetches).
        pjm_api_type = "ZONE" if loc_type.upper() == "ZONE" else "HUB"
        try:
            return _fetch_pjm_lmp_direct(api_key, start, end, loc_type=pjm_api_type)
        except Exception as exc:
            print(f"  [grid] PJM LMP fetch failed: {exc}")
            return pd.DataFrame()
    params = inspect.signature(iso_obj.get_lmp).parameters
    market_required = (
        "market" in params
        and params["market"].default is inspect.Parameter.empty
    )
    try:
        if market_required:
            df = iso_obj.get_lmp(date=start, end=end, market=loc_type, verbose=False)
        elif "location_type" in params:
            df = iso_obj.get_lmp(date=start, end=end, location_type=loc_type, verbose=False)
        elif "market" in params:
            df = iso_obj.get_lmp(date=start, end=end, market=loc_type, verbose=False)
        else:
            df = iso_obj.get_lmp(date=start, end=end, verbose=False)
    except Exception as exc:
        print(f"  [grid] {iso} LMP fetch failed: {exc}")
        return pd.DataFrame()
    return _normalise_lmp_columns(df)


def _fetch_load_raw(iso_obj, iso: str, start: str, end: str) -> pd.DataFrame:
    print(f"[grid] Fetching {iso} load {start} → {end}…")
    if iso == "PJM":
        api_key = os.environ.get("PJM_API_KEY", "")
        try:
            return _fetch_pjm_load_direct(api_key, start, end)
        except Exception as exc:
            print(f"  [grid] PJM load fetch failed: {exc}")
            return pd.DataFrame()
    try:
        if iso == "CAISO":
            # get_load() uses the outlook/history endpoint which only keeps ~2-3 years.
            # get_load_hourly() uses CAISO OASIS and has full historical depth.
            df = iso_obj.get_load_hourly(date=start, end=end, verbose=False)
        elif iso == "SPP":
            # SPP.get_load() does not accept an `end` parameter — fetch day-by-day.
            frames = []
            for d in pd.date_range(start, end, freq="D"):
                try:
                    day = iso_obj.get_load(date=d.strftime("%Y-%m-%d"), verbose=False)
                    if not day.empty:
                        frames.append(day)
                except Exception:
                    pass
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            df = iso_obj.get_load(date=start, end=end, verbose=False)
    except Exception as exc:
        print(f"  [grid] {iso} load fetch failed: {exc}")
        return pd.DataFrame()
    return _normalise_load_columns(df)


def _fetch_fuel_mix_raw(iso_obj, iso: str, start: str, end: str) -> pd.DataFrame:
    print(f"[grid] Fetching {iso} fuel mix {start} → {end}…")
    try:
        df = iso_obj.get_fuel_mix(date=start, end=end, verbose=False)
    except Exception as exc:
        print(f"  [grid] {iso} fuel mix fetch failed: {exc}")
        return pd.DataFrame()
    return df


def _fetch_spp_lmp_raw(spp_obj, start: str, end: str) -> pd.DataFrame:
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
    return _normalise_lmp_columns(result)


# ── Public fetch functions ────────────────────────────────────────────────────

def fetch_lmp(
    iso: str,
    start: str,
    end: str,
    location_type: str | None = None,
    locations: list[str] | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
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
    use_cache     : read/write parquet cache (monthly chunks)
    force_refresh : ignore cache and re-download the full range
    """
    loc_type = location_type or ISO_LOCATION_TYPE.get(iso, "hub")
    cache_base = _cache_path(iso, f"lmp_{loc_type}")

    if use_cache and not force_refresh:
        missing = find_missing_months(cache_base, start, end)
        if missing:
            if iso == "ERCOT":
                _fill_ercot_lmp_cache(missing, cache_base, loc_type)
            else:
                iso_obj = _get_iso(iso)

                def _fetch_month(ym: tuple[int, int]) -> None:
                    if iso == "PJM":
                        time.sleep(12)
                    ms, me = month_bounds(*ym)
                    df = _fetch_lmp_raw(iso_obj, iso, ms, me, loc_type)
                    if not df.empty:
                        save_monthly_chunks(df, cache_base, "time")
                    elif iso != "PJM":
                        # Only write empty placeholder for non-PJM ISOs so that
                        # PJM 429 failures are always retried on the next run.
                        p = chunk_path(cache_base, *ym)
                        if not p.exists():
                            pd.DataFrame().to_parquet(p, index=False)

                # PJM has a strict rate limit; serialize its requests to avoid 429s
                workers = 1 if iso == "PJM" else min(len(missing), 4)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(ex.map(_fetch_month, missing))

        merged = read_monthly_cache(cache_base, start, end)
    else:
        iso_obj = _get_iso(iso)
        merged = _fetch_lmp_raw(iso_obj, iso, start, end, loc_type)
        if use_cache and not merged.empty:
            save_monthly_chunks(merged, cache_base, "time")

    return _filter_locations(_filter_time(merged, start, end), locations)


def _fill_ercot_lmp_cache(
    missing: list[tuple[int, int]],
    cache_base: Path,
    loc_type: str,
) -> None:
    """
    Route ERCOT missing months to the right downloader.

    Historical months (> 90 days ago): annual bulk archive via ercot_bulk.
    The CDR API is public — no B2C auth is needed for these calls.
    One Excel file per year covers all 12 months efficiently.

    Recent months (within live API window): parallel month-by-month via
    gridstatus live API (B2C auth only triggered if recent months exist).
    """
    from grid_resilience.data.ercot_bulk import backfill_lmp, is_historical

    hist = [(y, m) for y, m in missing if is_historical(f"{y}-{m:02d}-01")]
    recent = [(y, m) for y, m in missing if not is_historical(f"{y}-{m:02d}-01")]

    if hist:
        h_start = f"{min(y for y, _ in hist)}-01-01"
        h_end = f"{max(y for y, _ in hist)}-12-31"
        backfill_lmp(h_start, h_end, cache_base)  # creates its own Ercot(), no auth

    if recent:
        iso_obj = _get_iso("ERCOT")  # B2C auth only triggered when live API needed

        def _fetch_month(ym: tuple[int, int]) -> None:
            ms, me = month_bounds(*ym)
            df = _fetch_lmp_raw(iso_obj, "ERCOT", ms, me, loc_type)
            if not df.empty:
                save_monthly_chunks(df, cache_base, "time")
            else:
                p = chunk_path(cache_base, *ym)
                if not p.exists():
                    pd.DataFrame().to_parquet(p, index=False)

        with ThreadPoolExecutor(max_workers=min(len(recent), 4)) as ex:
            list(ex.map(_fetch_month, recent))


def fetch_load(
    iso: str,
    start: str,
    end: str,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly total load (demand) for an ISO.

    Returns DataFrame with columns: time, load_mw
    """
    # SPP's get_load() calls the real-time chart API unconditionally before
    # checking the date, then raises NotSupported for any historical date.
    # Short-circuit here to avoid 31 wasted HTTP requests per missing month.
    if iso == "SPP":
        return pd.DataFrame()

    cache_base = _cache_path(iso, "load")

    if use_cache and not force_refresh:
        missing = find_missing_months(cache_base, start, end)
        if missing:
            iso_obj = _get_iso(iso)

            def _fetch_month(ym: tuple[int, int]) -> None:
                if iso == "PJM":
                    time.sleep(12)
                ms, me = month_bounds(*ym)
                df = _fetch_load_raw(iso_obj, iso, ms, me)
                if not df.empty:
                    save_monthly_chunks(df, cache_base, "time")
                elif iso != "PJM":
                    # Only write empty placeholder for non-PJM ISOs so that
                    # PJM 429 failures are always retried on the next run.
                    p = chunk_path(cache_base, *ym)
                    if not p.exists():
                        pd.DataFrame().to_parquet(p, index=False)

            # PJM has a strict rate limit; serialize its requests to avoid 429s
            workers = 1 if iso == "PJM" else min(len(missing), 4)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_fetch_month, missing))

        merged = read_monthly_cache(cache_base, start, end)
    else:
        iso_obj = _get_iso(iso)
        merged = _fetch_load_raw(iso_obj, iso, start, end)
        if use_cache and not merged.empty:
            save_monthly_chunks(merged, cache_base, "time")

    return _filter_time(merged, start, end)


def _fetch_zonal_load_raw(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Wraps _fetch_pjm_load_by_zone_direct with the same try/except-returns-empty
    pattern used by every other PJM fetcher in this module (_fetch_lmp_raw,
    _fetch_load_raw) so a missing PJM_API_KEY or PJM outage doesn't crash the
    whole run — it just leaves the DC load signal unpopulated for the month.
    """
    try:
        return _fetch_pjm_load_by_zone_direct(api_key, start, end)
    except Exception as exc:
        print(f"  [grid] PJM zonal load fetch failed: {exc}")
        return pd.DataFrame(columns=["time", "zone", "load_mw"])


# PJM's hrl_load_metered feed reports zones under short codes that don't match
# TICKER_NODE_MAP's zone names (e.g. "PS" not "PSEG"). Verified against the
# actual cached hrl_load_metered vocabulary — see finding #1 of the
# 2026-08-13 whole-branch review. AEP, ATSI, DOM (and others not listed here,
# e.g. AE, AP, DEOK, DPL, DUQ, EKPC, RECO, RTO) already match TICKER_NODE_MAP
# directly or aren't used by any covered ticker's load_zones.
# ME/PN added alongside the 2026-08-26 FE load_zones widening (METED,
# PENELEC) — PJM's real short codes for FirstEnergy's Met-Ed and Penelec
# operating companies; without these two, zone_size() would silently fail
# to match FE's new zones against real load data (see
# test_normalized_real_vocabulary_covers_every_pjm_ticker_load_zone).
_PJM_LOAD_ZONE_ALIASES = {
    "PS":  "PSEG",
    "PL":  "PPL",
    "PE":  "PECO",
    "BC":  "BGE",
    "PEP": "PEPCO",
    "CE":  "COMED",
    "JC":  "JCPL",
    "DAY": "DAYTON",
    "ME":  "METED",
    "PN":  "PENELEC",
}


def _normalize_pjm_load_zone(zone):
    """Map a raw PJM hrl_load_metered zone code to its TICKER_NODE_MAP name."""
    if not isinstance(zone, str):
        return zone
    return _PJM_LOAD_ZONE_ALIASES.get(zone, zone)


def fetch_zonal_load(
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch PJM hourly load broken out by zone. PJM-only (other ISOs return empty) —
    used for the DC load signal's zone-size denominator.
    Returns DataFrame with columns: time, zone, load_mw

    The `zone` column is normalized to TICKER_NODE_MAP's zone-name vocabulary
    via _PJM_LOAD_ZONE_ALIASES (applied here, once, at the fetch boundary —
    including for rows already sitting in the on-disk cache under PJM's raw
    short codes — rather than scattered across callers).
    """
    cache_base = _cache_path("PJM", "load_zonal")
    api_key = os.environ.get("PJM_API_KEY", "")

    if use_cache:
        missing = find_missing_months(cache_base, start, end)
        if missing:
            def _fetch_month(ym: tuple[int, int]) -> None:
                time.sleep(12)
                ms, me = month_bounds(*ym)
                df = _fetch_zonal_load_raw(api_key, ms, me)
                if not df.empty:
                    save_monthly_chunks(df, cache_base, "time")

            for ym in missing:
                _fetch_month(ym)

        merged = read_monthly_cache(cache_base, start, end)
    else:
        merged = _fetch_zonal_load_raw(api_key, start, end)

    merged = _filter_time(merged, start, end)
    if not merged.empty and "zone" in merged.columns:
        merged = merged.copy()
        merged["zone"] = merged["zone"].apply(_normalize_pjm_load_zone)
    return merged


def fetch_fuel_mix(
    iso: str,
    start: str,
    end: str,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly generation fuel mix (MW by fuel type) for an ISO.
    Used as input to the reserve-tightness sub-signal.
    """
    cache_base = _cache_path(iso, "fuel_mix")

    if use_cache and not force_refresh:
        missing = find_missing_months(cache_base, start, end)
        if missing:
            iso_obj = _get_iso(iso)

            def _fetch_month(ym: tuple[int, int]) -> None:
                ms, me = month_bounds(*ym)
                df = _fetch_fuel_mix_raw(iso_obj, iso, ms, me)
                if not df.empty:
                    save_monthly_chunks(df, cache_base, "time")
                else:
                    p = chunk_path(cache_base, *ym)
                    if not p.exists():
                        pd.DataFrame().to_parquet(p, index=False)

            with ThreadPoolExecutor(max_workers=min(len(missing), 4)) as ex:
                list(ex.map(_fetch_month, missing))

        merged = read_monthly_cache(cache_base, start, end)
    else:
        iso_obj = _get_iso(iso)
        merged = _fetch_fuel_mix_raw(iso_obj, iso, start, end)
        if use_cache and not merged.empty:
            save_monthly_chunks(merged, cache_base, "time")

    return _filter_time(merged, start, end)


# ── Filter helpers ────────────────────────────────────────────────────────────

def _filter_time(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    time_col = _time_col(df)
    mask = pd.to_datetime(df[time_col]).between(start, end)
    return df[mask].reset_index(drop=True)


def _filter_locations(df: pd.DataFrame, locations: list[str] | None) -> pd.DataFrame:
    if not locations or df.empty:
        return df
    loc_col = _location_col(df)
    return df[df[loc_col].isin(locations)]


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


def daily_spread_summary(zone_lmp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily inter-zonal price spread as a congestion_frac proxy.

    Hourly spread = max_lmp - min_lmp across all locations.
    Normalised as frac = spread / (|mean_lmp| + 1e-6).
    Daily value = mean of hourly fracs.

    Returns DataFrame indexed by date with column: congestion_frac
    """
    if zone_lmp_df.empty:
        return pd.DataFrame(columns=["congestion_frac"])

    time_col = _time_col(zone_lmp_df)
    df = zone_lmp_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    hourly = df.groupby(level=0)["lmp"].agg(["max", "min", "mean"])
    hourly["spread"] = hourly["max"] - hourly["min"]
    hourly["frac"] = (hourly["spread"] / (hourly["mean"].abs() + 1e-6)).clip(upper=10.0)

    daily_frac = hourly["frac"].resample("D").mean().rename("congestion_frac")
    daily_frac.index = pd.to_datetime(daily_frac.index)
    daily_frac.index.name = "date"
    return daily_frac.to_frame()


def fill_congestion_from_spread(
    daily_lmp: pd.DataFrame,
    spread_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill NaN congestion_frac rows in daily_lmp with spread-based values.
    Component-based values (non-NaN) are preserved.
    Returns the modified daily_lmp DataFrame.
    """
    if spread_df.empty or "congestion_frac" not in spread_df.columns:
        return daily_lmp
    if "congestion_frac" not in daily_lmp.columns:
        return daily_lmp
    daily_lmp = daily_lmp.copy()
    mask = daily_lmp["congestion_frac"].isna()
    daily_lmp.loc[mask, "congestion_frac"] = (
        spread_df["congestion_frac"].reindex(daily_lmp.index[mask])
    )
    return daily_lmp


# ── Internal column normalisation ────────────────────────────────────────────

def _normalise_lmp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common ISO-specific column names to standard schema."""
    rename: dict[str, str] = {}

    # Time column
    for c in df.columns:
        cl = c.lower()
        if cl in ("time", "datetime", "interval_ending", "delivery_date",
                  "operatingday", "mktintervalstart", "timestamp",
                  "interval start", "interval_start"):
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
        if cl in ("time", "datetime", "interval_ending", "timestamp", "interval start"):
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
