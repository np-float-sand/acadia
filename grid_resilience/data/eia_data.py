from __future__ import annotations

"""
EIA API data fetcher for fleet composition and generation mix.

Data source: U.S. Energy Information Administration Open Data API v2
  Docs  : https://www.eia.gov/opendata/
  Key   : Free — register at https://www.eia.gov/opendata/register.php
  Set env var: EIA_API_KEY=your_key

Key datasets used here:
  /electricity/facility-fuel    — monthly net generation by plant + fuel type
  /electricity/operating-generator/generator — plant-level capacity (Form 860)

Both endpoints are public and free once you have a key.
"""

import os
from pathlib import Path

import pandas as pd
import requests

from grid_resilience.config import CACHE_DIR

_EIA_BASE = "https://api.eia.gov/v2"

# EIA balancing authority codes that map to our ISOs
ISO_BA_CODES: dict[str, list[str]] = {
    "ERCOT": ["ERCO"],
    "PJM":   ["PJM"],
    "MISO":  ["MISO"],
    "CAISO": ["CISO"],
    "SPP":   ["SWPP"],
}

# Renewable fuel type codes used in EIA facility-fuel data
RENEWABLE_FUELS = {"SUN", "WND", "WAT", "GEO", "OBG", "OBL", "OBS", "OTH"}
# All fuel type codes for reference:
# NG=Natural Gas, COL=Coal, NUC=Nuclear, SUN=Solar, WND=Wind,
# WAT=Hydro, OIL=Petroleum, GEO=Geothermal, OBG/OBL/OBS=Biomass


def _api_key() -> str:
    key = os.environ.get("EIA_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "EIA_API_KEY not set. Register free at https://www.eia.gov/opendata/register.php "
            "and set the env var before calling EIA functions."
        )
    return key


def fetch_generation_mix(
    iso: str,
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch monthly net generation (MWh) by fuel type for an ISO's balancing authority.

    Returns DataFrame with columns:
        period      : YYYY-MM string
        fuel_type   : EIA fuel type code (NG, COL, SUN, WND, …)
        generation  : net generation (MWh)
        ba_code     : balancing authority code

    Used to compute the renewable penetration share for each ISO,
    which feeds into the fleet quality component of the resilience factor.
    """
    ba_codes = ISO_BA_CODES.get(iso)
    if not ba_codes:
        raise ValueError(f"No EIA BA codes defined for ISO: {iso}")

    cache_file = CACHE_DIR / f"eia_genmix_{iso.lower()}_{start[:7]}_{end[:7]}.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    key = _api_key()
    frames = []

    for ba in ba_codes:
        url = f"{_EIA_BASE}/electricity/facility-fuel/data/"
        params = {
            "api_key":                  key,
            "frequency":                "monthly",
            "data[0]":                  "generation",
            "facets[balancing_authority_code][]": ba,
            "start":                    start[:7],
            "end":                      end[:7],
            "sort[0][column]":          "period",
            "sort[0][direction]":       "asc",
            "length":                   5000,
        }
        print(f"[eia] Fetching generation mix for {ba} ({start} → {end})…")
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json().get("response", {}).get("data", [])
            if data:
                df = pd.DataFrame(data)
                df["ba_code"] = ba
                frames.append(df)
        except Exception as exc:
            print(f"  [eia] {ba} fetch failed: {exc}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"fuelTypeDescription": "fuel_type",
                                     "net-generation": "generation"})

    if use_cache:
        result.to_parquet(cache_file, index=False)
    return result


def fetch_plant_capacity(
    iso: str,
    year: int = 2023,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch plant-level nameplate capacity (MW) by fuel type (EIA Form 860).

    Returns DataFrame with columns:
        plant_id, plant_name, state, fuel_type, capacity_mw, ba_code

    Used for the utility → node asset mapping validation.
    """
    ba_codes = ISO_BA_CODES.get(iso)
    if not ba_codes:
        raise ValueError(f"No EIA BA codes defined for ISO: {iso}")

    cache_file = CACHE_DIR / f"eia_capacity_{iso.lower()}_{year}.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    key = _api_key()
    frames = []

    for ba in ba_codes:
        url = f"{_EIA_BASE}/electricity/operating-generator/generator/data/"
        params = {
            "api_key":                  key,
            "frequency":                "annual",
            "data[0]":                  "nameplate-capacity-mw",
            "facets[balancing_authority_code][]": ba,
            "start":                    str(year),
            "end":                      str(year),
            "length":                   5000,
        }
        print(f"[eia] Fetching plant capacity for {ba} ({year})…")
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json().get("response", {}).get("data", [])
            if data:
                df = pd.DataFrame(data)
                df["ba_code"] = ba
                frames.append(df)
        except Exception as exc:
            print(f"  [eia] {ba} capacity fetch failed: {exc}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    if use_cache:
        result.to_parquet(cache_file, index=False)
    return result


def compute_renewable_share(gen_mix_df: pd.DataFrame) -> pd.Series:
    """
    Given the output of fetch_generation_mix(), return a monthly Series of
    renewable generation share (0–1) for each period.
    """
    if gen_mix_df.empty:
        return pd.Series(dtype=float)

    fuel_col = next((c for c in gen_mix_df.columns if "fuel" in c.lower()), None)
    gen_col  = next((c for c in gen_mix_df.columns if "generation" in c.lower() or
                     gen_mix_df[c].dtype in (float, int) and c != "period"), None)

    if fuel_col is None or gen_col is None:
        return pd.Series(dtype=float)

    df = gen_mix_df.copy()
    df[gen_col] = pd.to_numeric(df[gen_col], errors="coerce").fillna(0)
    df["is_renewable"] = df[fuel_col].str.upper().isin(RENEWABLE_FUELS)

    total = df.groupby("period")[gen_col].sum()
    renew = df[df["is_renewable"]].groupby("period")[gen_col].sum()
    share = (renew / total).fillna(0).rename("renewable_share")
    return share
