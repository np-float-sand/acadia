from __future__ import annotations

import pandas as pd

from grid_resilience.config import CACHE_DIR

# One entry per PJM Load Analysis Subcommittee "Large Load Adjustment
# Requests" vintage, confirmed live during design (2026-08-26). Add new
# vintages here as PJM posts them each fall — never guess the URL pattern,
# PJM's media paths are not predictable from the date alone.
LARGE_LOAD_SOURCE_URLS = {
    "2026": (
        "https://www.pjm.com/-/media/DotCom/committees-groups/subcommittees/"
        "las/2025/20250916/20250916-post-meeting---informational-only---"
        "large-load-adjustment-requests.xlsx"
    ),
}


def _parse_industry_tags(raw_summary: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'summary' sheet's ZONENAME/AREANAME/industry columns."""
    df = raw_summary.iloc[1:].copy()
    df.columns = ["zone", "area", "has_capacity", "has_demand", "industry", "note"]
    return df[df["zone"].notna()][["zone", "area", "industry"]].reset_index(drop=True)


def _parse_requests_sheet(raw: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Parse a '<vintage> DEMAND/CAPACITY Requests' sheet into long format."""
    header_row = raw.iloc[3]
    years = [int(y) for y in header_row[3:] if pd.notna(y)]
    df = raw.iloc[4:].copy()
    df.columns = ["_drop", "zone", "area"] + years
    df = df[df["zone"].notna()].drop(columns="_drop")
    long_df = df.melt(id_vars=["zone", "area"], var_name="year", value_name=value_name)
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    return long_df.dropna(subset=[value_name]).reset_index(drop=True)


def fetch_large_load_adjustment(vintage: str = "2026", use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch and parse a PJM LAS 'Large Load Adjustment Requests' vintage.
    Returns long format: [zone, area, industry, year, mw_demand, mw_capacity].
    """
    cache_file = CACHE_DIR / f"pjm_large_load_adjustment_{vintage}.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    url = LARGE_LOAD_SOURCE_URLS[vintage]
    xls = pd.ExcelFile(url)
    industry = _parse_industry_tags(xls.parse("summary"))
    demand = _parse_requests_sheet(xls.parse(f"{vintage} DEMAND Requests"), "mw_demand")
    capacity = _parse_requests_sheet(xls.parse(f"{vintage} CAPACITY Requests"), "mw_capacity")

    merged = demand.merge(capacity, on=["zone", "area", "year"], how="outer")
    merged = merged.merge(industry, on=["zone", "area"], how="left")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_file)
    return merged
