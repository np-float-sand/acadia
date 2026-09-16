"""National (cross-ISO) Grid Stress Index for the Screened Electrification supplier
universe -- spec docs/superpowers/specs/2026-09-15-screened-electrification-sow-plan-design.md
§3.4 Step A.

Equipment suppliers (ETN, HUBB, GEV, ...) are national/global manufacturers with no
single home ISO, unlike grid_resilience's utilities (which have a service territory).
Rather than force a ticker_iso_map onto grid_stress_index.gsi_for_ticker(), this blends
grid_resilience's existing per-ISO GSI into one national series and maps every supplier
ticker to it via a synthetic single key -- reusing conditional_beta.py unmodified.
"""
from __future__ import annotations

import pandas as pd

from grid_resilience.config import SUPPORTED_ISOS
from grid_resilience.data.grid_data import fetch_lmp, fetch_load, daily_lmp_summary, daily_load_summary
from grid_resilience.signals.stress_events import build_full_event_calendar
from grid_resilience.signals.grid_stress_index import build_multi_iso_gsi

NATIONAL_KEY = "NATIONAL"


def build_national_gsi(
    daily_lmp_by_iso: dict[str, pd.DataFrame],
    daily_load_by_iso: dict[str, pd.DataFrame],
    isos: list[str] = SUPPORTED_ISOS,
) -> pd.DataFrame:
    """Equal-weighted mean of each ISO's GSI, per day. Returns a DataFrame with a
    single 'gsi' column, or an empty DataFrame if no requested ISO has data."""
    events_df = build_full_event_calendar(daily_lmp_by_iso, isos=isos)
    gsi_by_iso = build_multi_iso_gsi(daily_lmp_by_iso, daily_load_by_iso, events_df)
    series = [df["gsi"] for iso, df in gsi_by_iso.items() if iso in isos and not df.empty]
    if not series:
        return pd.DataFrame(columns=["gsi"])
    national = pd.concat(series, axis=1).mean(axis=1, skipna=True)
    return national.to_frame("gsi")


def fetch_and_build_national_gsi(
    start: str, end: str, isos: list[str] = SUPPORTED_ISOS,
) -> pd.DataFrame:
    """Fetch raw LMP/load for each ISO (cached) and build the blended national GSI.
    Not unit tested directly -- an orchestration layer, consistent with
    grid_resilience.main.run() also being covered only by real runs, not pytest.
    Covered by Task 5's real-data gate run."""
    daily_lmp_by_iso, daily_load_by_iso = {}, {}
    for iso in isos:
        lmp_raw = fetch_lmp(iso, start, end)
        load_raw = fetch_load(iso, start, end)
        daily_lmp_by_iso[iso] = daily_lmp_summary(lmp_raw, iso)
        daily_load_by_iso[iso] = daily_load_summary(load_raw)
    return build_national_gsi(daily_lmp_by_iso, daily_load_by_iso, isos=isos)


def uniform_ticker_iso_map(tickers: list[str]) -> dict[str, str]:
    """Map every ticker to NATIONAL_KEY so conditional_beta.gsi_for_ticker() resolves
    every ticker to the same blended national series, unmodified."""
    return {t: NATIONAL_KEY for t in tickers}
