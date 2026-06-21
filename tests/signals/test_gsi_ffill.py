"""Tests that build_gsi forward-fills gaps within the data range (limit 10 days)."""
import numpy as np
import pandas as pd
import pytest

from grid_resilience.signals.grid_stress_index import build_gsi


def _make_daily_lmp(dates: pd.DatetimeIndex, lmp_values=None) -> pd.DataFrame:
    if lmp_values is None:
        lmp_values = np.random.default_rng(0).uniform(20, 80, len(dates))
    return pd.DataFrame({"lmp_max": lmp_values}, index=dates)


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["start", "end", "iso", "type", "name", "severity"])


def test_gsi_has_no_gaps_within_5_day_hole():
    """A 5-day hole in LMP data should be ffilled — GSI is continuous."""
    # Jan 1–20, skip 21–25, resume 26–31
    dates = pd.date_range("2024-01-01", "2024-01-20").append(
        pd.date_range("2024-01-26", "2024-01-31")
    )
    daily_lmp = _make_daily_lmp(dates)
    result = build_gsi(daily_lmp, pd.DataFrame(), _empty_events(), iso="PJM")
    # All dates from min to max should be present after ffill
    full_range = pd.date_range(result.index.min(), result.index.max(), freq="D")
    missing = full_range.difference(result.index)
    assert len(missing) == 0, f"Expected no gaps within 10-day limit, got {len(missing)} missing dates"


def test_gsi_gap_larger_than_limit_stays_nan():
    """A 15-day hole exceeds the ffill limit — those dates must remain absent."""
    dates = pd.date_range("2024-01-01", "2024-01-10").append(
        pd.date_range("2024-01-26", "2024-02-10")
    )
    daily_lmp = _make_daily_lmp(dates)
    result = build_gsi(daily_lmp, pd.DataFrame(), _empty_events(), iso="PJM")
    # Dates 11–25 are a 15-day gap; only 10 days should be filled
    # so at least some dates in 2024-01-21 to 2024-01-25 should still be absent
    hole = pd.date_range("2024-01-21", "2024-01-25")
    present_in_hole = hole.intersection(result.index)
    assert len(present_in_hole) == 0, (
        f"Expected dates deep in 15-day gap to be absent, but {len(present_in_hole)} are present"
    )


def test_gsi_no_change_when_data_already_dense():
    """Dense data (no gaps) should be unaffected by the ffill."""
    dates = pd.date_range("2024-01-01", "2024-03-31")
    daily_lmp = _make_daily_lmp(dates)
    result = build_gsi(daily_lmp, pd.DataFrame(), _empty_events(), iso="ERCOT")
    assert len(result) == len(dates)
    assert result["gsi"].notna().all()
