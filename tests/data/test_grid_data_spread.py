import numpy as np
import pandas as pd
import pytest

from grid_resilience.data.grid_data import daily_spread_summary, fill_congestion_from_spread


# ── daily_spread_summary ──────────────────────────────────────────────────────

def test_spread_empty_input_returns_empty_with_column():
    result = daily_spread_summary(pd.DataFrame())
    assert result.empty
    assert "congestion_frac" in result.columns


def test_spread_single_location_gives_zero_spread():
    """One location → max == min → spread == 0 → frac == 0."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    df = pd.DataFrame({
        "time": times,
        "location": ["HUB_A"] * 24,
        "lmp": [50.0] * 24,
    })
    result = daily_spread_summary(df)
    assert not result.empty
    assert result["congestion_frac"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_spread_two_zones_constant_difference():
    """ZONE_A at 100, ZONE_B at 50 for 24 hours.
    Hourly: spread=50, mean=75, frac=50/75.
    Daily mean of 24 identical frac values = 50/75."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    df = pd.DataFrame({
        "time": list(times) * 2,
        "location": ["ZONE_A"] * 24 + ["ZONE_B"] * 24,
        "lmp": [100.0] * 24 + [50.0] * 24,
    })
    result = daily_spread_summary(df)
    expected_frac = 50.0 / 75.0
    assert result["congestion_frac"].iloc[0] == pytest.approx(expected_frac, rel=1e-4)


def test_spread_multi_day_aggregation():
    """Two days of data produce two rows, one per day."""
    times_d1 = pd.date_range("2024-01-01", periods=24, freq="h")
    times_d2 = pd.date_range("2024-01-02", periods=24, freq="h")
    all_times = list(times_d1) * 2 + list(times_d2) * 2

    df = pd.DataFrame({
        "time": all_times,
        "location": (["ZONE_A"] * 24 + ["ZONE_B"] * 24) * 2,
        "lmp": [100.0] * 24 + [50.0] * 24 + [80.0] * 24 + [60.0] * 24,
    })
    result = daily_spread_summary(df)
    assert len(result) == 2
    assert result.index.name == "date"
    # Day 1: spread=50, mean=75, frac=50/75
    assert result["congestion_frac"].iloc[0] == pytest.approx(50.0 / 75.0, rel=1e-4)
    # Day 2: spread=20, mean=70, frac=20/70
    assert result["congestion_frac"].iloc[1] == pytest.approx(20.0 / 70.0, rel=1e-4)


# ── fill_congestion_from_spread ───────────────────────────────────────────────

def test_fill_replaces_nan_rows():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    daily_lmp = pd.DataFrame({"congestion_frac": [np.nan, np.nan]}, index=dates)
    daily_lmp.index.name = "date"
    spread_df = pd.DataFrame({"congestion_frac": [0.1, 0.2]}, index=dates)
    spread_df.index.name = "date"

    result = fill_congestion_from_spread(daily_lmp, spread_df)
    assert result["congestion_frac"].iloc[0] == pytest.approx(0.1)
    assert result["congestion_frac"].iloc[1] == pytest.approx(0.2)


def test_fill_preserves_existing_non_nan_values():
    """Component-based value (0.3) must not be overwritten by spread (0.9)."""
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    daily_lmp = pd.DataFrame({"congestion_frac": [np.nan, 0.3]}, index=dates)
    daily_lmp.index.name = "date"
    spread_df = pd.DataFrame({"congestion_frac": [0.1, 0.9]}, index=dates)
    spread_df.index.name = "date"

    result = fill_congestion_from_spread(daily_lmp, spread_df)
    assert result["congestion_frac"].iloc[0] == pytest.approx(0.1)   # filled
    assert result["congestion_frac"].iloc[1] == pytest.approx(0.3)   # preserved


def test_fill_empty_spread_leaves_nans_untouched():
    dates = pd.to_datetime(["2024-01-01"])
    daily_lmp = pd.DataFrame({"congestion_frac": [np.nan]}, index=dates)
    result = fill_congestion_from_spread(daily_lmp, pd.DataFrame(columns=["congestion_frac"]))
    assert pd.isna(result["congestion_frac"].iloc[0])


def test_fill_returns_dataframe():
    dates = pd.to_datetime(["2024-01-01"])
    daily_lmp = pd.DataFrame({"congestion_frac": [np.nan]}, index=dates)
    spread_df = pd.DataFrame({"congestion_frac": [0.5]}, index=dates)
    result = fill_congestion_from_spread(daily_lmp, spread_df)
    assert isinstance(result, pd.DataFrame)


def test_fill_does_not_mutate_input():
    """Caller's DataFrame must be unchanged after fill."""
    dates = pd.to_datetime(["2024-01-01"])
    daily_lmp = pd.DataFrame({"congestion_frac": [np.nan]}, index=dates)
    spread_df = pd.DataFrame({"congestion_frac": [0.5]}, index=dates)
    fill_congestion_from_spread(daily_lmp, spread_df)
    assert pd.isna(daily_lmp["congestion_frac"].iloc[0])


def test_spread_clamps_extreme_frac():
    """Symmetric positive/negative prices → near-zero mean → frac capped at 10.0."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    df = pd.DataFrame({
        "time": list(times) * 2,
        "location": ["ZONE_A"] * 24 + ["ZONE_B"] * 24,
        "lmp": [100.0] * 24 + [-100.0] * 24,  # mean=0, spread=200, raw_frac >> 10
    })
    result = daily_spread_summary(df)
    assert result["congestion_frac"].iloc[0] == pytest.approx(10.0)
