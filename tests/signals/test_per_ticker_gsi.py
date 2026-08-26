"""Tests for per-ticker GSI routing (PJM zone GSI support)."""
import numpy as np
import pandas as pd
import pytest

from grid_resilience.signals.grid_stress_index import (
    build_multi_iso_gsi,
    gsi_for_ticker,
    build_gsi,
)


def _make_gsi_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", name="date")
    return pd.DataFrame({"gsi": rng.uniform(0, 1, n)}, index=dates)


def _make_daily_lmp(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", name="date")
    return pd.DataFrame({
        "lmp_max": rng.uniform(30, 100, n),
        "lmp_mean": rng.uniform(20, 80, n),
        "spike_hours": rng.integers(0, 5, n).astype(float),
        "congestion_frac": rng.uniform(0, 0.3, n),
    }, index=dates)


# ── gsi_for_ticker ────────────────────────────────────────────────────────────

def test_gsi_for_ticker_returns_iso_wide_when_no_ticker_key():
    iso_gsi = _make_gsi_df(seed=1)
    gsi_by_iso = {"PJM": pd.DataFrame({"gsi": iso_gsi["gsi"]})}
    ticker_iso_map = {"AEP": "PJM"}
    result = gsi_for_ticker("AEP", gsi_by_iso, ticker_iso_map)
    pd.testing.assert_series_equal(result, iso_gsi["gsi"].rename("gsi_AEP"))


def test_gsi_for_ticker_prefers_ticker_specific_key():
    iso_gsi   = _make_gsi_df(seed=1)
    zone_gsi  = _make_gsi_df(seed=99)  # different values
    gsi_by_iso = {
        "PJM":     pd.DataFrame({"gsi": iso_gsi["gsi"]}),
        "PJM:AEP": pd.DataFrame({"gsi": zone_gsi["gsi"]}),
    }
    ticker_iso_map = {"AEP": "PJM"}
    result = gsi_for_ticker("AEP", gsi_by_iso, ticker_iso_map)
    pd.testing.assert_series_equal(result, zone_gsi["gsi"].rename("gsi_AEP"))


def test_gsi_for_ticker_falls_back_when_ticker_key_absent():
    iso_gsi = _make_gsi_df(seed=1)
    # Only system-wide key; no PJM:EXC
    gsi_by_iso = {"PJM": pd.DataFrame({"gsi": iso_gsi["gsi"]})}
    ticker_iso_map = {"EXC": "PJM"}
    result = gsi_for_ticker("EXC", gsi_by_iso, ticker_iso_map)
    assert result.name == "gsi_EXC"
    assert len(result) == len(iso_gsi)


def test_gsi_for_ticker_returns_empty_for_unknown_ticker():
    gsi_by_iso = {"ERCOT": _make_gsi_df()}
    result = gsi_for_ticker("UNKNOWN", gsi_by_iso, {})
    assert result.empty
    assert result.name == "gsi_UNKNOWN"


def test_gsi_for_ticker_result_named_after_ticker():
    gsi_by_iso = {"PJM": pd.DataFrame({"gsi": _make_gsi_df()["gsi"]}),
                  "PJM:PPL": pd.DataFrame({"gsi": _make_gsi_df(seed=5)["gsi"]})}
    result = gsi_for_ticker("PPL", gsi_by_iso, {"PPL": "PJM"})
    assert result.name == "gsi_PPL"


# ── build_multi_iso_gsi with compound keys ────────────────────────────────────

def test_build_multi_iso_gsi_handles_compound_keys():
    lmp_pjm    = _make_daily_lmp(seed=0)
    lmp_pjm_aep = _make_daily_lmp(seed=1)
    load_pjm   = pd.DataFrame({"load_max_mw": np.full(100, 50000)},
                               index=lmp_pjm.index)
    daily_lmp_by_iso  = {"PJM": lmp_pjm, "PJM:AEP": lmp_pjm_aep}
    daily_load_by_iso = {"PJM": load_pjm}
    events_df = pd.DataFrame(columns=["start", "end", "iso", "type", "name", "severity"])

    result = build_multi_iso_gsi(daily_lmp_by_iso, daily_load_by_iso, events_df)

    assert "PJM" in result
    assert "PJM:AEP" in result
    assert "gsi" in result["PJM"].columns
    assert "gsi" in result["PJM:AEP"].columns


def test_build_multi_iso_gsi_compound_key_uses_base_iso_load():
    """PJM:AEP should use PJM load data, producing non-constant reserve_tightness_z."""
    rng = np.random.default_rng(42)
    lmp = _make_daily_lmp(seed=2)
    # Varying load so rolling std > 0 and reserve_tightness_z is computable
    load_vals = rng.uniform(40000, 60000, 100)
    load_pjm = pd.DataFrame({"load_max_mw": load_vals}, index=lmp.index)
    daily_lmp_by_iso  = {"PJM:AEP": lmp}
    daily_load_by_iso = {"PJM": load_pjm}
    events_df = pd.DataFrame(columns=["start", "end", "iso", "type", "name", "severity"])

    result_with_load    = build_multi_iso_gsi(daily_lmp_by_iso, daily_load_by_iso, events_df)
    result_without_load = build_multi_iso_gsi(daily_lmp_by_iso, {}, events_df)

    # Without load, reserve_tightness_z falls back to constant 0.0
    rt_without = result_without_load["PJM:AEP"]["reserve_tightness_z"]
    assert (rt_without == 0.0).all(), "no-load path must return constant 0"
    # With load, reserve_tightness_z varies (non-zero std after warm-up)
    rt_with = result_with_load["PJM:AEP"]["reserve_tightness_z"].dropna()
    assert rt_with.std() > 0, "load path must produce varying reserve_tightness_z"
