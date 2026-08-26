import pandas as pd
from grid_resilience.data.ercot_large_load_data import load_ercot_large_load_seed, compute_ercot_signal


def test_load_ercot_large_load_seed_reads_system_wide_rows():
    df = load_ercot_large_load_seed()
    assert set(df.columns) == {"snapshot_date", "scope", "tsp", "standalone_mw", "co_located_mw", "total_mw", "source_url"}
    system_wide = df[df["scope"] == "system_wide"]
    assert len(system_wide) == 11
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_date"])
    row = system_wide[system_wide["snapshot_date"] == pd.Timestamp("2026-03-01")].iloc[0]
    assert row["total_mw"] == 238629


def test_compute_ercot_signal_level_and_momentum_for_mapped_ticker():
    history = pd.DataFrame([
        {"snapshot_date": pd.Timestamp("2025-06-01"), "scope": "tsp_level", "tsp": "AEP", "total_mw": 30000, "standalone_mw": 30000, "co_located_mw": 0, "source_url": "x"},
        {"snapshot_date": pd.Timestamp("2025-08-01"), "scope": "tsp_level", "tsp": "AEP", "total_mw": 34000, "standalone_mw": 34000, "co_located_mw": 0, "source_url": "x"},
    ])
    tsp_map = {"AEP": "AEP"}

    result = compute_ercot_signal(
        tickers=["AEP"], tsp_map=tsp_map, ercot_history=history,
        as_of_dates=[pd.Timestamp("2025-09-01")],
    )
    assert result.loc[pd.Timestamp("2025-09-01"), "AEP"] == 34000.0  # latest known level, no denominator yet in v1


def test_compute_ercot_signal_unmapped_ticker_is_nan():
    history = pd.DataFrame(columns=["snapshot_date", "scope", "tsp", "total_mw", "standalone_mw", "co_located_mw", "source_url"])
    result = compute_ercot_signal(
        tickers=["FE"], tsp_map={"AEP": "AEP"}, ercot_history=history,
        as_of_dates=[pd.Timestamp("2025-09-01")],
    )
    assert pd.isna(result.loc[pd.Timestamp("2025-09-01"), "FE"])
