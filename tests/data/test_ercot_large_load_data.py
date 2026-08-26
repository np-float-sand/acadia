import pandas as pd
from grid_resilience.data.ercot_large_load_data import load_ercot_large_load_seed


def test_load_ercot_large_load_seed_reads_system_wide_rows():
    df = load_ercot_large_load_seed()
    assert set(df.columns) == {"snapshot_date", "scope", "tsp", "standalone_mw", "co_located_mw", "total_mw", "source_url"}
    system_wide = df[df["scope"] == "system_wide"]
    assert len(system_wide) == 11
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_date"])
    row = system_wide[system_wide["snapshot_date"] == pd.Timestamp("2026-03-01")].iloc[0]
    assert row["total_mw"] == 238629
