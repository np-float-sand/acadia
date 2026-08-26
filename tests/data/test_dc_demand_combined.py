import pandas as pd
import numpy as np


def test_combine_dc_demand_layers_zscores_each_layer_separately():
    from grid_resilience.data.dc_demand_combined import combine_dc_demand_layers

    dates = [pd.Timestamp("2025-12-31")]
    pjm = pd.DataFrame({"AEP": [10.0], "FE": [20.0]}, index=dates)       # raw scale ~0-100s
    ercot = pd.DataFrame({"CNP": [34000.0]}, index=dates)                # raw scale ~10,000s
    hyperscaler = pd.DataFrame({"CEG": [835.0], "TLN": [1920.0]}, index=dates)

    result = combine_dc_demand_layers(pjm, ercot, hyperscaler)

    assert set(result.columns) == {"AEP", "FE", "CNP", "CEG", "TLN"}
    # single-column-per-layer or single-value populations z-score to 0 (no within-layer spread)
    assert result.loc[dates[0], "CNP"] == 0.0
    # two-ticker PJM population: AEP (lower) should be negative, FE positive
    assert result.loc[dates[0], "AEP"] < 0 < result.loc[dates[0], "FE"]


def test_combine_dc_demand_layers_ticker_with_no_layer_is_absent():
    from grid_resilience.data.dc_demand_combined import combine_dc_demand_layers
    dates = [pd.Timestamp("2025-12-31")]
    pjm = pd.DataFrame({"AEP": [10.0]}, index=dates)
    ercot = pd.DataFrame(index=dates)
    hyperscaler = pd.DataFrame(index=dates)
    result = combine_dc_demand_layers(pjm, ercot, hyperscaler)
    assert "PCG" not in result.columns
