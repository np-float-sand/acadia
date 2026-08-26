import pandas as pd
from grid_resilience.analysis.short_sample_report import window_sensitivity_report, sample_size_caveat


def test_window_sensitivity_report_drops_first_and_last():
    returns = pd.Series([0.10, 0.02, 0.03, 0.04, -0.20], index=pd.date_range("2025-01-01", periods=5, freq="ME"))
    result = window_sensitivity_report(returns)
    full_row = result[result["variant"] == "full"].iloc[0]
    assert full_row["n_periods"] == 5
    drop_first = result[result["variant"] == "drop_first"].iloc[0]
    assert drop_first["n_periods"] == 4
    assert abs(drop_first["mean_return"] - pd.Series([0.02, 0.03, 0.04, -0.20]).mean()) < 1e-9
    drop_both = result[result["variant"] == "drop_first_and_last"].iloc[0]
    assert drop_both["n_periods"] == 3


def test_sample_size_caveat_below_threshold():
    msg = sample_size_caveat(n_periods=5, minimum_trusted=12)
    assert "5" in msg and "12" in msg


def test_sample_size_caveat_above_threshold_is_empty():
    assert sample_size_caveat(n_periods=24, minimum_trusted=12) == ""
