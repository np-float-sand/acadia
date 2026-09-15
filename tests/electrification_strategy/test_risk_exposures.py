import pandas as pd
import pytest

from electrification_strategy import config, risk_exposures as rx


def test_load_missing_file_returns_empty(tmp_path):
    df = rx.load_risk_exposures(tmp_path / "does_not_exist.csv")
    assert df.empty
    assert list(df.columns) == ["date", "ticker", "factor", "exposure"]


def test_load_sample_fixture():
    df = rx.load_risk_exposures(config.DATA_DIR / "risk_exposures_sample.csv")
    assert len(df) == 8
    assert df["date"].dtype == "datetime64[ns]"
    assert set(df["factor"]) == {"value", "momentum"}


def test_align_to_universe_picks_most_recent_at_or_before_lagged_asof():
    df = rx.load_risk_exposures(config.DATA_DIR / "risk_exposures_sample.csv")
    # asof 2025-05-01, lag 42 days -> effective date 2025-03-20 -> most recent
    # available row at/before that is 2024-12-31 (2025-03-31 is still in the future
    # relative to the lagged point-in-time date)
    aligned = rx.align_to_universe(df, ["ETN", "HUBB"], pd.Timestamp("2025-05-01"), lag_days=42)
    assert aligned.loc["ETN", "value"] == 0.42
    assert aligned.loc["HUBB", "momentum"] == 0.55


def test_align_to_universe_missing_ticker_is_nan():
    df = rx.load_risk_exposures(config.DATA_DIR / "risk_exposures_sample.csv")
    aligned = rx.align_to_universe(df, ["ETN", "GEV"], pd.Timestamp("2025-05-01"), lag_days=42)
    assert aligned.loc["GEV"].isna().all()


def test_portfolio_exposure_report_weighted_average():
    df = rx.load_risk_exposures(config.DATA_DIR / "risk_exposures_sample.csv")
    weights = pd.Series({"ETN": 0.6, "HUBB": 0.4})
    report = rx.portfolio_exposure_report(df, weights, pd.Timestamp("2025-05-01"), lag_days=42)
    assert report["value"] == pytest.approx(0.6 * 0.42 + 0.4 * 0.10)
    assert report["momentum"] == pytest.approx(0.6 * -0.15 + 0.4 * 0.55)
