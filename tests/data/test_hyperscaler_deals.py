import pandas as pd


def test_load_hyperscaler_deals_excludes_unverified_by_default(tmp_path):
    from grid_resilience.data.hyperscaler_deals import load_hyperscaler_deals
    csv_path = tmp_path / "deals.csv"
    csv_path.write_text(
        "ticker,counterparty,mw,sign,event_type,first_disclosure_date,date_confidence,source_url,note\n"
        "CEG,Microsoft,835,1,announcement,2024-09-20,confirmed_primary_source,x,\n"
        "TLN,Amazon,1920,1,announcement,2024-03-01,needs_verification,x,\n"
    )
    result = load_hyperscaler_deals(path=csv_path)
    assert list(result["ticker"]) == ["CEG"]

    result_all = load_hyperscaler_deals(path=csv_path, include_unverified=True)
    assert set(result_all["ticker"]) == {"CEG", "TLN"}


def test_compute_hyperscaler_signal_cumulative_sum_with_setback_and_resolution():
    from grid_resilience.data.hyperscaler_deals import compute_hyperscaler_signal
    deals = pd.DataFrame([
        {"ticker": "TLN", "mw": 1920, "sign": 1, "first_disclosure_date": pd.Timestamp("2024-03-01")},
        {"ticker": "TLN", "mw": 1920, "sign": -1, "first_disclosure_date": pd.Timestamp("2024-11-04")},
        {"ticker": "TLN", "mw": 1920, "sign": 1, "first_disclosure_date": pd.Timestamp("2025-06-11")},
    ])
    result = compute_hyperscaler_signal(
        tickers=["TLN"], deals=deals,
        as_of_dates=[pd.Timestamp("2024-06-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")],
    )
    assert result.loc[pd.Timestamp("2024-06-01"), "TLN"] == 1920
    assert result.loc[pd.Timestamp("2025-01-01"), "TLN"] == 0
    assert result.loc[pd.Timestamp("2025-12-31"), "TLN"] == 1920


def test_compute_hyperscaler_signal_uncovered_ticker_is_nan():
    from grid_resilience.data.hyperscaler_deals import compute_hyperscaler_signal
    deals = pd.DataFrame(columns=["ticker", "mw", "sign", "first_disclosure_date"])
    result = compute_hyperscaler_signal(tickers=["FE"], deals=deals, as_of_dates=[pd.Timestamp("2025-01-01")])
    assert pd.isna(result.loc[pd.Timestamp("2025-01-01"), "FE"])


def test_compute_hyperscaler_signal_zero_before_first_disclosure():
    from grid_resilience.data.hyperscaler_deals import compute_hyperscaler_signal
    deals = pd.DataFrame([
        {"ticker": "TLN", "mw": 1920, "sign": 1, "first_disclosure_date": pd.Timestamp("2024-03-04")},
    ])
    result = compute_hyperscaler_signal(
        tickers=["TLN"], deals=deals,
        as_of_dates=[pd.Timestamp("2024-01-01")],
    )
    # Ticker is in deals (covered by layer), but as_of_date is before first disclosure,
    # so exposure is 0.0 (not NaN which would mean uncovered/unmeasured)
    assert result.loc[pd.Timestamp("2024-01-01"), "TLN"] == 0.0
