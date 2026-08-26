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


def test_compute_hyperscaler_signal_nan_before_first_disclosure():
    """SEMANTICS CHANGED 2026-08-26 (whole-branch review finding #3).

    This test previously asserted 0.0 before a covered ticker's first
    disclosure, on the reasoning that "the ticker is in `deals`, so the
    layer covers it". That conflated two different things:

      * "this layer measured the ticker and the answer is zero exposure"
      * "this layer had nothing to say about the ticker on this date"

    Only the first is a 0.0. Returning 0.0 for the second made
    apply_layer_precedence() — which claims a (date, ticker) cell for the
    highest-precedence layer that has a non-NaN value there — hand the
    hyperscaler layer every rebalance date back to 2018 for CEG/TLN, on
    the strength of deals not announced until 2024, and thereby DISPLACE
    the PJM generation-queue layer's real, dispersed data for that whole
    pre-disclosure period. NaN is the honest answer and lets the lower
    layer supply the cell.

    See test_compute_hyperscaler_signal_zero_when_disclosed_sum_nets_to_zero
    below for the case that must still return a real 0.0.
    """
    from grid_resilience.data.hyperscaler_deals import compute_hyperscaler_signal
    deals = pd.DataFrame([
        {"ticker": "TLN", "mw": 1920, "sign": 1, "first_disclosure_date": pd.Timestamp("2024-03-04")},
    ])
    result = compute_hyperscaler_signal(
        tickers=["TLN"], deals=deals,
        as_of_dates=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01")],
    )
    assert pd.isna(result.loc[pd.Timestamp("2024-01-01"), "TLN"])
    # …and on/after the first disclosure the layer does have information.
    assert result.loc[pd.Timestamp("2024-06-01"), "TLN"] == 1920.0


def test_compute_hyperscaler_signal_zero_when_disclosed_sum_nets_to_zero():
    """The other half of the NaN/0.0 distinction: once a ticker HAS a
    disclosed deal, a running sign*mw sum that nets to exactly zero (a
    setback fully offsetting an announcement) is a real measurement of
    zero net contracted MW and must stay 0.0, not become NaN."""
    from grid_resilience.data.hyperscaler_deals import compute_hyperscaler_signal
    deals = pd.DataFrame([
        {"ticker": "TLN", "mw": 1920, "sign": 1, "first_disclosure_date": pd.Timestamp("2024-03-01")},
        {"ticker": "TLN", "mw": 1920, "sign": -1, "first_disclosure_date": pd.Timestamp("2024-11-04")},
    ])
    result = compute_hyperscaler_signal(
        tickers=["TLN"], deals=deals,
        as_of_dates=[pd.Timestamp("2024-02-01"), pd.Timestamp("2025-01-01")],
    )
    assert pd.isna(result.loc[pd.Timestamp("2024-02-01"), "TLN"])   # pre-disclosure → no information
    assert result.loc[pd.Timestamp("2025-01-01"), "TLN"] == 0.0     # disclosed, nets to zero → measured 0


def test_compute_hyperscaler_signal_vst_only_deal_is_outside_backtest_window():
    """VST's single disclosed deal (2026-01-09) postdates config.BACKTEST_END
    (2025-12-31), so across the whole backtest window the hyperscaler layer
    has NO information about VST. With the corrected NaN semantics that is
    an all-NaN column, which apply_layer_precedence() will not claim — so
    VST correctly falls through to the ICR fallback instead of being pinned
    to the bottom of the {CEG,TLN,VST} z-score range for a period where it
    has zero information (whole-branch review finding #2)."""
    from grid_resilience.config import BACKTEST_END
    from grid_resilience.data.hyperscaler_deals import (
        compute_hyperscaler_signal, load_hyperscaler_deals,
    )
    deals = load_hyperscaler_deals()
    vst_first = deals[deals["ticker"] == "VST"]["first_disclosure_date"].min()
    assert vst_first > pd.Timestamp(BACKTEST_END)

    dates = list(pd.date_range("2018-01-31", BACKTEST_END, freq="ME"))
    result = compute_hyperscaler_signal(tickers=["VST"], deals=deals, as_of_dates=dates)
    assert result["VST"].isna().all()
