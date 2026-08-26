from unittest.mock import patch

import pandas as pd

from grid_resilience.data.equity_prices import _download_with_retry


def _frame(cols, n=3):
    return pd.DataFrame({c: [1.0] * n for c in cols}, index=pd.date_range("2024-01-02", periods=n))


def test_retries_once_when_ticker_missing_from_first_attempt():
    """yfinance can silently drop a ticker under concurrent load without raising —
    a second attempt that comes back complete should be used."""
    incomplete = _frame(["VST", "NRG"])          # CEG silently dropped
    complete = _frame(["VST", "NRG", "CEG"])
    with patch(
        "grid_resilience.data.equity_prices._download",
        side_effect=[incomplete, complete],
    ) as mock_dl, patch("grid_resilience.data.equity_prices.time.sleep"):
        result = _download_with_retry(["VST", "NRG", "CEG"], "2024-01-01", "2024-01-31")

    assert mock_dl.call_count == 2
    assert set(result.columns) == {"VST", "NRG", "CEG"}


def test_no_retry_when_first_attempt_already_complete():
    complete = _frame(["VST", "NRG", "CEG"])
    with patch(
        "grid_resilience.data.equity_prices._download", side_effect=[complete]
    ) as mock_dl, patch("grid_resilience.data.equity_prices.time.sleep") as mock_sleep:
        result = _download_with_retry(["VST", "NRG", "CEG"], "2024-01-01", "2024-01-31")

    assert mock_dl.call_count == 1
    mock_sleep.assert_not_called()
    assert set(result.columns) == {"VST", "NRG", "CEG"}


def test_gives_up_after_max_attempts_for_a_genuinely_pre_listing_ticker():
    """A ticker with no data all month (e.g. before its IPO/spinoff date) will
    never come back no matter how many times it's retried — must stop after
    max_attempts rather than looping, and return the best available result."""
    always_missing = _frame(["VST", "NRG"])  # CEG never appears — legitimate gap
    with patch(
        "grid_resilience.data.equity_prices._download",
        side_effect=[always_missing, always_missing],
    ) as mock_dl, patch("grid_resilience.data.equity_prices.time.sleep"):
        result = _download_with_retry(
            ["VST", "NRG", "CEG"], "2018-01-01", "2018-01-31", max_attempts=2
        )

    assert mock_dl.call_count == 2
    assert set(result.columns) == {"VST", "NRG"}
