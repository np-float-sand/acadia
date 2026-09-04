"""Tests for the per-auction-month FTR bids cache wrapper."""
from unittest.mock import patch

import pandas as pd

from grid_resilience.data import grid_data


def test_fetch_ftr_bids_by_month_caches_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(grid_data, "CACHE_DIR", tmp_path)
    rows = pd.DataFrame([{"sink_pnode_name": "AEP", "quoted_price": 10.0, "quoted_mw": 1.0}])

    with patch("grid_resilience.data.grid_data._fetch_pjm_ftr_bids_raw",
              return_value=rows) as mock_fetch, \
         patch.dict("os.environ", {"PJM_API_KEY": "test-key"}):
        df1 = grid_data.fetch_ftr_bids_by_month("JAN 2019 Auction")
    assert not df1.empty
    mock_fetch.assert_called_once()

    with patch("grid_resilience.data.grid_data._fetch_pjm_ftr_bids_raw") as mock_fetch2:
        df2 = grid_data.fetch_ftr_bids_by_month("JAN 2019 Auction")
    mock_fetch2.assert_not_called()
    pd.testing.assert_frame_equal(df1, df2)


def test_fetch_ftr_bids_by_month_uses_a_distinct_file_per_market_name(tmp_path, monkeypatch):
    monkeypatch.setattr(grid_data, "CACHE_DIR", tmp_path)
    jan_rows = pd.DataFrame([{"sink_pnode_name": "AEP", "quoted_price": 1.0, "quoted_mw": 1.0}])
    feb_rows = pd.DataFrame([{"sink_pnode_name": "AEP", "quoted_price": 2.0, "quoted_mw": 1.0}])

    with patch("grid_resilience.data.grid_data._fetch_pjm_ftr_bids_raw",
              side_effect=[jan_rows, feb_rows]):
        jan = grid_data.fetch_ftr_bids_by_month("JAN 2019 Auction")
        feb = grid_data.fetch_ftr_bids_by_month("FEB 2019 Auction")

    assert jan.iloc[0]["quoted_price"] == 1.0
    assert feb.iloc[0]["quoted_price"] == 2.0


def test_fetch_ftr_bids_by_month_does_not_cache_empty_result(tmp_path, monkeypatch):
    monkeypatch.setattr(grid_data, "CACHE_DIR", tmp_path)
    with patch("grid_resilience.data.grid_data._fetch_pjm_ftr_bids_raw",
              return_value=pd.DataFrame()) as mock_fetch:
        grid_data.fetch_ftr_bids_by_month("JAN 2019 Auction")

    with patch("grid_resilience.data.grid_data._fetch_pjm_ftr_bids_raw",
              return_value=pd.DataFrame()) as mock_fetch2:
        grid_data.fetch_ftr_bids_by_month("JAN 2019 Auction")
    # empty months (e.g. beyond the 4-month posting delay) must retry, not
    # be cached forever as a permanent empty result
    mock_fetch2.assert_called_once()
