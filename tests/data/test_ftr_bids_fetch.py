"""Tests for the PJM monthly-FTR-auction-bids fetcher (ftr_bids_mnt)."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from grid_resilience.data.grid_data import _fetch_pjm_ftr_bids_raw


def _mock_response(status_code: int, items=None, total_rows: int = 0):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = items if items is not None else []
    resp.headers = {"X-TotalRows": str(total_rows)}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


def test_sends_market_name_and_filter_params():
    ok = _mock_response(200, items=[], total_rows=0)
    with patch("requests.get", return_value=ok) as mock_get:
        _fetch_pjm_ftr_bids_raw("key", "JAN 2019 Auction")
    params = mock_get.call_args.kwargs["params"]
    assert params["market_name"] == "JAN 2019 Auction"
    assert params["trade_type"] == "Buy"
    assert params["hedge_type"] == "Obligation"
    assert params["class_type"] == "OnPeak"
    assert params["download"] == "true"


def test_single_page_returns_all_rows():
    rows = [{"sink_pnode_name": "AEP", "quoted_price": 10.0, "quoted_mw": 1.0}]
    ok = _mock_response(200, items=rows, total_rows=1)
    with patch("requests.get", return_value=ok):
        df = _fetch_pjm_ftr_bids_raw("key", "JAN 2019 Auction")
    assert len(df) == 1
    assert df.iloc[0]["sink_pnode_name"] == "AEP"


def test_paginates_across_multiple_pages():
    page1 = _mock_response(200, items=[{"sink_pnode_name": f"N{i}", "quoted_price": 1.0,
                                        "quoted_mw": 1.0} for i in range(50000)],
                           total_rows=60000)
    page2 = _mock_response(200, items=[{"sink_pnode_name": f"N{i}", "quoted_price": 1.0,
                                        "quoted_mw": 1.0} for i in range(10000)],
                           total_rows=60000)
    with patch("requests.get", side_effect=[page1, page2]) as mock_get:
        df = _fetch_pjm_ftr_bids_raw("key", "JAN 2019 Auction")
    assert len(df) == 60000
    assert mock_get.call_count == 2
    second_call_params = mock_get.call_args_list[1].kwargs["params"]
    assert second_call_params["startRow"] == 50001


def test_zero_total_rows_returns_empty_dataframe():
    ok = _mock_response(200, items=[], total_rows=0)
    with patch("requests.get", return_value=ok):
        df = _fetch_pjm_ftr_bids_raw("key", "JAN 2019 Auction")
    assert df.empty
