"""Tests for PJM rate-limit retry and inter-month delay."""
import time
from unittest.mock import MagicMock, patch, call
import pytest
import requests

from grid_resilience.data.grid_data import _pjm_get


def _mock_response(status_code: int, json_data: dict = None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {"items": [], "totalRows": 0}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


def test_pjm_get_succeeds_on_first_try():
    ok = _mock_response(200, {"items": [{"x": 1}], "totalRows": 1})
    with patch("requests.get", return_value=ok) as mock_get:
        result = _pjm_get("https://api.pjm.com/test", {}, "key")
    assert result == {"items": [{"x": 1}], "totalRows": 1}
    assert mock_get.call_count == 1


def test_pjm_get_retries_on_429_then_succeeds():
    rate_limited = _mock_response(429)
    ok = _mock_response(200, {"items": [], "totalRows": 0})
    with patch("requests.get", side_effect=[rate_limited, rate_limited, ok]):
        with patch("time.sleep") as mock_sleep:
            result = _pjm_get("https://api.pjm.com/test", {}, "key", retries=6)
    assert result == {"items": [], "totalRows": 0}
    # Must have slept at least twice (once per 429)
    assert mock_sleep.call_count >= 2


def test_pjm_get_first_sleep_is_at_least_15s():
    """Initial backoff must be >= 15s (was 5s — caused serial 429 floods)."""
    rate_limited = _mock_response(429)
    ok = _mock_response(200, {"items": [], "totalRows": 0})
    with patch("requests.get", side_effect=[rate_limited, ok]):
        with patch("time.sleep") as mock_sleep:
            _pjm_get("https://api.pjm.com/test", {}, "key", retries=6)
    first_sleep = mock_sleep.call_args_list[0][0][0]
    assert first_sleep >= 15, f"Expected first sleep >= 15s, got {first_sleep}s"


def test_pjm_get_raises_after_all_retries_exhausted():
    rate_limited = _mock_response(429)
    with patch("requests.get", return_value=rate_limited):
        with patch("time.sleep"):
            with pytest.raises(requests.HTTPError):
                _pjm_get("https://api.pjm.com/test", {}, "key", retries=2)


def test_pjm_get_return_response_true_returns_response_object_not_json():
    """Some newer-envelope feeds (e.g. ftr_bids_mnt with download=true) need
    response headers (X-TotalRows) that .json() alone discards."""
    ok = _mock_response(200, {"items": [{"x": 1}], "totalRows": 1})
    with patch("requests.get", return_value=ok):
        result = _pjm_get("https://api.pjm.com/test", {}, "key", return_response=True)
    assert result is ok
    ok.raise_for_status.assert_called_once()


def test_pjm_get_return_response_still_retries_on_429():
    rate_limited = _mock_response(429)
    ok = _mock_response(200, {"items": [], "totalRows": 0})
    with patch("requests.get", side_effect=[rate_limited, ok]):
        with patch("time.sleep"):
            result = _pjm_get("https://api.pjm.com/test", {}, "key", return_response=True)
    assert result is ok


def test_pjm_get_retries_on_read_timeout_then_succeeds():
    """A large 50k-row page can hit a transient read timeout mid-download
    (seen live during the FTR-bids backfill) -- this must retry like a 429,
    not crash the whole multi-month fetch."""
    ok = _mock_response(200, {"items": [], "totalRows": 0})
    with patch("requests.get", side_effect=[requests.exceptions.ReadTimeout("timed out"), ok]):
        with patch("time.sleep") as mock_sleep:
            result = _pjm_get("https://api.pjm.com/test", {}, "key")
    assert result == {"items": [], "totalRows": 0}
    assert mock_sleep.call_count >= 1


def test_pjm_get_retries_on_connection_error_then_succeeds():
    ok = _mock_response(200, {"items": [], "totalRows": 0})
    with patch("requests.get", side_effect=[requests.exceptions.ConnectionError("reset"), ok]):
        with patch("time.sleep"):
            result = _pjm_get("https://api.pjm.com/test", {}, "key")
    assert result == {"items": [], "totalRows": 0}


def test_pjm_get_raises_after_all_retries_exhausted_on_timeout():
    with patch("requests.get", side_effect=requests.exceptions.ReadTimeout("timed out")):
        with patch("time.sleep"):
            with pytest.raises(requests.exceptions.ReadTimeout):
                _pjm_get("https://api.pjm.com/test", {}, "key", retries=2)
