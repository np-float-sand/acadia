"""
Tests that _fetch_lmp_raw uses the correct parameter for each ISO.
Uses a mock iso_obj to intercept the call without hitting the network.
"""
import inspect
import pandas as pd
import pytest
from unittest.mock import MagicMock

from grid_resilience.data.grid_data import _fetch_lmp_raw


def _make_iso_obj(param_names):
    """Create a mock whose get_lmp() accepts only the given parameter names."""
    mock = MagicMock()
    params = {n: inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
              for n in param_names}
    mock.get_lmp.__signature__ = inspect.Signature(list(params.values()))
    mock.get_lmp.return_value = pd.DataFrame()
    return mock


def test_ercot_passes_location_type():
    iso_obj = _make_iso_obj(["date", "end", "location_type", "verbose"])
    _fetch_lmp_raw(iso_obj, "ERCOT", "2024-01-01", "2024-01-31", "settlement point")
    call_kwargs = iso_obj.get_lmp.call_args.kwargs
    assert "location_type" in call_kwargs
    assert "market" not in call_kwargs


def test_miso_passes_market_param():
    iso_obj = _make_iso_obj(["date", "end", "market", "locations", "verbose"])
    _fetch_lmp_raw(iso_obj, "MISO", "2024-01-01", "2024-01-31", "LMP")
    call_kwargs = iso_obj.get_lmp.call_args.kwargs
    assert "location_type" not in call_kwargs
    assert call_kwargs.get("market") == "LMP"


def test_caiso_omits_location_type():
    iso_obj = _make_iso_obj(["date", "market", "locations", "sleep", "end", "verbose"])
    _fetch_lmp_raw(iso_obj, "CAISO", "2024-01-01", "2024-01-31", "trading_hub")
    call_kwargs = iso_obj.get_lmp.call_args.kwargs
    assert "location_type" not in call_kwargs
    assert call_kwargs.get("market") == "trading_hub"


def test_empty_return_on_exception():
    """If get_lmp raises, _fetch_lmp_raw returns an empty DataFrame (no re-raise)."""
    iso_obj = _make_iso_obj(["date", "end", "location_type", "verbose"])
    iso_obj.get_lmp.side_effect = RuntimeError("network error")
    result = _fetch_lmp_raw(iso_obj, "ERCOT", "2024-01-01", "2024-01-31", "hub")
    assert result.empty
