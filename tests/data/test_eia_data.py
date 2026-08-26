from unittest.mock import patch, MagicMock

import pandas as pd

from grid_resilience.data.eia_data import fetch_plant_capacity


def _mock_response(rows):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"response": {"data": rows}}
    return resp


def test_fetch_plant_capacity_calls_correct_eia_route(tmp_path, monkeypatch):
    """fetch_plant_capacity() must hit the real EIA v2 dataset id
    'operating-generator-capacity', not the dead 'operating-generator/generator' route
    (confirmed 404 live against the EIA API)."""
    from grid_resilience.data import eia_data
    monkeypatch.setattr(eia_data, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("EIA_API_KEY", "test-key")

    mock_get = MagicMock(return_value=_mock_response([]))
    with patch("grid_resilience.data.eia_data.requests.get", mock_get):
        fetch_plant_capacity("ERCOT", year=2019, use_cache=False)

    called_url = mock_get.call_args.args[0]
    assert called_url == (
        "https://api.eia.gov/v2/electricity/operating-generator-capacity/data/"
    )


def test_fetch_plant_capacity_uses_monthly_frequency_and_month_range(tmp_path, monkeypatch):
    """The dataset only supports frequency=monthly (confirmed live: frequency=annual
    returns HTTP 400), and start/end must be YYYY-MM, not a bare year."""
    from grid_resilience.data import eia_data
    monkeypatch.setattr(eia_data, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("EIA_API_KEY", "test-key")

    mock_get = MagicMock(return_value=_mock_response([]))
    with patch("grid_resilience.data.eia_data.requests.get", mock_get):
        fetch_plant_capacity("ERCOT", year=2019, use_cache=False)

    call_params = mock_get.call_args.kwargs["params"]
    assert call_params["frequency"] == "monthly"
    assert call_params["start"] == "2019-01"
    assert call_params["end"] == "2019-12"


def test_fetch_plant_capacity_paginates_past_page_cap(tmp_path, monkeypatch):
    """A full year for one BA can exceed the API's per-request row cap (confirmed
    live: ~10,600 rows/year for ERCOT alone vs. a 5000-row cap) — must page via
    offset until a short page signals the end, not silently truncate."""
    from grid_resilience.data import eia_data
    monkeypatch.setattr(eia_data, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(eia_data, "_EIA_PAGE_LENGTH", 2)
    monkeypatch.setenv("EIA_API_KEY", "test-key")

    page1 = [{"plantid": "1", "nameplate-capacity-mw": "10"},
             {"plantid": "2", "nameplate-capacity-mw": "20"}]
    page2 = [{"plantid": "3", "nameplate-capacity-mw": "30"}]
    mock_get = MagicMock(side_effect=[_mock_response(page1), _mock_response(page2)])

    with patch("grid_resilience.data.eia_data.requests.get", mock_get):
        df = fetch_plant_capacity("ERCOT", year=2019, use_cache=False)

    assert sorted(df["plantid"].tolist()) == ["1", "2", "3"]
    assert mock_get.call_args_list[0].kwargs["params"]["offset"] == 0
    assert mock_get.call_args_list[1].kwargs["params"]["offset"] == 2


def test_fetch_plant_capacity_returns_rows(tmp_path, monkeypatch):
    from grid_resilience.data import eia_data
    monkeypatch.setattr(eia_data, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("EIA_API_KEY", "test-key")

    rows = [
        {
            "period": "2019-06", "plantid": "298", "plantName": "Limestone",
            "entityName": "NRG Texas Power LLC", "status": "OP",
            "nameplate-capacity-mw": "893",
        },
    ]
    with patch(
        "grid_resilience.data.eia_data.requests.get",
        return_value=_mock_response(rows),
    ):
        df = fetch_plant_capacity("ERCOT", year=2019, use_cache=False)

    assert len(df) == 1
    assert df.iloc[0]["entityName"] == "NRG Texas Power LLC"
    assert df.iloc[0]["ba_code"] == "ERCO"
