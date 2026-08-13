import pandas as pd
from unittest.mock import patch
from grid_resilience.data.grid_data import _fetch_pjm_load_by_zone_direct


def _mock_pjm_get_response(items):
    return {"items": items, "totalRows": len(items)}


def test_fetch_pjm_load_by_zone_direct_keeps_zone_column():
    items = [
        {"datetime_beginning_utc": "2024-06-01T04:00:00", "zone": "AEP", "mw": 3160.712},
        {"datetime_beginning_utc": "2024-06-01T04:00:00", "zone": "DOM", "mw": 10942.767},
        {"datetime_beginning_utc": "2024-06-01T05:00:00", "zone": "AEP", "mw": 3100.0},
    ]
    with patch(
        "grid_resilience.data.grid_data._pjm_get",
        return_value=_mock_pjm_get_response(items),
    ):
        df = _fetch_pjm_load_by_zone_direct("key", "2024-06-01", "2024-06-02")

    assert set(df.columns) == {"time", "zone", "load_mw"}
    assert len(df) == 3
    aep_rows = df[df["zone"] == "AEP"]
    assert len(aep_rows) == 2
    assert aep_rows["load_mw"].tolist() == [3160.712, 3100.0]


def test_fetch_pjm_load_by_zone_direct_empty_response():
    with patch(
        "grid_resilience.data.grid_data._pjm_get",
        return_value=_mock_pjm_get_response([]),
    ):
        df = _fetch_pjm_load_by_zone_direct("key", "2024-06-01", "2024-06-02")
    assert df.empty


def test_fetch_zonal_load_caches_to_disk(tmp_path, monkeypatch):
    from grid_resilience.data import grid_data
    monkeypatch.setattr(grid_data, "CACHE_DIR", tmp_path)

    items = [{"datetime_beginning_utc": "2024-06-01T04:00:00", "zone": "AEP", "mw": 100.0}]
    with patch(
        "grid_resilience.data.grid_data._pjm_get",
        return_value=_mock_pjm_get_response(items),
    ), patch.dict("os.environ", {"PJM_API_KEY": "test-key"}):
        df1 = grid_data.fetch_zonal_load("2024-06-01", "2024-06-30")
    assert not df1.empty

    cache_file = grid_data._cache_path("PJM", "load_zonal")
    chunk = grid_data.chunk_path(cache_file, 2024, 6)
    assert chunk.exists()

    with patch("grid_resilience.data.grid_data._pjm_get") as mock_get:
        df2 = grid_data.fetch_zonal_load("2024-06-01", "2024-06-30")
    mock_get.assert_not_called()
    assert len(df2) == len(df1)
