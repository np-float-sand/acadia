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


# ── Finding #1 / #11 (2026-08-13 review): real PJM short-code vocabulary ──────
#
# PJM's hrl_load_metered feed reports zones under short codes, not
# TICKER_NODE_MAP's zone names. The original tests above (and the original
# compute_dc_load_signal tests) used TICKER_NODE_MAP-style names for BOTH the
# queue fixture and the zonal-load fixture, so they matched by construction
# and could never catch a zone-vocabulary mismatch. These tests pin the real
# vocabulary, measured against the actual cached hrl_load_metered data as of
# the 2026-08-13 review.
REAL_PJM_HRL_LOAD_METERED_ZONE_VOCAB = [
    "AE", "AEP", "AP", "ATSI", "BC", "CE", "DAY", "DEOK", "DOM", "DPL", "DUQ",
    "EKPC", "JC", "ME", "PE", "PEP", "PL", "PN", "PS", "RECO", "RTO",
]


def test_pjm_load_zone_aliases_match_verified_mapping():
    """Each alias must map the real short code to the exact TICKER_NODE_MAP
    zone name it was verified against (see review's cross-checked table)."""
    from grid_resilience.data.grid_data import _normalize_pjm_load_zone
    expected = {
        "PS":  "PSEG",
        "PL":  "PPL",
        "PE":  "PECO",
        "BC":  "BGE",
        "PEP": "PEPCO",
        "CE":  "COMED",
        "JC":  "JCPL",
        "DAY": "DAYTON",
        "ME":  "METED",
        "PN":  "PENELEC",
        # Already match TICKER_NODE_MAP directly — no alias needed.
        "AEP":  "AEP",
        "ATSI": "ATSI",
        "DOM":  "DOM",
    }
    for short_code, expected_zone in expected.items():
        assert _normalize_pjm_load_zone(short_code) == expected_zone


def test_normalized_real_vocabulary_covers_every_pjm_ticker_load_zone():
    """Every TICKER_NODE_MAP zone name used by a PJM ticker must be reachable
    by normalizing some code in the real hrl_load_metered vocabulary — this
    is exactly the check that would have caught finding #1 immediately."""
    from grid_resilience.data.grid_data import _normalize_pjm_load_zone
    from grid_resilience.data.utility_node_map import TICKER_NODE_MAP

    normalized_vocab = {
        _normalize_pjm_load_zone(z) for z in REAL_PJM_HRL_LOAD_METERED_ZONE_VOCAB
    }

    pjm_zone_names = set()
    for info in TICKER_NODE_MAP.values():
        if info.get("iso") == "PJM":
            pjm_zone_names.update(info.get("load_zones", []))

    missing = pjm_zone_names - normalized_vocab
    assert not missing, (
        f"TICKER_NODE_MAP zone names {missing} have no PJM hrl_load_metered "
        "short code that normalizes to them — the DC load signal would "
        "silently discard that zone's queue MW (finding #1)."
    )


def test_fetch_zonal_load_normalizes_short_codes_from_disk_cache(tmp_path, monkeypatch):
    """The normalization must apply to data already sitting in the on-disk
    cache under PJM's raw short codes (from before this fix), not just to
    freshly-fetched data — otherwise stale cache silently keeps the bug."""
    from grid_resilience.data import grid_data

    monkeypatch.setattr(grid_data, "CACHE_DIR", tmp_path)
    cache_base = grid_data._cache_path("PJM", "load_zonal")

    stale = pd.DataFrame([
        {"time": pd.Timestamp("2024-06-01T04:00:00"), "zone": "PS", "load_mw": 100.0},
        {"time": pd.Timestamp("2024-06-01T04:00:00"), "zone": "PL", "load_mw": 50.0},
    ])
    grid_data.save_monthly_chunks(stale, cache_base, "time")

    with patch("grid_resilience.data.grid_data._pjm_get") as mock_get:
        df = grid_data.fetch_zonal_load("2024-06-01", "2024-06-30")
    mock_get.assert_not_called()  # month already cached, no re-fetch needed

    assert set(df["zone"]) == {"PSEG", "PPL"}
