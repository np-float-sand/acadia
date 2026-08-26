import pandas as pd
from grid_resilience.data.pjm_large_load_data import _parse_industry_tags, _parse_requests_sheet, cross_check_dc_signal


# ── Finding #4 (2026-08-26 whole-branch review): real LAS zone vocabulary ─────
#
# The PJM Large Load Adjustment Requests spreadsheet reports zones in PJM's own
# vocabulary, which MIXES full TICKER_NODE_MAP-style names with short codes —
# `DAY` for AEP's `DAYTON` zone. cross_check_dc_signal used to match
# `large_load["zone"]` raw against TICKER_NODE_MAP's load_zones, so every
# short-code row was silently dropped: AEP's mw_demand_2030 came out 9,296.2 MW
# instead of 13,996.2 MW, a 34% understatement. This list is the actual zone
# column of the live 2026 vintage, measured from the on-disk cache.
REAL_PJM_LAS_ZONE_VOCAB = [
    "AEP", "APS", "ATSI", "BGE", "COMED", "DAY",
    "DEOK", "DOM", "PECO", "PEPCO", "PPL", "PSEG",
]

# LAS zones that legitimately map to no ticker in TICKER_NODE_MAP: APS
# (Allegheny Power System) and DEOK (Duke Energy Ohio/Kentucky) have no
# universe ticker whose load_zones list them.
LAS_ZONES_WITH_NO_UNIVERSE_TICKER = {"APS", "DEOK"}


def test_parse_industry_tags_drops_header_repeat_row_and_keeps_only_zone_rows():
    raw = pd.DataFrame([
        ["ZONENAME", "AREANAME", "Capacity", "Demand", "industry", "note"],
        ["BGE", "BGE", "x", "x", "data center", None],
        ["AEP", "APCO", "x", None, "industrial & crypto", None],
    ])
    result = _parse_industry_tags(raw)
    assert list(result.columns) == ["zone", "area", "industry"]
    assert len(result) == 2
    assert result.iloc[0].to_dict() == {"zone": "BGE", "area": "BGE", "industry": "data center"}


def test_parse_industry_tags_strips_whitespace_from_industry():
    raw = pd.DataFrame([
        ["ZONENAME", "AREANAME", "Capacity", "Demand", "industry", "note"],
        ["BGE", "BGE", "x", "x", "data center  ", None],
        ["AEP", "APCO", "x", None, "  industrial & crypto", None],
        ["DUQ", "DUQ", "x", "x", "  data center  ", None],
    ])
    result = _parse_industry_tags(raw)
    assert result.iloc[0]["industry"] == "data center"
    assert result.iloc[1]["industry"] == "industrial & crypto"
    assert result.iloc[2]["industry"] == "data center"
    # Verify that the duplicate (with/without whitespace) is now normalized
    assert result[result["industry"] == "data center"].shape[0] == 2


def test_parse_requests_sheet_melts_year_columns_to_long_format():
    raw = pd.DataFrame([
        ["Total Demand Request for Large Load Adjustment"] + [None] * 5,
        ["Note..."] + [None] * 5,
        [None] * 6,
        [None, "ZONENAME", "AREANAME", 2025.0, 2026.0, 2027.0],
        [None, None, None, None, None, None],   # the stray no-data "AE" row
        [None, "BGE", "BGE", None, 17.756, 25.369],
    ])
    result = _parse_requests_sheet(raw, "mw_demand")
    assert set(result.columns) == {"zone", "area", "year", "mw_demand"}
    bge_2026 = result[(result["zone"] == "BGE") & (result["year"] == 2026)]
    assert bge_2026["mw_demand"].iloc[0] == 17.756
    assert not (result["zone"] == "AE").any()


def test_cross_check_flags_disagreement_when_industry_tag_is_not_data_center():
    dc_history = pd.DataFrame({"AEP": [0.8]}, index=[pd.Timestamp("2025-12-31")])
    large_load = pd.DataFrame([
        {"zone": "AEP", "area": "APCO", "industry": "industrial & crypto", "year": 2030, "mw_demand": 500.0, "mw_capacity": 600.0},
    ])
    node_map = {"AEP": {"iso": "PJM", "load_zones": ["AEP", "DAYTON"]}}

    result = cross_check_dc_signal(dc_history, large_load, node_map)
    row = result[result["ticker"] == "AEP"].iloc[0]
    assert row["agrees"] == False
    assert "industrial & crypto" in row["industries"]


def test_cross_check_agrees_when_industry_tag_is_data_center():
    dc_history = pd.DataFrame({"EXC": [0.5]}, index=[pd.Timestamp("2025-12-31")])
    large_load = pd.DataFrame([
        {"zone": "PECO", "area": "PECO", "industry": "data center", "year": 2030, "mw_demand": 800.0, "mw_capacity": 900.0},
    ])
    node_map = {"EXC": {"iso": "PJM", "load_zones": ["PECO", "BGE", "PEPCO", "COMED"]}}

    result = cross_check_dc_signal(dc_history, large_load, node_map)
    assert result[result["ticker"] == "EXC"].iloc[0]["agrees"] == True


def test_normalized_las_vocabulary_resolves_against_every_pjm_ticker_load_zone():
    """Every zone code in the real LAS vocabulary must, after normalization,
    either name a zone some PJM ticker actually holds or be one of the
    explicitly-known non-universe zones. Matching raw (as cross_check_dc_signal
    used to) leaves DAY unresolved and silently discards AEP's Dayton rows —
    this is the check that would have caught finding #4 immediately. Modelled
    on test_zonal_load.py::test_normalized_real_vocabulary_covers_every_pjm_ticker_load_zone."""
    from grid_resilience.data.grid_data import _normalize_pjm_load_zone
    from grid_resilience.data.utility_node_map import TICKER_NODE_MAP

    pjm_zone_names = set()
    for info in TICKER_NODE_MAP.values():
        if info.get("iso") == "PJM":
            pjm_zone_names.update(info.get("load_zones", []))

    unresolved = {
        z for z in REAL_PJM_LAS_ZONE_VOCAB
        if _normalize_pjm_load_zone(z) not in pjm_zone_names
    } - LAS_ZONES_WITH_NO_UNIVERSE_TICKER
    assert not unresolved, (
        f"LAS zone codes {unresolved} normalize to nothing in TICKER_NODE_MAP's "
        "PJM load_zones vocabulary — cross_check_dc_signal would silently drop "
        "those rows' MW and industry tags (finding #4)."
    )

    # And the specific case that was broken: DAY is AEP's Dayton zone.
    assert _normalize_pjm_load_zone("DAY") == "DAYTON"
    assert "DAYTON" in TICKER_NODE_MAP["AEP"]["load_zones"]


def test_cross_check_includes_short_code_zone_rows():
    """The regression itself: a `DAY` row must be counted toward AEP's
    mw_demand_2030 and its industry tags, not dropped for not literally
    equalling the 'DAYTON' name in TICKER_NODE_MAP."""
    dc_history = pd.DataFrame({"AEP": [0.8]}, index=[pd.Timestamp("2025-12-31")])
    large_load = pd.DataFrame([
        {"zone": "AEP", "area": "APCO", "industry": "industrial & crypto", "year": 2030, "mw_demand": 500.0, "mw_capacity": 600.0},
        {"zone": "DAY", "area": "DAY",  "industry": "data center",         "year": 2030, "mw_demand": 700.0, "mw_capacity": 800.0},
    ])
    node_map = {"AEP": {"iso": "PJM", "load_zones": ["AEP", "DAYTON"]}}

    row = cross_check_dc_signal(dc_history, large_load, node_map).iloc[0]
    assert row["mw_demand_2030"] == 1200.0
    assert row["industries"] == ["data center", "industrial & crypto"]
    assert row["agrees"] == True


def test_cross_check_data_center_tag_match_is_case_insensitive():
    """Finding #10 hardening: all live LAS industry tags are lowercase today,
    but a capitalized 'Data Center' must not silently read as a disagreement."""
    dc_history = pd.DataFrame({"PPL": [0.5]}, index=[pd.Timestamp("2025-12-31")])
    large_load = pd.DataFrame([
        {"zone": "PPL", "area": "PPL", "industry": "Data Center", "year": 2030, "mw_demand": 100.0, "mw_capacity": 100.0},
    ])
    node_map = {"PPL": {"iso": "PJM", "load_zones": ["PPL"]}}

    assert cross_check_dc_signal(dc_history, large_load, node_map).iloc[0]["agrees"] == True


def test_large_load_source_urls_has_exactly_one_vintage_today():
    """Finding #6: the results doc claimed 2 LAS vintages (Nov 2024, Nov 2025);
    the module ships exactly 1 (the 2025-09-16 posting, keyed '2026'). This
    pins the count so the doc and the code can't drift apart again — update
    both together when PJM posts the next vintage."""
    from grid_resilience.data.pjm_large_load_data import LARGE_LOAD_SOURCE_URLS
    assert list(LARGE_LOAD_SOURCE_URLS) == ["2026"]
