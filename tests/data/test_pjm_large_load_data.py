import pandas as pd
from grid_resilience.data.pjm_large_load_data import _parse_industry_tags, _parse_requests_sheet, cross_check_dc_signal


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
