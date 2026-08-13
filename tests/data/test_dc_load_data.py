import pandas as pd
from grid_resilience.data.dc_load_data import normalize_transmission_owner


def test_normalize_transmission_owner_direct_match():
    assert normalize_transmission_owner("AEP") == "AEP"
    assert normalize_transmission_owner("PSEG") == "PSEG"
    assert normalize_transmission_owner("PECO") == "PECO"


def test_normalize_transmission_owner_case_insensitive():
    assert normalize_transmission_owner("Dayton") == "DAYTON"
    assert normalize_transmission_owner("ComEd") == "COMED"


def test_normalize_transmission_owner_dominion_override():
    assert normalize_transmission_owner("Dominion") == "DOM"


def test_normalize_transmission_owner_strips_semicolon_duplicates():
    assert normalize_transmission_owner("PSEG; PSEG") == "PSEG"


def test_normalize_transmission_owner_handles_nan():
    assert normalize_transmission_owner(float("nan")) is None


def _raw_queue_row(**overrides):
    row = {
        "Transmission Owner": "PSEG",
        "MW Capacity": 500.0,
        "Project Type": "Generation Interconnection",
        "Status": "Active",
        "Submitted Date": "1/1/2019",
        "Withdrawal Date": None,
        "Actual In Service Date": None,
    }
    row.update(overrides)
    return row


def test_clean_queue_keeps_valid_generation_rows():
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row()])
    cleaned = _clean_queue(raw)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["zone"] == "PSEG"
    assert cleaned.iloc[0]["mw_capacity"] == 500.0
    assert pd.notna(cleaned.iloc[0]["submitted_date"])


def test_clean_queue_drops_sub_threshold_mw():
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row(**{"MW Capacity": 50.0})])
    cleaned = _clean_queue(raw)
    assert cleaned.empty


def test_clean_queue_drops_non_generation_project_types():
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row(**{"Project Type": "Merchant Transmission"})])
    cleaned = _clean_queue(raw)
    assert cleaned.empty


def test_clean_queue_drops_unmappable_transmission_owner():
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row(**{"Transmission Owner": None})])
    cleaned = _clean_queue(raw)
    assert cleaned.empty


def test_clean_queue_does_not_filter_by_status():
    """Status is a current-snapshot field with no date attached — filtering by
    it would corrupt the point-in-time reconstruction (see design spec).
    A withdrawn project must still appear in the cleaned queue; date-based
    filtering happens later in queued_mw()/in_queue()."""
    from grid_resilience.data.dc_load_data import _clean_queue
    raw = pd.DataFrame([_raw_queue_row(**{
        "Status": "Withdrawn",
        "Withdrawal Date": "6/1/2020",
    })])
    cleaned = _clean_queue(raw)
    assert len(cleaned) == 1
