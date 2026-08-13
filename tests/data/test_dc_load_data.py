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


def _queue_row(zone="PSEG", mw=500.0, submitted="1/1/2019", withdrawn=None, in_service=None):
    return {
        "zone": zone,
        "mw_capacity": mw,
        "submitted_date": pd.Timestamp(submitted),
        "withdrawal_date": pd.Timestamp(withdrawn) if withdrawn else pd.NaT,
        "actual_in_service_date": pd.Timestamp(in_service) if in_service else pd.NaT,
    }


def test_in_queue_true_before_submission_is_false():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="6/1/2020")])
    mask = in_queue(queue, pd.Timestamp("2020-01-01"))
    assert mask.tolist() == [False]


def test_in_queue_true_after_submission_before_exit():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="1/1/2019")])
    mask = in_queue(queue, pd.Timestamp("2020-01-01"))
    assert mask.tolist() == [True]


def test_in_queue_false_after_withdrawal():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="1/1/2019", withdrawn="6/1/2020")])
    assert in_queue(queue, pd.Timestamp("2020-01-01")).tolist() == [True]
    assert in_queue(queue, pd.Timestamp("2020-07-01")).tolist() == [False]


def test_in_queue_false_after_actual_in_service():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="1/1/2019", in_service="6/1/2020")])
    assert in_queue(queue, pd.Timestamp("2020-01-01")).tolist() == [True]
    assert in_queue(queue, pd.Timestamp("2020-07-01")).tolist() == [False]


def test_queued_mw_sums_only_matching_zone_and_active_projects():
    from grid_resilience.data.dc_load_data import queued_mw
    queue = pd.DataFrame([
        _queue_row(zone="PSEG", mw=500.0, submitted="1/1/2019"),
        _queue_row(zone="PSEG", mw=300.0, submitted="1/1/2019", withdrawn="6/1/2019"),
        _queue_row(zone="DOM", mw=1000.0, submitted="1/1/2019"),
    ])
    assert queued_mw(queue, "PSEG", pd.Timestamp("2020-01-01")) == 500.0


def test_new_mw_since_only_counts_recent_submissions():
    from grid_resilience.data.dc_load_data import new_mw_since
    queue = pd.DataFrame([
        _queue_row(zone="PSEG", mw=200.0, submitted="2020-01-15"),  # within window
        _queue_row(zone="PSEG", mw=800.0, submitted="2019-01-01"),  # outside window
    ])
    as_of = pd.Timestamp("2020-02-01")
    result = new_mw_since(queue, "PSEG", as_of, window_days=90)
    assert result == 200.0


def test_new_mw_since_excludes_projects_already_exited():
    from grid_resilience.data.dc_load_data import new_mw_since
    queue = pd.DataFrame([
        _queue_row(zone="PSEG", mw=200.0, submitted="2020-01-15", withdrawn="2020-01-20"),
    ])
    as_of = pd.Timestamp("2020-02-01")
    assert new_mw_since(queue, "PSEG", as_of, window_days=90) == 0.0
