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


def test_in_queue_true_exactly_on_submission_date():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="2020-01-01")])
    mask = in_queue(queue, pd.Timestamp("2020-01-01"))
    assert mask.tolist() == [True]


def test_in_queue_false_exactly_on_withdrawal_date():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="2019-01-01", withdrawn="2020-06-01")])
    mask = in_queue(queue, pd.Timestamp("2020-06-01"))
    assert mask.tolist() == [False]


def test_in_queue_false_exactly_on_actual_in_service_date():
    from grid_resilience.data.dc_load_data import in_queue
    queue = pd.DataFrame([_queue_row(submitted="2019-01-01", in_service="2020-06-01")])
    mask = in_queue(queue, pd.Timestamp("2020-06-01"))
    assert mask.tolist() == [False]


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


def _zonal_load_row(zone, time, mw):
    return {"time": pd.Timestamp(time), "zone": zone, "load_mw": mw}


def test_zone_size_averages_trailing_12_months_single_zone():
    from grid_resilience.data.dc_load_data import zone_size
    rows = [_zonal_load_row("PSEG", f"2019-{m:02d}-15", 1000.0 + m) for m in range(1, 13)]
    zonal_load = pd.DataFrame(rows)
    result = zone_size(zonal_load, ["PSEG"], pd.Timestamp("2020-01-01"))
    expected = sum(1000.0 + m for m in range(1, 13)) / 12
    assert abs(result - expected) < 1e-6


def test_zone_size_sums_multiple_zones():
    from grid_resilience.data.dc_load_data import zone_size
    zonal_load = pd.DataFrame([
        _zonal_load_row("PECO", "2019-06-15", 1000.0),
        _zonal_load_row("BGE", "2019-06-15", 500.0),
    ])
    result = zone_size(zonal_load, ["PECO", "BGE"], pd.Timestamp("2019-12-01"))
    assert result == 1500.0


def test_zone_size_excludes_data_after_as_of():
    from grid_resilience.data.dc_load_data import zone_size
    zonal_load = pd.DataFrame([
        _zonal_load_row("PSEG", "2019-06-15", 1000.0),
        _zonal_load_row("PSEG", "2020-06-15", 9000.0),  # future — must not leak in
    ])
    result = zone_size(zonal_load, ["PSEG"], pd.Timestamp("2019-12-01"))
    assert result == 1000.0


def test_zone_size_returns_nan_when_no_data():
    from grid_resilience.data.dc_load_data import zone_size
    zonal_load = pd.DataFrame(columns=["time", "zone", "load_mw"])
    result = zone_size(zonal_load, ["PSEG"], pd.Timestamp("2019-12-01"))
    assert pd.isna(result)


def _node_map(**tickers):
    return tickers


def test_compute_dc_load_signal_single_pjm_ticker():
    from grid_resilience.data.dc_load_data import compute_dc_load_signal
    queue = pd.DataFrame([_queue_row(zone="PSEG", mw=1000.0, submitted="2019-01-01")])
    zonal_load = pd.DataFrame([_zonal_load_row("PSEG", "2019-06-15", 5000.0)])
    node_map = _node_map(PEG={"iso": "PJM", "load_zones": ["PSEG"]})

    result = compute_dc_load_signal(
        tickers=["PEG"], node_map=node_map, queue=queue, zonal_load=zonal_load,
        as_of_dates=[pd.Timestamp("2020-01-01")],
    )
    assert list(result.columns) == ["PEG"]
    # level = 1000/5000 = 0.2, momentum = 0 (submitted well outside 90d window)
    assert abs(result.loc[pd.Timestamp("2020-01-01"), "PEG"] - (0.6 * 0.2 + 0.4 * 0.0)) < 1e-6


def test_compute_dc_load_signal_non_pjm_ticker_is_nan():
    from grid_resilience.data.dc_load_data import compute_dc_load_signal
    queue = pd.DataFrame(columns=["zone", "mw_capacity", "submitted_date", "withdrawal_date", "actual_in_service_date"])
    zonal_load = pd.DataFrame(columns=["time", "zone", "load_mw"])
    node_map = _node_map(VST={"iso": "ERCOT", "load_zones": ["NORTH"]})

    result = compute_dc_load_signal(
        tickers=["VST"], node_map=node_map, queue=queue, zonal_load=zonal_load,
        as_of_dates=[pd.Timestamp("2020-01-01")],
    )
    assert pd.isna(result.loc[pd.Timestamp("2020-01-01"), "VST"])


def test_compute_dc_load_signal_multi_zone_ticker_sums_zones():
    from grid_resilience.data.dc_load_data import compute_dc_load_signal
    queue = pd.DataFrame([
        _queue_row(zone="PECO", mw=400.0, submitted="2019-01-01"),
        _queue_row(zone="BGE", mw=100.0, submitted="2019-01-01"),
    ])
    zonal_load = pd.DataFrame([
        _zonal_load_row("PECO", "2019-06-15", 2000.0),
        _zonal_load_row("BGE", "2019-06-15", 500.0),
    ])
    node_map = _node_map(EXC={"iso": "PJM", "load_zones": ["PECO", "BGE"]})

    result = compute_dc_load_signal(
        tickers=["EXC"], node_map=node_map, queue=queue, zonal_load=zonal_load,
        as_of_dates=[pd.Timestamp("2020-01-01")],
    )
    # level = (400+100)/(2000+500) = 0.2
    assert abs(result.loc[pd.Timestamp("2020-01-01"), "EXC"] - 0.6 * 0.2) < 1e-6


def test_fill_with_icr_fills_nan_columns_from_most_recent_icr():
    """Fills the NaN cell from ICR (not left NaN) and doesn't touch a cell
    that already had real DC data. Post-2026-08-13-fix, both cells are
    z-scored within their own (single-member) subpopulation, so the exact
    value is 0.0 rather than a raw passthrough — see
    test_fill_with_icr_zscores_dc_and_icr_subpopulations_separately for a
    multi-member-subpopulation check of the ordering/scale contract."""
    from grid_resilience.data.dc_load_data import fill_with_icr
    dc_history = pd.DataFrame(
        {"PEG": [0.3], "WEC": [float("nan")]},
        index=[pd.Timestamp("2020-06-01")],
    )
    icr_history = pd.DataFrame(
        {"WEC": [4.0]},
        index=[pd.Timestamp("2020-03-01")],  # 92 days before as_of, past a 45d lag
    )
    result = fill_with_icr(dc_history, icr_history, lag_days=45)
    assert pd.notna(result.loc[pd.Timestamp("2020-06-01"), "WEC"])  # filled, not NaN
    assert pd.notna(result.loc[pd.Timestamp("2020-06-01"), "PEG"])  # untouched — already had a value


def test_fill_with_icr_respects_reporting_lag():
    from grid_resilience.data.dc_load_data import fill_with_icr
    dc_history = pd.DataFrame({"WEC": [float("nan")]}, index=[pd.Timestamp("2020-06-01")])
    icr_history = pd.DataFrame({"WEC": [4.0]}, index=[pd.Timestamp("2020-05-01")])  # only 31 days back
    result = fill_with_icr(dc_history, icr_history, lag_days=45)
    assert pd.isna(result.loc[pd.Timestamp("2020-06-01"), "WEC"])


def test_fill_with_icr_handles_missing_icr_history():
    from grid_resilience.data.dc_load_data import fill_with_icr
    dc_history = pd.DataFrame({"WEC": [float("nan")]}, index=[pd.Timestamp("2020-06-01")])
    result = fill_with_icr(dc_history, None, lag_days=45)
    assert pd.isna(result.loc[pd.Timestamp("2020-06-01"), "WEC"])


def test_fill_with_icr_ticker_absent_from_icr_stays_nan():
    from grid_resilience.data.dc_load_data import fill_with_icr
    dc_history = pd.DataFrame({"EVRG": [float("nan")]}, index=[pd.Timestamp("2020-06-01")])
    icr_history = pd.DataFrame({"WEC": [4.0]}, index=[pd.Timestamp("2020-03-01")])
    result = fill_with_icr(dc_history, icr_history, lag_days=45)
    assert pd.isna(result.loc[pd.Timestamp("2020-06-01"), "EVRG"])


# ── Finding #2 (2026-08-13 review): DC/ICR subpopulations z-scored separately ──

def test_fill_with_icr_zscores_dc_and_icr_subpopulations_separately():
    """Raw DC ratios (~0.2-0.8) and raw ICR values (~1.5-4.0) must not be
    concatenated and z-scored together — that crushes whichever subgroup has
    less spread. Each subpopulation should independently have mean ~0 and
    unit-ish std after fill_with_icr, and ordering within each group must be
    preserved (higher raw value → higher z-score)."""
    from grid_resilience.data.dc_load_data import fill_with_icr

    date = pd.Timestamp("2020-06-01")
    dc_history = pd.DataFrame(
        {
            "AEP": [0.8],   # real DC data, highest in DC group
            "D":   [0.2],   # real DC data, lowest in DC group
            "PEG": [float("nan")],  # to be filled from ICR
            "WEC": [float("nan")],  # to be filled from ICR
        },
        index=[date],
    )
    icr_history = pd.DataFrame(
        {"PEG": [4.0], "WEC": [1.5]},  # PEG highest ICR, WEC lowest
        index=[pd.Timestamp("2020-03-01")],  # within lag window
    )
    result = fill_with_icr(dc_history, icr_history, lag_days=45)
    row = result.loc[date]

    # Ordering preserved within each subpopulation.
    assert row["AEP"] > row["D"]
    assert row["PEG"] > row["WEC"]

    # Each subpopulation is independently centered — neither group is pinned
    # near a constant offset because the other group has a different scale.
    dc_group  = row[["AEP", "D"]]
    icr_group = row[["PEG", "WEC"]]
    assert abs(dc_group.mean()) < 1e-6
    assert abs(icr_group.mean()) < 1e-6

    # The two subpopulations no longer carry their raw absolute scale — a
    # naive concatenate-then-zscore would leave the ICR group's z-scores
    # dominated by its larger raw range (4.0 vs 1.5) relative to the DC
    # group's (0.8 vs 0.2); after independent z-scoring both groups have the
    # same normalized spread regardless of their original units.
    assert abs(dc_group.iloc[0] - icr_group.iloc[0]) < 1e-6  # both +z (max of pair)
    assert abs(dc_group.iloc[1] - icr_group.iloc[1]) < 1e-6  # both -z (min of pair)


def test_fill_with_icr_docstring_contract_is_zscored_not_raw_passthrough():
    """A single real DC value with no other DC-covered ticker that date has
    no within-group spread — cross_section_zscore demeans it to 0.0. This
    pins down the new (z-scored) contract explicitly, replacing the old
    raw-passthrough assumption the previous test suite baked in."""
    from grid_resilience.data.dc_load_data import fill_with_icr
    dc_history = pd.DataFrame(
        {"PEG": [0.3], "WEC": [float("nan")]},
        index=[pd.Timestamp("2020-06-01")],
    )
    icr_history = pd.DataFrame(
        {"WEC": [4.0]},
        index=[pd.Timestamp("2020-03-01")],
    )
    result = fill_with_icr(dc_history, icr_history, lag_days=45)
    # Single-member subpopulations demean to exactly 0 (cross_section_zscore's
    # std==0 branch), not the raw input value.
    assert result.loc[pd.Timestamp("2020-06-01"), "PEG"] == 0.0
    assert result.loc[pd.Timestamp("2020-06-01"), "WEC"] == 0.0


# ── Finding #11 (2026-08-13 review): realistic short-code zone_size fixture ──

def test_zone_size_resolves_for_every_pjm_ticker_against_realistic_fixture():
    """Build a zonal-load fixture the way fetch_zonal_load actually produces
    one — i.e. already normalized from PJM's raw short codes via
    _normalize_pjm_load_zone — and confirm zone_size() is non-NaN for every
    PJM ticker's load_zones in TICKER_NODE_MAP. This is exactly the check
    that would have caught finding #1 (only AEP/DOM/ATSI resolving, EXC/PPL/
    PEG/FE's non-ATSI zones silently discarded) before it shipped."""
    from grid_resilience.data.dc_load_data import zone_size
    from grid_resilience.data.grid_data import _normalize_pjm_load_zone
    from grid_resilience.data.utility_node_map import TICKER_NODE_MAP

    raw_short_codes = [
        "AE", "AEP", "AP", "ATSI", "BC", "CE", "DAY", "DEOK", "DOM", "DPL",
        "DUQ", "EKPC", "JC", "ME", "PE", "PEP", "PL", "PN", "PS", "RECO", "RTO",
    ]
    as_of = pd.Timestamp("2020-01-01")
    rows = [
        {
            "time": as_of - pd.Timedelta(days=30),
            "zone": _normalize_pjm_load_zone(code),
            "load_mw": 1000.0,
        }
        for code in raw_short_codes
    ]
    zonal_load = pd.DataFrame(rows)

    for ticker, info in TICKER_NODE_MAP.items():
        if info.get("iso") != "PJM":
            continue
        zones = info.get("load_zones", [])
        if not zones:
            continue
        result = zone_size(zonal_load, zones, as_of)
        assert pd.notna(result), (
            f"{ticker}'s load_zones {zones} resolved to NaN zone_size against "
            "the realistic short-code-normalized fixture — a zone mapping gap."
        )
