import pandas as pd
import pytest

from transmission_rate_base.data import ferc_form1 as f1


def test_offline_uses_cache_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(f1, "CACHE_DIR", tmp_path)
    cached = pd.DataFrame({"utility_id_ferc1": [1], "report_year": [2020]})
    cached.to_parquet(tmp_path / "core_ferc1__yearly_plant_in_service_sched204.parquet")

    def _boom(url):  # network must not be touched
        raise AssertionError(f"network hit for {url}")
    monkeypatch.setattr(f1, "_read_remote", _boom)

    got = f1.fetch_table("plant_in_service", offline=True)
    pd.testing.assert_frame_equal(got, cached)


def test_offline_without_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(f1, "CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        f1.fetch_table("dep_by_function", offline=True)


def test_download_writes_cache_then_reuses_it(tmp_path, monkeypatch):
    monkeypatch.setattr(f1, "CACHE_DIR", tmp_path)
    payload = pd.DataFrame({"utility_id_ferc1": [7, 7], "report_year": [2019, 2020]})
    calls = []

    def _fake_remote(url):
        calls.append(url)
        return payload
    monkeypatch.setattr(f1, "_read_remote", _fake_remote)

    first = f1.fetch_table("plant_summary")
    second = f1.fetch_table("plant_summary")
    pd.testing.assert_frame_equal(first, payload)
    pd.testing.assert_frame_equal(second, payload)
    assert len(calls) == 1
    assert calls[0].endswith("core_ferc1__yearly_utility_plant_summary_sched200.parquet")


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        f1.fetch_table("not_a_table")


# ── Task 3: extraction (real PUDL schedule vocabularies) ──────────────────────

def _sched204_rows():
    return pd.DataFrame([
        # filer 1, 2020: explicit subtotal (calculated_value)=100 AND leaves 60+30 -> subtotal wins
        dict(utility_id_ferc1=1, report_year=2020, ferc_account_label="transmission_plant",
             row_type_xbrl="calculated_value", ending_balance=100.0),
        dict(utility_id_ferc1=1, report_year=2020, ferc_account_label="towers_and_fixtures_transmission_plant",
             row_type_xbrl="reported_value", ending_balance=60.0),
        dict(utility_id_ferc1=1, report_year=2020, ferc_account_label="station_equipment_transmission_plant",
             row_type_xbrl="reported_value", ending_balance=30.0),
        # filer 2, 2020: leaves only 40+25; an RTO row (999) and a correction row (7) must be ignored
        dict(utility_id_ferc1=2, report_year=2020,
             ferc_account_label="overhead_conductors_and_devices_transmission_plant",
             row_type_xbrl="reported_value", ending_balance=40.0),
        dict(utility_id_ferc1=2, report_year=2020, ferc_account_label="poles_and_fixtures_transmission_plant",
             row_type_xbrl="reported_value", ending_balance=25.0),
        dict(utility_id_ferc1=2, report_year=2020,
             ferc_account_label="communication_equipment_regional_transmission_and_market_operation_plant",
             row_type_xbrl="reported_value", ending_balance=999.0),
        dict(utility_id_ferc1=2, report_year=2020, ferc_account_label="transmission_plant_correction",
             row_type_xbrl="correction", ending_balance=7.0),
    ])


def test_gross_transmission_prefers_subtotal_sums_leaves_drops_rto_and_correction():
    out = f1.gross_transmission_plant(_sched204_rows()).set_index("utility_id_ferc1")["gross_tx"]
    assert out.loc[1] == 100.0     # calculated_value subtotal beats 60+30
    assert out.loc[2] == 65.0      # 40+25 ; RTO 999 and correction 7 excluded


def test_transmission_accum_depreciation_filters_function_type_and_abs():
    s219 = pd.DataFrame([
        dict(utility_id_ferc1=1, report_year=2020, plant_function="transmission",
             depreciation_type="accumulated_depreciation", row_type_xbrl="reported_value",
             ending_balance=-30.0),
        dict(utility_id_ferc1=1, report_year=2020, plant_function="distribution",
             depreciation_type="accumulated_depreciation", row_type_xbrl="reported_value",
             ending_balance=-999.0),
        dict(utility_id_ferc1=1, report_year=2020, plant_function="transmission",
             depreciation_type="accumulated_depreciation_subdimension_correction",
             row_type_xbrl="subdimension_correction", ending_balance=-5.0),
    ])
    out = f1.transmission_accum_depreciation(s219).set_index("utility_id_ferc1")["accum_dep_tx"]
    assert out.loc[1] == 30.0      # only the transmission/reported_value row, abs()


def test_total_net_utility_plant_reads_asset_type_and_electric_only():
    s200 = pd.DataFrame([
        dict(utility_id_ferc1=9, report_year=2021, utility_type="electric",
             utility_plant_asset_type="utility_plant_net", row_type_xbrl="calculated_value",
             ending_balance=600.0),
        dict(utility_id_ferc1=9, report_year=2021, utility_type="electric",
             utility_plant_asset_type="utility_plant_in_service_classified",
             row_type_xbrl="reported_value", ending_balance=1000.0),
        dict(utility_id_ferc1=9, report_year=2021, utility_type="total",
             utility_plant_asset_type="utility_plant_net", row_type_xbrl="calculated_value",
             ending_balance=1234.0),
    ])
    out = f1.total_net_utility_plant(s200).set_index("utility_id_ferc1")
    assert out.loc[9, "net_total"] == 600.0
    assert out.loc[9, "gross_total"] == 1000.0


def _sched200_for(uid, gross, net):
    return [
        dict(utility_id_ferc1=uid, report_year=2021, utility_type="electric",
             utility_plant_asset_type="utility_plant_in_service_classified",
             row_type_xbrl="reported_value", ending_balance=float(gross)),
        dict(utility_id_ferc1=uid, report_year=2021, utility_type="electric",
             utility_plant_asset_type="utility_plant_net",
             row_type_xbrl="calculated_value", ending_balance=float(net)),
    ]


def test_net_transmission_subtracts_depreciation_when_present():
    s204 = pd.DataFrame([dict(utility_id_ferc1=5, report_year=2021,
        ferc_account_label="transmission_plant", row_type_xbrl="calculated_value", ending_balance=200.0)])
    s219 = pd.DataFrame([dict(utility_id_ferc1=5, report_year=2021, plant_function="transmission",
        depreciation_type="accumulated_depreciation", row_type_xbrl="reported_value", ending_balance=-50.0)])
    s200 = pd.DataFrame(_sched200_for(5, 1500, 999))
    out = f1.net_transmission_plant(s204, s219, s200).set_index("utility_id_ferc1")
    assert out.loc[5, "net_tx"] == 150.0
    assert bool(out.loc[5, "net_tx_prorated"]) is False


def test_net_transmission_prorates_when_depreciation_missing():
    s204 = pd.DataFrame([dict(utility_id_ferc1=9, report_year=2021,
        ferc_account_label="transmission_plant", row_type_xbrl="calculated_value", ending_balance=200.0)])
    s219 = pd.DataFrame(columns=["utility_id_ferc1", "report_year", "plant_function",
                                 "depreciation_type", "row_type_xbrl", "ending_balance"])
    s200 = pd.DataFrame(_sched200_for(9, 1000, 600))
    out = f1.net_transmission_plant(s204, s219, s200).set_index("utility_id_ferc1")
    assert out.loc[9, "net_tx"] == 120.0      # 200 * (600/1000)
    assert bool(out.loc[9, "net_tx_prorated"]) is True
