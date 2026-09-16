import pandas as pd
import pytest

from electrification_stress_beta import national_gsi


def test_build_national_gsi_averages_per_iso(monkeypatch):
    idx = pd.bdate_range("2021-01-04", periods=5)
    gsi_a = pd.DataFrame({"gsi": [0.2, 0.4, 0.6, 0.8, 1.0]}, index=idx)
    gsi_b = pd.DataFrame({"gsi": [0.0, 0.2, 0.4, 0.6, 0.8]}, index=idx)
    monkeypatch.setattr(national_gsi, "build_full_event_calendar", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(national_gsi, "build_multi_iso_gsi",
                         lambda *a, **k: {"ISO_A": gsi_a, "ISO_B": gsi_b})
    out = national_gsi.build_national_gsi(
        {"ISO_A": pd.DataFrame({"lmp_max": [1]}), "ISO_B": pd.DataFrame({"lmp_max": [1]})},
        {"ISO_A": pd.DataFrame(), "ISO_B": pd.DataFrame()},
        isos=["ISO_A", "ISO_B"],
    )
    expected = (gsi_a["gsi"] + gsi_b["gsi"]) / 2.0
    pd.testing.assert_series_equal(out["gsi"], expected, check_names=False)


def test_build_national_gsi_empty_when_no_isos_available(monkeypatch):
    monkeypatch.setattr(national_gsi, "build_full_event_calendar", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(national_gsi, "build_multi_iso_gsi", lambda *a, **k: {})
    out = national_gsi.build_national_gsi({}, {}, isos=["ERCOT"])
    assert out.empty


def test_build_national_gsi_ignores_isos_outside_the_requested_set(monkeypatch):
    idx = pd.bdate_range("2021-01-04", periods=3)
    gsi_a = pd.DataFrame({"gsi": [0.1, 0.2, 0.3]}, index=idx)
    gsi_extra = pd.DataFrame({"gsi": [0.9, 0.9, 0.9]}, index=idx)
    monkeypatch.setattr(national_gsi, "build_full_event_calendar", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(national_gsi, "build_multi_iso_gsi",
                         lambda *a, **k: {"ISO_A": gsi_a, "ISO_EXTRA": gsi_extra})
    out = national_gsi.build_national_gsi(
        {"ISO_A": pd.DataFrame({"lmp_max": [1]})}, {"ISO_A": pd.DataFrame()}, isos=["ISO_A"],
    )
    pd.testing.assert_series_equal(out["gsi"], gsi_a["gsi"], check_names=False)


def test_uniform_ticker_iso_map():
    m = national_gsi.uniform_ticker_iso_map(["ETN", "HUBB"])
    assert m == {"ETN": national_gsi.NATIONAL_KEY, "HUBB": national_gsi.NATIONAL_KEY}
