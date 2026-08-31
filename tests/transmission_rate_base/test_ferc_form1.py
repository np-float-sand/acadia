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
