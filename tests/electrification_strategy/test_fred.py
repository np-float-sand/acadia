import io

import pandas as pd
import pytest

from electrification_strategy import fred


_CSV = "observation_date,DFII10\n2020-01-02,0.15\n2020-01-03,.\n2020-01-06,0.10\n"


def test_parse_dot_as_nan_and_slice(monkeypatch, tmp_path):
    monkeypatch.setattr(fred.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fred, "_http_csv", lambda url: pd.read_csv(io.StringIO(_CSV)))
    s = fred.fetch_series("DFII10", "2020-01-01", "2020-01-31")
    assert list(s.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    assert s.name == "DFII10"
    assert s.loc["2020-01-02"] == 0.15


def test_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(fred.config, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_http(url):
        calls["n"] += 1
        return pd.read_csv(io.StringIO(_CSV))

    monkeypatch.setattr(fred, "_http_csv", fake_http)
    fred.fetch_series("DFII10", "2020-01-01", "2020-01-31")
    fred.fetch_series("DFII10", "2020-01-01", "2020-01-31")
    assert calls["n"] == 1  # second call served from parquet cache


def test_offline_fallback_to_committed_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(fred.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fred.config, "DATA_DIR", tmp_path)
    (tmp_path / "fred_DFII10.csv").write_text(_CSV)

    def boom(url):
        raise RuntimeError("network down")

    monkeypatch.setattr(fred, "_http_csv", boom)
    with pytest.warns(RuntimeWarning):
        s = fred.fetch_series("DFII10", "2020-01-01", "2020-01-31")
    assert s.loc["2020-01-06"] == 0.10
