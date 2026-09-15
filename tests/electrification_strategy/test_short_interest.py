import pandas as pd
import pytest

from electrification_strategy import config, short_interest as si


def _fake_info(ticker):
    table = {
        "DLR": {"sharesShort": 12_000_000, "shortRatio": 2.1, "shortPercentOfFloat": 0.031,
                "dateShortInterest": 1735689600},  # 2025-01-01 UTC epoch seconds
        "EQIX": {"sharesShort": 900_000, "shortRatio": 1.4, "shortPercentOfFloat": 0.012,
                 "dateShortInterest": 1735689600},
    }
    return table[ticker]


def test_fetch_short_interest_shapes_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_fetch_one", _fake_info)
    df = si.fetch_short_interest(["DLR", "EQIX"])
    assert list(df.index) == ["DLR", "EQIX"]
    assert set(["shares_short", "short_ratio", "short_pct_float",
                "date_short_interest", "fetched_at"]).issubset(df.columns)
    assert df.loc["DLR", "short_pct_float"] == 0.031
    assert (tmp_path / config.SHORT_INTEREST_CACHE_FILE).exists()


def test_cache_roundtrip_appends_without_duplicating(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def counting_fetch(ticker):
        calls["n"] += 1
        return _fake_info(ticker)

    monkeypatch.setattr(si, "_fetch_one", counting_fetch)
    si.fetch_short_interest(["DLR", "EQIX"])
    si.fetch_short_interest(["DLR", "EQIX"])
    history = pd.read_parquet(tmp_path / config.SHORT_INTEREST_CACHE_FILE)
    assert calls["n"] == 4  # both calls hit the network; snapshot re-fetches by design
    # but the second snapshot on the same date_short_interest doesn't duplicate rows
    assert len(history[history["ticker"] == "DLR"]) == 1


def test_crowding_diagnostic_sorts_by_short_pct_float(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_fetch_one", _fake_info)
    si.fetch_short_interest(["DLR", "EQIX"])
    history = pd.read_parquet(tmp_path / config.SHORT_INTEREST_CACHE_FILE)
    diag = si.crowding_diagnostic(history, tickers=("DLR", "EQIX"))
    assert list(diag.index) == ["DLR", "EQIX"]  # DLR has the higher short_pct_float
