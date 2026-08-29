import pandas as pd
import pytest

from grid_equipment_basket.data import prices as px


def _fake_frame(tickers, start, end):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame(
        {t: [float(i + 1) for i in range(len(idx))] for t in tickers},
        index=idx,
    )


def test_fetch_prices_returns_sorted_requested_tickers(monkeypatch, tmp_path):
    monkeypatch.setattr(px, "_PRICE_CACHE_BASE", tmp_path / "prices.parquet")
    monkeypatch.setattr(
        px, "_download_with_retry",
        lambda t, s, e, **k: _fake_frame(t, s, e),
    )
    out = px.fetch_prices(["VRT", "PWR"], "2023-01-02", "2023-03-31")
    assert list(out.columns) == ["PWR", "VRT"]
    assert out.index.is_monotonic_increasing
    assert out.notna().all().all()


def test_fetch_prices_uses_cache_on_second_call(monkeypatch, tmp_path):
    monkeypatch.setattr(px, "_PRICE_CACHE_BASE", tmp_path / "prices.parquet")
    calls = []

    def _spy(t, s, e, **k):
        calls.append((tuple(sorted(t)), s, e))
        return _fake_frame(t, s, e)

    monkeypatch.setattr(px, "_download_with_retry", _spy)
    px.fetch_prices(["VRT"], "2023-01-02", "2023-02-28")
    n_first = len(calls)
    assert n_first > 0
    px.fetch_prices(["VRT"], "2023-01-02", "2023-02-28")
    assert len(calls) == n_first          # nothing re-downloaded


def test_fetch_prices_gap_fills_only_new_months(monkeypatch, tmp_path):
    monkeypatch.setattr(px, "_PRICE_CACHE_BASE", tmp_path / "prices.parquet")
    months = []

    def _spy(t, s, e, **k):
        months.append(s[:7])
        return _fake_frame(t, s, e)

    monkeypatch.setattr(px, "_download_with_retry", _spy)
    px.fetch_prices(["VRT"], "2023-01-02", "2023-02-28")
    months.clear()
    px.fetch_prices(["VRT"], "2023-01-02", "2023-04-28")
    assert set(months) == {"2023-03", "2023-04"}


def test_download_with_retry_retries_on_missing_column(monkeypatch):
    monkeypatch.setattr(px, "_RETRY_DELAY_SECONDS", 0)
    frames = [
        pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2023-01-02"])),
        pd.DataFrame({"A": [1.0], "B": [2.0]}, index=pd.to_datetime(["2023-01-02"])),
    ]
    seen = []

    def _fake_download(t, s, e):
        seen.append(1)
        return frames[len(seen) - 1]

    monkeypatch.setattr(px, "_download", _fake_download)
    out = px._download_with_retry(["A", "B"], "2023-01-02", "2023-01-02", max_attempts=2)
    assert list(out.columns) == ["A", "B"]
    assert len(seen) == 2
