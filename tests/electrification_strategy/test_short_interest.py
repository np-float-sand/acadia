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


def test_crowding_diagnostic_returns_complete_latest_row_not_chimera(monkeypatch, tmp_path):
    """Regression: groupby().last() returns per-column last non-null, creating
    chimera records. Test that we get a coherent single-row snapshot, not
    backfilled columns from older snapshots.
    """
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)

    # Manually construct history with two DLR snapshots:
    # - Older: complete data
    # - Newer: missing short_pct_float (NaN)
    # Test should preserve the NaN, not backfill from older snapshot.
    older_ts = pd.Timestamp("2025-01-01", tz="UTC")
    newer_ts = pd.Timestamp("2025-01-02", tz="UTC")

    history_rows = [
        {
            "ticker": "DLR",
            "shares_short": 12_000_000,
            "short_ratio": 2.1,
            "short_pct_float": 0.031,
            "date_short_interest": pd.Timestamp("2025-01-01", tz="UTC"),
            "fetched_at": older_ts,
        },
        {
            "ticker": "DLR",
            "shares_short": 12_500_000,
            "short_ratio": 2.2,
            "short_pct_float": None,  # NaN in newer snapshot
            "date_short_interest": pd.Timestamp("2025-01-02", tz="UTC"),
            "fetched_at": newer_ts,
        },
    ]
    history = pd.DataFrame(history_rows)

    diag = si.crowding_diagnostic(history, tickers=("DLR",))

    # Should return the newer snapshot as a complete row.
    assert diag.loc["DLR", "shares_short"] == 12_500_000  # newer value
    assert diag.loc["DLR", "short_ratio"] == 2.2  # newer value
    assert pd.isna(diag.loc["DLR", "short_pct_float"])  # NaN from newer, not backfilled


# -- Final-review finding 1: implausible-snapshot guard --
def _fake_info_anomalous(ticker):
    """DLR as actually returned by yfinance in the v6 run (near-zero shares AND 0% of
    float -- the signature of a stale/missing field for a ~$50B REIT), alongside a
    normal-looking EQIX row."""
    table = {
        "DLR": {"sharesShort": 8_721, "shortRatio": 0.0, "shortPercentOfFloat": 0.0,
                "dateShortInterest": 1735689600},
        "EQIX": {"sharesShort": 1_646_357, "shortRatio": 2.81, "shortPercentOfFloat": 0.0191,
                 "dateShortInterest": 1735689600},
    }
    return table[ticker]


def test_data_quality_flag_fires_on_implausible_row_only(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_fetch_one", _fake_info_anomalous)
    df = si.fetch_short_interest(["DLR", "EQIX"])
    assert "data_quality_flag" in df.columns
    assert bool(df.loc["DLR", "data_quality_flag"]) is True
    assert bool(df.loc["EQIX", "data_quality_flag"]) is False


def test_data_quality_flag_absent_for_normal_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_fetch_one", _fake_info)
    df = si.fetch_short_interest(["DLR", "EQIX"])
    assert not df["data_quality_flag"].any()


def test_data_quality_flag_missing_pct_with_large_shares_is_not_flagged(monkeypatch, tmp_path):
    """A missing % of float alone isn't the signature -- only the near-zero-shares
    combination is. Don't over-flag."""
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_fetch_one", lambda t: {
        "sharesShort": 5_000_000, "shortRatio": 3.0, "shortPercentOfFloat": None,
        "dateShortInterest": 1735689600})
    df = si.fetch_short_interest(["EQIX"])
    assert bool(df.loc["EQIX", "data_quality_flag"]) is False


def test_crowding_diagnostic_threads_data_quality_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_fetch_one", _fake_info_anomalous)
    si.fetch_short_interest(["DLR", "EQIX"])
    history = pd.read_parquet(tmp_path / config.SHORT_INTEREST_CACHE_FILE)
    diag = si.crowding_diagnostic(history, tickers=("DLR", "EQIX"))
    assert "data_quality_flag" in diag.columns
    assert bool(diag.loc["DLR", "data_quality_flag"]) is True
    assert bool(diag.loc["EQIX", "data_quality_flag"]) is False


def test_flag_backfilled_when_appending_to_a_legacy_cache(monkeypatch, tmp_path):
    """A cache written before the guard existed has no data_quality_flag column; appending a
    new snapshot must not leave the older rows carrying a null flag (which would surface as
    None in the diagnostic if that row is the latest for a ticker)."""
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    legacy = pd.DataFrame([
        {"ticker": "DLR", "shares_short": 8_721, "short_ratio": 0.0, "short_pct_float": 0.0,
         "date_short_interest": pd.Timestamp("2024-12-01", tz="UTC"),
         "fetched_at": pd.Timestamp("2024-12-01", tz="UTC")},
    ])
    legacy.to_parquet(tmp_path / config.SHORT_INTEREST_CACHE_FILE, index=False)

    monkeypatch.setattr(si, "_fetch_one", _fake_info_anomalous)
    si.fetch_short_interest(["DLR", "EQIX"])

    history = pd.read_parquet(tmp_path / config.SHORT_INTEREST_CACHE_FILE)
    assert history["data_quality_flag"].notna().all()
    assert history["data_quality_flag"].dtype == bool
    legacy_row = history[history["date_short_interest"] == pd.Timestamp("2024-12-01", tz="UTC")]
    assert bool(legacy_row["data_quality_flag"].iloc[0]) is True

    diag = si.crowding_diagnostic(history, tickers=("DLR", "EQIX"))
    assert diag["data_quality_flag"].notna().all()


def test_crowding_diagnostic_recomputes_flag_for_legacy_history(monkeypatch, tmp_path):
    """A cache written before the guard existed has no data_quality_flag column;
    crowding_diagnostic must still report one rather than KeyError."""
    history = pd.DataFrame([
        {"ticker": "DLR", "shares_short": 8_721, "short_ratio": 0.0, "short_pct_float": 0.0,
         "date_short_interest": pd.Timestamp("2025-01-01", tz="UTC"),
         "fetched_at": pd.Timestamp("2025-01-01", tz="UTC")},
        {"ticker": "EQIX", "shares_short": 1_646_357, "short_ratio": 2.81, "short_pct_float": 0.0191,
         "date_short_interest": pd.Timestamp("2025-01-01", tz="UTC"),
         "fetched_at": pd.Timestamp("2025-01-01", tz="UTC")},
    ])
    diag = si.crowding_diagnostic(history, tickers=("DLR", "EQIX"))
    assert bool(diag.loc["DLR", "data_quality_flag"]) is True
    assert bool(diag.loc["EQIX", "data_quality_flag"]) is False


# -- Final-review finding 14: empty ticker list --
def test_fetch_short_interest_empty_tickers_returns_empty_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(si, "_fetch_one", _fake_info)
    df = si.fetch_short_interest([])
    assert df.empty
    assert df.index.name == "ticker"
    for col in ("shares_short", "short_ratio", "short_pct_float",
                "date_short_interest", "fetched_at", "data_quality_flag"):
        assert col in df.columns
    # nothing fetched => nothing written
    assert not (tmp_path / config.SHORT_INTEREST_CACHE_FILE).exists()


def test_cache_preserves_snapshots_with_different_date_short_interest(monkeypatch, tmp_path):
    """Regression: Cache should preserve multiple snapshots for the same ticker
    if they have different date_short_interest values (different reporting dates).
    Deduplication should only remove rows with identical date_short_interest.
    """
    monkeypatch.setattr(si.config, "CACHE_DIR", tmp_path)

    def fake_info_variant(ticker):
        # First call: date_short_interest = 2025-01-01
        # Second call: date_short_interest = 2025-01-08 (one week later)
        state = {"call_count": 0}
        if not hasattr(fake_info_variant, "_state"):
            fake_info_variant._state = state
        fake_info_variant._state["call_count"] += 1

        if fake_info_variant._state["call_count"] == 1:
            return {
                "sharesShort": 12_000_000,
                "shortRatio": 2.1,
                "shortPercentOfFloat": 0.031,
                "dateShortInterest": 1735689600,  # 2025-01-01
            }
        else:
            return {
                "sharesShort": 12_100_000,
                "shortRatio": 2.15,
                "shortPercentOfFloat": 0.032,
                "dateShortInterest": 1736294400,  # 2025-01-08
            }

    monkeypatch.setattr(si, "_fetch_one", fake_info_variant)

    # First fetch
    si.fetch_short_interest(["DLR"])
    # Second fetch with updated date_short_interest
    si.fetch_short_interest(["DLR"])

    history = pd.read_parquet(tmp_path / config.SHORT_INTEREST_CACHE_FILE)
    dlr_rows = history[history["ticker"] == "DLR"]

    # Should have 2 distinct rows (different date_short_interest)
    assert len(dlr_rows) == 2
    dates = dlr_rows["date_short_interest"].unique()
    assert len(dates) == 2
    # Verify both snapshots are distinct and ordered by date
    assert dlr_rows.iloc[0]["shares_short"] == 12_000_000
    assert dlr_rows.iloc[1]["shares_short"] == 12_100_000
