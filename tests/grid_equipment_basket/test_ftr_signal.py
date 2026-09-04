import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import ftr_signal as fs


# ── parse_auction_month ──────────────────────────────────────────────────────

def test_parse_auction_month_reads_month_and_year():
    assert fs.parse_auction_month("JAN 2019 Auction") == pd.Timestamp("2019-01-01")
    assert fs.parse_auction_month("SEP 2025 Auction") == pd.Timestamp("2025-09-01")


def test_parse_auction_month_rejects_unrecognised_string():
    with pytest.raises(ValueError):
        fs.parse_auction_month("not a market name")


# ── mw_weighted_price ────────────────────────────────────────────────────────

def test_mw_weighted_price_weights_by_quoted_mw():
    bids = pd.DataFrame({
        "quoted_price": [100.0, 0.0],
        "quoted_mw": [1.0, 9.0],
    })
    # (100*1 + 0*9) / 10 = 10.0 -- the tiny 1MW bid barely moves the average
    assert fs.mw_weighted_price(bids) == pytest.approx(10.0)


def test_mw_weighted_price_empty_frame_is_nan():
    bids = pd.DataFrame(columns=["quoted_price", "quoted_mw"])
    assert fs.mw_weighted_price(bids) != fs.mw_weighted_price(bids)  # NaN != NaN


def test_mw_weighted_price_zero_total_mw_is_nan():
    bids = pd.DataFrame({"quoted_price": [5.0, -5.0], "quoted_mw": [0.0, 0.0]})
    result = fs.mw_weighted_price(bids)
    assert result != result


# ── zone_monthly_composite ───────────────────────────────────────────────────

def _bids(rows):
    return pd.DataFrame(rows, columns=["sink_pnode_name", "quoted_price", "quoted_mw"])


def test_zone_monthly_composite_averages_zones_equal_weight():
    bids = _bids([
        ("AEP", 100.0, 10.0),   # AEP MW-weighted price = 100
        ("COMED", 200.0, 10.0),  # COMED MW-weighted price = 200
    ])
    zones = {"AEP": ["AEP"], "COMED": ["COMED"]}
    # equal-weight mean of the two zone prices, not MW-weighted across zones
    assert fs.zone_monthly_composite(bids, zones) == pytest.approx(150.0)


def test_zone_monthly_composite_skips_zones_with_no_matching_bids():
    bids = _bids([("AEP", 100.0, 10.0)])
    zones = {"AEP": ["AEP"], "COMED": ["COMED"]}
    # COMED absent this month -> composite is AEP alone, not NaN
    assert fs.zone_monthly_composite(bids, zones) == pytest.approx(100.0)


def test_zone_monthly_composite_matches_any_sink_alias_for_a_zone():
    bids = _bids([("DOMINION HUB", 50.0, 4.0)])
    zones = {"DOM": ["DOM", "DOMINION HUB"]}
    assert fs.zone_monthly_composite(bids, zones) == pytest.approx(50.0)


def test_zone_monthly_composite_all_zones_absent_is_nan():
    bids = _bids([("SOME_OTHER_NODE", 1.0, 1.0)])
    zones = {"AEP": ["AEP"]}
    result = fs.zone_monthly_composite(bids, zones)
    assert result != result


# ── monthly_composite_series ─────────────────────────────────────────────────

def test_monthly_composite_series_indexes_by_auction_month_sorted():
    by_month = {
        "FEB 2019 Auction": _bids([("AEP", 20.0, 1.0)]),
        "JAN 2019 Auction": _bids([("AEP", 10.0, 1.0)]),
    }
    s = fs.monthly_composite_series(by_month, {"AEP": ["AEP"]})
    assert list(s.index) == [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-01")]
    assert s.iloc[0] == pytest.approx(10.0)
    assert s.iloc[1] == pytest.approx(20.0)


# ── available_series ─────────────────────────────────────────────────────────

def test_available_series_shifts_index_forward_by_lag_months():
    monthly = pd.Series([10.0, 20.0],
                        index=[pd.Timestamp("2019-01-01"), pd.Timestamp("2019-02-01")])
    avail = fs.available_series(monthly, lag_months=4)
    assert list(avail.index) == [pd.Timestamp("2019-05-01"), pd.Timestamp("2019-06-01")]
    assert list(avail.values) == [10.0, 20.0]


# ── broadcast_daily ───────────────────────────────────────────────────────────

def test_broadcast_daily_forward_fills_from_availability_date():
    avail = pd.Series([10.0, 20.0],
                      index=[pd.Timestamp("2019-05-01"), pd.Timestamp("2019-06-01")])
    daily = fs.broadcast_daily(avail, "2019-04-25", "2019-06-05")
    assert daily.loc["2019-04-30"] != daily.loc["2019-04-30"]  # NaN before first availability
    assert daily.loc["2019-05-01"] == 10.0
    assert daily.loc["2019-05-31"] == 10.0                      # held through the month
    assert daily.loc["2019-06-01"] == 20.0
    assert daily.loc["2019-06-05"] == 20.0


# ── market_name_for / _market_names_for_range ────────────────────────────────

def test_market_name_for_round_trips_with_parse_auction_month():
    month = pd.Timestamp("2019-01-01")
    assert fs.market_name_for(month) == "JAN 2019 Auction"
    assert fs.parse_auction_month(fs.market_name_for(month)) == month


def test_market_names_for_range_covers_every_month_inclusive():
    names = fs._market_names_for_range("2019-01-15", "2019-03-01")
    assert names == ["JAN 2019 Auction", "FEB 2019 Auction", "MAR 2019 Auction"]


# ── ftr_composite (orchestration, injected bids_fn) ──────────────────────────

def test_ftr_composite_applies_lag_and_zscores(monkeypatch):
    # Six months of a rising AEP price so the z-score has real dispersion to
    # measure once warmed up; single zone keeps the composite == AEP's price.
    months = pd.date_range("2019-01-01", periods=6, freq="MS")
    prices = [100.0, 100.0, 100.0, 100.0, 100.0, 400.0]

    def fake_bids_fn(market_names):
        out = {}
        for mn in market_names:
            m = fs.parse_auction_month(mn)
            if m in months:
                price = prices[list(months).index(m)]
                out[mn] = pd.DataFrame([{"sink_pnode_name": "AEP",
                                         "quoted_price": price, "quoted_mw": 10.0}])
            else:
                out[mn] = pd.DataFrame(columns=["sink_pnode_name", "quoted_price", "quoted_mw"])
        return out

    result = fs.ftr_composite(
        "2019-01-01", "2019-12-31", lag_months=4,
        zone_sinks={"AEP": ["AEP"]},
        zscore_window=300, zscore_minp=5, winsor=3.0,
        bids_fn=fake_bids_fn,
    )
    # JAN's value (100) isn't knowable until lag_months later (May) -- before
    # that the series must be NaN, not leak the future value early.
    assert pd.isna(result.loc["2019-04-30"])
    # The June auction's spike (400) becomes available in October and should
    # register as a positive z vs the flat 100 history preceding it (which
    # has zero variance, so it z-scores to NaN via _trailing_zscore's
    # divide-by-zero guard -- only the spike breaks that).
    assert result.loc["2019-10-05"] > 1.0


def test_ftr_composite_empty_bids_returns_empty_series():
    result = fs.ftr_composite("2019-01-01", "2019-03-01",
                              bids_fn=lambda names: {n: pd.DataFrame() for n in names})
    assert result.empty


# ── ftr_signal_report (gate evaluation, injected price_fn/bids_fn) ──────────

def _flat_price_fn(tickers, start, end):
    """Deterministic flat-then-jump prices so the basket has a real return
    series to overlay, with no network I/O."""
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(0)
    data = {}
    for t in tickers:
        rets = rng.normal(0.0003, 0.01, size=len(idx))
        data[t] = 100.0 * (1.0 + pd.Series(rets, index=idx)).cumprod()
    return pd.DataFrame(data)


def test_ftr_signal_report_runs_baselines_and_gate_end_to_end(monkeypatch):
    from grid_equipment_basket import config as cfg

    def fake_bids_fn(market_names):
        rng = np.random.default_rng(1)
        out = {}
        for mn in market_names:
            price = 100.0 + rng.normal(0, 20)
            out[mn] = pd.DataFrame([{"sink_pnode_name": "AEP", "quoted_price": price,
                                     "quoted_mw": 10.0}])
        return out

    rep = fs.ftr_signal_report(
        price_fn=_flat_price_fn, bids_fn=fake_bids_fn,
        signal_start="2018-01-01",
        primary=("2023-01-01", "2023-06-30"),
        prior=("2020-01-01", "2020-06-30"),
        zone_sinks={"AEP": ["AEP"]},
    )
    assert "baselines" in rep and "buy_and_hold" in rep["baselines"]
    assert "layer1_only" in rep["baselines"]
    assert "ftr_rung" in rep
    assert "gate" in rep["ftr_rung"] and "verdict" in rep
