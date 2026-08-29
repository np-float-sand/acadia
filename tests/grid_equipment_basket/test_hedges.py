import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import hedges as hg


def _flat_pair_inputs():
    # 260 business days. Makers rise 0.05%/day, contractors flat.
    idx = pd.bdate_range("2023-01-02", periods=260)
    makers = {t: 100 * np.exp(np.cumsum(np.full(len(idx), 0.0005))) for t in ["ETN", "HUBB", "GEV", "VRT", "NVT"]}
    contr = {t: np.full(len(idx), 100.0) for t in ["PWR", "MYRG", "PRIM", "FLNC"]}
    prices = pd.DataFrame({**makers, **contr}, index=idx)
    return prices, idx


def _equal_weight_pair_weights(available, asof, fund_df, backlog_df):
    mk = sorted(t for t in available if t in ["ETN", "HUBB", "GEV", "VRT", "NVT"])
    ct = sorted(t for t in available if t in ["PWR", "MYRG", "PRIM", "FLNC"])
    lw = pd.Series(1.0 / len(mk), index=mk) if mk else pd.Series(dtype=float)
    sw = pd.Series(1.0 / len(ct), index=ct) if ct else pd.Series(dtype=float)
    return lw, sw


def test_simulate_pair_is_long_minus_short(monkeypatch):
    prices, idx = _flat_pair_inputs()

    # Deterministic legs: equal-weight each bucket.
    def fake_pair_weights(available, asof, fund_df, backlog_df):
        mk = sorted(t for t in available if t in ["ETN", "HUBB", "GEV", "VRT", "NVT"])
        ct = sorted(t for t in available if t in ["PWR", "MYRG", "PRIM", "FLNC"])
        lw = pd.Series(1.0 / len(mk), index=mk) if mk else pd.Series(dtype=float)
        sw = pd.Series(1.0 / len(ct), index=ct) if ct else pd.Series(dtype=float)
        return lw, sw

    monkeypatch.setattr(hg.value_chain, "pair_weights", fake_pair_weights)
    pair = hg.simulate_pair(prices, "2023-01-02", str(idx[-1].date()), fund_df=None, backlog_df=None)
    # long ~ +0.05%/day, short ~ 0 -> pair ~ +0.05%/day
    assert pair.mean() == pytest.approx(0.0005, abs=5e-5)
    assert pair.std() == pytest.approx(0.0, abs=1e-6)


def test_pair_overlay_adds_weighted_pair():
    idx = pd.bdate_range("2023-01-02", periods=5)
    base = pd.Series([0.01, -0.02, 0.00, 0.03, -0.01], index=idx)
    pair = pd.Series([0.02, 0.02, 0.02, 0.02, 0.02], index=idx)
    out = hg.pair_overlay(base, pair, 0.30)
    assert out.iloc[0] == pytest.approx(0.01 + 0.30 * 0.02)
    assert len(out) == 5


def test_simulate_pair_ignores_off_bucket_columns(monkeypatch):
    # Controller ruling: a price frame that also carries an off-bucket column
    # (benchmark ticker / QQQ / here "XLI") must be handled by filtering to
    # bucket members first, not by crashing in value_chain.bucket_of.
    prices, idx = _flat_pair_inputs()
    monkeypatch.setattr(hg.value_chain, "pair_weights", _equal_weight_pair_weights)

    start, end = "2023-01-02", str(idx[-1].date())
    baseline = hg.simulate_pair(prices, start, end, fund_df=None, backlog_df=None)

    with_extra = prices.copy()
    with_extra["XLI"] = 100.0
    got = hg.simulate_pair(with_extra, start, end, fund_df=None, backlog_df=None)

    pd.testing.assert_series_equal(got, baseline)
