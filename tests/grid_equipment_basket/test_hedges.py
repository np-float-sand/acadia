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


def test_conditional_short_mask_true_only_when_trend_down_and_vol_high():
    idx = pd.bdate_range("2022-01-03", periods=600)
    # First ~15 months: steady uptrend, low vol. Then a sharp draw + noise spike.
    up = np.linspace(100, 200, 320)
    down = 200 + np.cumsum(np.random.default_rng(0).normal(-0.6, 4.0, len(idx) - 320))
    close = pd.Series(np.concatenate([up, down]), index=idx)

    mask = hg.conditional_short_mask(close, ma_days=100, vol_days=20, vol_ref_days=252)
    assert mask.dtype == bool
    assert not mask.loc["2022-06-01":"2022-09-30"].any()       # calm uptrend -> off
    assert mask.loc["2023-06-01":"2023-12-31"].any()           # drawdown + vol spike -> on somewhere


def test_conditional_short_mask_holds_month_end_verdict_through_next_month():
    idx = pd.bdate_range("2022-01-03", periods=600)
    close = pd.Series(np.linspace(200, 100, len(idx)), index=idx)   # persistent downtrend
    mask = hg.conditional_short_mask(close)
    # within any single month the mask value is constant
    window = mask.loc["2023-01-01":"2023-06-30"]
    for _, chunk in window.groupby(window.index.to_period("M")):
        assert chunk.nunique() == 1


def test_conditional_short_overlay_subtracts_only_on_masked_days():
    idx = pd.bdate_range("2023-01-02", periods=4)
    base = pd.Series([0.01, 0.01, 0.01, 0.01], index=idx)
    qqq = pd.Series([0.02, 0.02, 0.02, 0.02], index=idx)
    mask = pd.Series([False, True, True, False], index=idx)
    out = hg.conditional_short_overlay(base, qqq, mask, 0.30)
    assert out.tolist() == pytest.approx([0.01, 0.01 - 0.006, 0.01 - 0.006, 0.01])


def test_find_drawdown_episode_picks_peak_in_window_then_trough():
    idx = pd.bdate_range("2024-01-01", "2025-08-01")
    curve = pd.Series(1.0, index=idx)
    curve.loc["2024-11-15"] = np.nan   # marker only; build returns instead
    # Build a return series: flat, then +/- to make a clear Nov-2024 peak and Apr-2025 trough.
    r = pd.Series(0.0, index=idx)
    r.loc["2024-07-01":"2024-11-15"] = 0.001     # rise into a mid-Nov peak
    r.loc["2024-11-18":"2025-04-10"] = -0.002    # fall into an April trough
    r.loc["2025-04-11":] = 0.001                 # recover
    peak, trough = hg.find_drawdown_episode(r, ("2024-07-01", "2024-12-31"), "2025-06-30")
    assert pd.Timestamp("2024-11-01") <= peak <= pd.Timestamp("2024-11-30")
    assert pd.Timestamp("2025-03-15") <= trough <= pd.Timestamp("2025-04-30")


def test_episode_drawdown_is_min_over_the_fixed_span():
    idx = pd.bdate_range("2024-10-01", "2025-05-01")
    r = pd.Series(0.0, index=idx)
    r.loc["2024-11-01":"2025-03-01"] = -0.01
    dd = hg.episode_drawdown(r, pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-01"))
    assert dd < -0.5 and dd > -0.95


def test_annualized_carry_matches_closed_form():
    r = pd.Series([0.001] * 252)
    # compute_metrics rounds cagr to 4 dp, so match the closed form within that rounding
    # (brief said rel=1e-6, which is tighter than compute_metrics' own precision).
    assert hg.annualized_carry(r) == pytest.approx((1.001 ** 252) - 1, rel=1e-3)


def test_risk_match_weight_scales_pair_to_target_vol():
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(0, 0.01, 500))
    pair = pd.Series(rng.normal(0, 0.008, 500))
    target = base - 0.30 * pd.Series(rng.normal(0, 0.02, 500))   # some hedged series
    k = hg.risk_match_weight(base, pair, target)
    lhs = (base + k * pair).std()
    assert lhs == pytest.approx(target.std(), rel=0.02)
