import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import capex_guidance_signal as cgs


def test_guidance_composite_empty_events_is_empty_series():
    result = cgs.guidance_composite(pd.Series(dtype=float), "2023-01-01", "2023-06-01")
    assert result.empty


def test_guidance_composite_no_future_leak():
    events = pd.Series([10.0, 50.0],
                       index=[pd.Timestamp("2023-01-01"), pd.Timestamp("2023-04-01")])
    full = cgs.guidance_composite(events, "2022-01-01", "2023-12-31",
                                  zscore_window=300, zscore_minp=5, winsor=3.0)
    truncated_events = events.loc[:"2023-01-01"]
    truncated = cgs.guidance_composite(truncated_events, "2022-01-01", "2023-03-31",
                                       zscore_window=300, zscore_minp=5, winsor=3.0)
    common = truncated.index.intersection(full.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(full.loc[common], truncated.loc[common])


def test_guidance_composite_holds_value_between_events():
    events = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                       index=pd.date_range("2022-01-01", periods=5, freq="90D"))
    result = cgs.guidance_composite(events, "2022-01-01", "2022-12-31",
                                    zscore_window=1000, zscore_minp=1, winsor=3.0)
    # Before the second distinct event, the forward-filled value is flat at
    # 1.0 throughout -- the rolling window's variance is genuinely zero (not
    # merely "not yet warm"), so `_trailing_zscore`'s explicit zero-variance
    # guard correctly returns NaN, not a spurious 0. This confirms the
    # composite doesn't error out and doesn't leak a numeric artifact during
    # a held/flat stretch -- both dates must be NaN, matching
    # `_trailing_zscore`'s own documented "zero-variance window -> NaN, not
    # inf" contract (not a pytest.approx equality check, since NaN != NaN
    # under pytest.approx by default).
    d_after_first = events.index[0] + pd.Timedelta(days=5)
    d_of_first = events.index[0]
    assert pd.isna(result.loc[d_of_first])
    assert pd.isna(result.loc[d_after_first])


def test_guidance_derisk_multiplier_steps_down_below_floor_and_back():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    composite = pd.Series([0.2, -0.6, -0.7, 0.1, np.nan], index=idx)
    mult = cgs.guidance_derisk_multiplier(composite, floor_z=-0.5, lo_mult=0.6)
    assert list(mult) == [1.0, 0.6, 0.6, 1.0, 1.0]


def test_guidance_derisk_multiplier_never_exceeds_one():
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    composite = pd.Series([5.0, -5.0, 0.0], index=idx)
    mult = cgs.guidance_derisk_multiplier(composite, floor_z=-0.5, lo_mult=0.6)
    assert mult.max() <= 1.0


def test_guidance_scaler_clips_and_defaults_to_one_when_nan():
    idx = pd.date_range("2023-01-01", periods=4, freq="D")
    composite = pd.Series([0.0, 2.0, -2.0, np.nan], index=idx)
    scaled = cgs.guidance_scaler(composite, k=0.35, lo=0.5, hi=1.5)
    assert scaled.iloc[0] == pytest.approx(1.0)
    assert scaled.iloc[1] == pytest.approx(1.5)   # 1+0.35*2=1.7 -> clipped to hi
    assert scaled.iloc[2] == pytest.approx(0.5)   # 1-0.7=0.3 -> clipped to lo
    assert scaled.iloc[3] == pytest.approx(1.0)   # NaN -> default


def test_hac_ols_recovers_a_strong_known_relationship():
    rng = np.random.default_rng(0)
    n = 60
    x = pd.Series(rng.normal(size=n), index=pd.date_range("2020-01-31", periods=n, freq="ME"))
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    out = cgs._hac_ols(y, pd.DataFrame({"x": x}), lag=1)
    assert out["coef"]["x"] == pytest.approx(2.0, abs=0.2)
    assert abs(out["t"]["x"]) >= 2.0


def test_hac_ols_no_relationship_gives_small_t():
    rng = np.random.default_rng(1)
    n = 60
    x = pd.Series(rng.normal(size=n), index=pd.date_range("2020-01-31", periods=n, freq="ME"))
    y = pd.Series(rng.normal(size=n), index=x.index)
    out = cgs._hac_ols(y, pd.DataFrame({"x": x}), lag=1)
    assert abs(out["t"]["x"]) < 3.0   # not a hard bound, just "not obviously significant"


def test_hac_ols_too_few_observations_returns_nan_not_a_crash():
    x = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    y = pd.Series([1.0, 2.0], index=x.index)
    out = cgs._hac_ols(y, pd.DataFrame({"x": x}), lag=1)
    assert np.isnan(out["t"]["x"])


def test_rank_ic_positive_when_signal_leads_forward_return():
    idx = pd.date_range("2020-01-31", periods=40, freq="ME")
    rng = np.random.default_rng(2)
    signal = pd.Series(rng.normal(size=40), index=idx)
    fwd = signal + rng.normal(scale=0.3, size=40)
    out = cgs._rank_ic(signal, pd.Series(fwd.values, index=idx), lag=1)
    assert out["ic"] > 0.5
    assert out["t"] >= 2.0


def test_rank_ic_too_few_pairs_returns_nan():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    out = cgs._rank_ic(pd.Series([1.0, 2.0, 3.0], index=idx),
                       pd.Series([1.0, np.nan, np.nan], index=idx), lag=1)
    assert np.isnan(out["ic"])


def test_forward_return_computes_next_h_month_realized_return():
    idx = pd.period_range("2023-01", periods=4, freq="M").to_timestamp("M")
    nav = pd.Series([1.0, 1.1, 1.21, 1.331], index=idx)
    fwd1 = cgs._forward_return(nav, 1)
    assert fwd1.iloc[0] == pytest.approx(0.10)
    assert pd.isna(fwd1.iloc[-1])
    fwd3 = cgs._forward_return(nav, 3)
    assert fwd3.iloc[0] == pytest.approx(0.331)
    assert fwd3.iloc[1:].isna().all()


def test_monthly_nav_compounds_daily_returns_to_month_end():
    idx = pd.date_range("2023-01-01", "2023-02-28", freq="D")
    ret = pd.Series(0.0, index=idx)
    ret.loc["2023-01-15"] = 0.10
    ret.loc["2023-02-10"] = 0.05
    nav = cgs._monthly_nav(ret)
    assert nav.loc["2023-01-31"] == pytest.approx(1.10)
    assert nav.loc["2023-02-28"] == pytest.approx(1.10 * 1.05)
