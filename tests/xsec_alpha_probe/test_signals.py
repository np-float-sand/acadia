import numpy as np
import pandas as pd
import pytest

from research.xsec_alpha_probe import signals as sg


def _facts(ticker="AAA", item="revenue", n=12, growth=0.0, filed_lag=40, start="2016-01-01"):
    """n discrete quarters; value grows by `growth` per quarter, filed `filed_lag` days after end."""
    rows = []
    s = pd.Timestamp(start)
    for q in range(n):
        st = s + pd.DateOffset(months=3 * q)
        en = st + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        rows.append(dict(ticker=ticker, item=item, tag="T", start=st, end=en,
                         filed=en + pd.Timedelta(days=filed_lag),
                         value=100 * (1 + growth) ** q))
    return pd.DataFrame(rows)


def test_regular_tail_keeps_contiguous_quarters_only():
    ends = pd.to_datetime(["2018-03-31", "2018-06-30", "2018-12-31", "2019-03-31", "2019-06-30"])
    s = pd.Series([1, 2, 3, 4, 5.0], index=ends)
    tail = sg._regular_tail(s)
    assert list(tail.values) == [3, 4, 5]


def test_features_constant_growth_has_zero_acceleration():
    q = pd.Series(100 * 1.05 ** np.arange(12), index=pd.date_range("2016-03-31", periods=12, freq="QE"))
    f = sg._features_from_series(q, None, None)
    assert f["rev_accel"] == pytest.approx(0.0, abs=1e-9)
    assert f["rev_growth"] > 0


def test_features_sue_positive_after_jump():
    vals = list(100 + np.arange(11) * 1.0) + [200.0]
    q = pd.Series(vals, index=pd.date_range("2016-03-31", periods=12, freq="QE"))
    f = sg._features_from_series(q, q, None)
    assert f["rev_sue"] > 3
    assert f["ni_sue"] > 3


def test_fundamental_panel_is_point_in_time():
    fr = _facts(n=12, growth=0.02)
    # a fact for a 13th quarter filed far in the future must not be visible at asof
    late = _facts(n=13, growth=0.02).iloc[[-1]].assign(filed=pd.Timestamp("2030-01-01"), value=1e9)
    asof = pd.DatetimeIndex([pd.Timestamp("2019-06-30")])
    a = sg.fundamental_panel(pd.concat([fr, late]), asof)
    b = sg.fundamental_panel(fr, asof)
    pd.testing.assert_frame_equal(a, b)


def test_fundamental_panel_drops_stale_filers():
    fr = _facts(n=10, growth=0.02)   # last quarter ends 2018-06-30
    out = sg.fundamental_panel(fr, pd.DatetimeIndex([pd.Timestamp("2020-06-30")]))
    assert out.empty or out["rev_growth"].isna().all()


def test_beta_times_shock_flips_rank_with_shock_sign():
    beta = pd.DataFrame({"a": [1.0, 1.0], "b": [-1.0, -1.0]}, index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    shock = pd.Series([0.5, -0.5], index=beta.index)
    out = sg.beta_x_shock(beta, shock)
    assert out.iloc[0]["a"] > out.iloc[0]["b"]
    assert out.iloc[1]["a"] < out.iloc[1]["b"]
