import numpy as np
import pandas as pd
import pytest

from transmission_rate_base import portfolio as pf


def _daily_index():
    return pd.bdate_range("2014-01-01", "2019-12-31")


def _signal_df():
    tickers = [f"T{i:02d}" for i in range(10)]
    rows = []
    for y in (2014, 2015, 2016, 2017, 2018):
        for i, t in enumerate(tickers):
            rows.append(dict(ticker=t, year=y, neutral_signal=float(i)))  # T09 best, T00 worst
    return pd.DataFrame(rows)


def test_rebalance_dates_are_first_trading_day_from_may():
    idx = _daily_index()
    ds = pf.rebalance_dates(idx, "2015-01-01", "2018-12-31")
    assert [d.year for d in ds] == [2015, 2016, 2017, 2018]
    assert all(d.month == 5 for d in ds)


def test_quintile_ls_weights_dollar_and_sign_structure():
    idx = _daily_index()
    dr = pd.DataFrame(0.0, index=idx, columns=[f"T{i:02d}" for i in range(10)])
    ew = pd.Series(0.0, index=idx)
    w = pf.quintile_ls_weights(_signal_df(), dr, ew)
    row = w.loc["2018-06-01"]
    assert row[row > 0].sum() == pytest.approx(1.0)
    assert row["T09"] > 0 and row["T08"] > 0
    assert row["T00"] < 0 and row["T01"] < 0
    assert row[row < 0].sum() == pytest.approx(-1.0)   # zero returns -> beta ratio 1.0


def test_quintile_ls_weights_beta_scales_short_leg():
    idx = _daily_index()
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    dr = pd.DataFrame(index=idx, columns=[f"T{i:02d}" for i in range(10)], dtype=float)
    for i in range(10):
        beta = 2.0 if i < 2 else (1.0 if i >= 8 else 1.5)   # shorts (T00,T01) beta 2, longs beta 1
        dr[f"T{i:02d}"] = beta * mkt + 0.001 * rng.normal(0, 1, len(idx))
    w = pf.quintile_ls_weights(_signal_df(), dr, mkt)
    row = w.loc["2018-06-01"]
    # short leg gross should be ~ beta_long/beta_short = 1/2 -> ~0.5
    assert row[row < 0].sum() == pytest.approx(-0.5, abs=0.15)


def test_long_tilt_weights_nonneg_and_sum_one():
    idx = _daily_index()
    dr = pd.DataFrame(0.0, index=idx, columns=[f"T{i:02d}" for i in range(10)])
    w = pf.long_tilt_weights(_signal_df(), dr)
    row = w.loc["2018-06-01"]
    assert (row >= 0).all()
    assert row.sum() == pytest.approx(1.0)
    assert row["T09"] > row["T05"] > row["T00"]
