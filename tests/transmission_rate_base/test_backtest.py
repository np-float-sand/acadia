import numpy as np
import pandas as pd
import pytest

from transmission_rate_base import backtest as bt


def test_run_backtest_hand_computed_two_months():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    w = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    w["A"] = 1.0
    w["B"] = -1.0
    mr = pd.DataFrame({"A": [np.nan, 0.10, 0.05], "B": [np.nan, 0.02, -0.03]},
                      index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]))
    ret, m = bt.run_backtest(w, mr)
    assert ret.loc["2020-02-29"] == pytest.approx(0.08)
    assert ret.loc["2020-03-31"] == pytest.approx(0.08)
    assert "2020-01-31" not in ret.index          # all-NaN month dropped
    assert m["hit_rate"] == pytest.approx(1.0)
    assert m["positive_years_frac"] == pytest.approx(1.0)


def test_annual_returns_compounds_within_year():
    s = pd.Series([0.1, -0.05, 0.2], index=pd.to_datetime(["2020-06-30", "2020-12-31", "2021-06-30"]))
    yr = bt.annual_returns(s)
    assert yr.loc[2020] == pytest.approx(1.1 * 0.95 - 1)
    assert yr.loc[2021] == pytest.approx(0.2)


def test_signal_rank_ic_positive_when_high_signal_outperforms():
    idx = pd.bdate_range("2015-01-01", "2019-12-31")
    tickers = [f"T{i:02d}" for i in range(12)]
    drift = {t: (i - 6) * 0.0002 for i, t in enumerate(tickers)}
    dr = pd.DataFrame({t: drift[t] for t in tickers}, index=idx)
    ew = dr.mean(axis=1)
    sdf = pd.DataFrame([dict(ticker=t, year=y, neutral_signal=float(i))
                        for y in (2015, 2016, 2017, 2018) for i, t in enumerate(tickers)])
    from transmission_rate_base.portfolio import rebalance_dates
    rd = rebalance_dates(idx, "2016-01-01", "2019-12-31")
    out = bt.signal_rank_ic(sdf, dr, ew, rd)
    assert out["mean_ic"] > 0.5
    assert out["n"] >= 2
