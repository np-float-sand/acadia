import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import backtest as bt


def test_compute_metrics_guard_on_zero_variance():
    out = bt.compute_metrics(pd.Series([0.1, 0.1, 0.1, 0.1]))
    assert out["n_obs"] == 4
    assert np.isnan(out["cagr"])
    assert np.isnan(out["sharpe"])


def test_compute_metrics_max_drawdown_closed_form():
    out = bt.compute_metrics(pd.Series([0.10, -0.50, 0.10]))
    # curve = [1.1, 0.55, 0.605]; peak 1.1 -> min dd = 0.55/1.1 - 1 = -0.5
    assert out["max_dd"] == pytest.approx(-0.5)


def test_compute_metrics_matches_independent_numpy():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.012, 252))
    out = bt.compute_metrics(r, rf_annual=0.04, ann_factor=252)
    growth = float((1 + r).prod())
    exp_cagr = growth ** (252 / len(r)) - 1
    ex = r - 0.04 / 252
    exp_sharpe = ex.mean() / ex.std() * np.sqrt(252)
    assert out["cagr"] == pytest.approx(round(exp_cagr, 4))
    assert out["sharpe"] == pytest.approx(round(float(exp_sharpe), 3))


def test_calendar_year_returns_split_on_year_boundary():
    idx = pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"])
    r = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    cy = bt.calendar_year_returns(r)
    assert set(cy.index) == {2023, 2024}
    assert cy.loc[2023] == pytest.approx((1.01 * 1.02) - 1, abs=1e-4)


def test_relative_metrics_identical_series():
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01],
                  index=pd.bdate_range("2023-01-02", periods=5))
    rel = bt.relative_metrics(r, r.copy())
    assert rel["corr"] == pytest.approx(1.0)
    assert rel["tracking_error"] == pytest.approx(0.0)
    assert rel["excess_cagr"] == pytest.approx(0.0)


def test_relative_metrics_uses_date_intersection():
    idx = pd.bdate_range("2023-01-02", periods=6)
    b = pd.Series([0.01] * 5, index=idx[:5])
    m = pd.Series([0.005] * 6, index=idx)
    rel = bt.relative_metrics(b, m)
    assert not np.isnan(rel["information_ratio"])


def test_run_with_synthetic_price_fn():
    idx = pd.bdate_range("2023-01-02", periods=260)
    rng = np.random.default_rng(1)

    def _prices(tickers, start, end):
        data = {}
        for k, t in enumerate(sorted(tickers)):
            steps = rng.normal(0.0005 + 0.0001 * k, 0.01, len(idx))
            data[t] = 100 * np.exp(np.cumsum(steps))
        return pd.DataFrame(data, index=idx)

    res = bt.run(
        "2023-01-02", str(idx[-1].date()),
        universe=["AAA", "BBB", "CCC"], benchmarks=["XLI", "GRID"],
        price_fn=_prices,
    )
    assert set(res["benchmarks"]) == {"XLI", "GRID"}
    assert res["basket"]["n_obs"] > 200
    assert np.isfinite(res["basket"]["sharpe"])
    tbl = bt.results_table(res)
    assert "CAGR" in tbl and "XLI" in tbl


def test_run_warns_on_universe_name_missing_from_prices():
    idx = pd.bdate_range("2023-01-02", periods=260)
    rng = np.random.default_rng(2)
    # 5-name universe, one name ("EEE") absent from the price frame -> 4 names
    # remain (cap 0.25 stays feasible, so no apply_cap noise), and run() warns.

    def _prices(tickers, start, end):
        keep = [t for t in sorted(tickers) if t != "EEE"]
        return pd.DataFrame(
            {t: 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(idx)))) for t in keep},
            index=idx,
        )

    with pytest.warns(RuntimeWarning, match="absent from price data"):
        res = bt.run(
            "2023-01-02", str(idx[-1].date()),
            universe=["AAA", "BBB", "CCC", "DDD", "EEE"], benchmarks=["XLI"],
            price_fn=_prices,
        )
    assert np.isfinite(res["basket"]["sharpe"])
