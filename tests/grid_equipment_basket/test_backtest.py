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


def _vc_price_fn():
    idx = pd.bdate_range("2023-01-02", periods=650)
    rng = np.random.default_rng(7)

    def _prices(tickers, start, end):
        data = {}
        for k, t in enumerate(sorted(tickers)):
            drift = 0.0006 if t in ("ETN", "HUBB", "GEV", "VRT", "NVT") else 0.0003
            data[t] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, len(idx))))
        return pd.DataFrame(data, index=idx).loc[start:end]

    return _prices, idx


def _vc_price_fn_2020():
    # 2020-01 .. 2022-12 window (prior-regime panel) — no 2024-H2 drawdown episode.
    idx = pd.bdate_range("2020-01-02", periods=780)
    rng = np.random.default_rng(13)

    def _prices(tickers, start, end):
        data = {}
        for k, t in enumerate(sorted(tickers)):
            drift = 0.0006 if t in ("ETN", "HUBB", "GEV", "VRT", "NVT") else 0.0003
            data[t] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, len(idx))))
        return pd.DataFrame(data, index=idx).loc[start:end]

    return _prices, idx


def _vc_calm_uptrend_price_fn():
    # Smooth compounding uptrend, ~zero realized vol -> close never below its MA and
    # rv never exceeds its median -> the conditional-short mask never engages.
    idx = pd.bdate_range("2023-01-02", periods=650)

    def _prices(tickers, start, end):
        data = {}
        for k, t in enumerate(sorted(tickers)):
            drift = 0.0005 + 0.00003 * k
            data[t] = 100 * np.exp(np.cumsum(np.full(len(idx), drift)))
        return pd.DataFrame(data, index=idx).loc[start:end]

    return _prices, idx


def _vc_fund_backlog():
    # 8 clean quarters for all 9 names; makers rising margin, contractors flat.
    from grid_equipment_basket import config
    from io import StringIO
    from grid_equipment_basket.backlog_data import load_backlog_csv
    qe = ["2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30",
          "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30",
          "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30"]
    av = ["2022-11-01", "2023-02-01", "2023-05-01", "2023-08-01",
          "2023-11-01", "2024-02-01", "2024-05-01", "2024-08-01",
          "2024-11-01", "2025-02-01", "2025-05-01", "2025-08-01"]
    rows = []
    for t in config.BUCKET_MAKERS + config.BUCKET_CONTRACTORS:
        rising = t in config.BUCKET_MAKERS
        for i, (q, a) in enumerate(zip(qe, av)):
            m = 0.20 + (0.01 * i if rising else 0.0)
            rows.append((pd.Timestamp(q), pd.Timestamp(a), 1000.0, 1000.0 * m, t))
    fund = pd.DataFrame(rows, columns=["quarter_end", "availability_date", "revenue", "gross_profit", "ticker"])
    bl = load_backlog_csv(StringIO("ticker,quarter_end,availability_date,metric_value,metric_unit,"
                                   "disclosure_type,segment_scope,source_url,notes\n"))
    return fund, bl


def test_value_chain_report_shape_and_gates():
    price_fn, idx = _vc_price_fn()
    fund, bl = _vc_fund_backlog()
    rep = bt.value_chain_report("2023-01-02", str(idx[-1].date()),
                                price_fn=price_fn, fund_df=fund, backlog_df=bl)
    assert set(rep["benchmarks"]) <= {"XLI", "SPY", "XLU", "GRID", "PAVE"}
    assert isinstance(rep["gate1"]["passed"], bool)
    assert isinstance(rep["gate2"]["passed"], bool)
    assert "peak" in rep["episode"] and "trough" in rep["episode"]
    for key in ("pair_30", "cond_short", "pair_risk_matched"):
        assert "episode_dd" in rep["overlays"][key]
    tbl = bt.value_chain_table(rep)
    assert "GATE 1" in tbl and "GATE 2" in tbl


def test_value_chain_report_drop_winners_removes_vrt_gev_from_makers():
    price_fn, idx = _vc_price_fn()
    fund, bl = _vc_fund_backlog()
    rep = bt.value_chain_report("2023-01-02", str(idx[-1].date()), price_fn=price_fn,
                                fund_df=fund, backlog_df=bl, drop_winners=True)
    assert rep["drop_winners"] is True
    assert {"VRT", "GEV"}.isdisjoint(rep["makers"])
    assert np.isfinite(rep["value_chain_tilt"]["sharpe"])


def test_value_chain_report_prior_regime_window_does_not_crash():
    # --prior-regime (2020-2022) has no data in the hard-coded 2024-H2 episode window;
    # Gate 1 + standalone rows must still compute, Gate 2 becomes not-applicable.
    price_fn, _ = _vc_price_fn_2020()
    fund, bl = _vc_fund_backlog()
    rep = bt.value_chain_report("2020-01-01", "2022-12-31",
                                price_fn=price_fn, fund_df=fund, backlog_df=bl)
    assert np.isfinite(rep["gate1"]["tilt_sharpe"])
    assert np.isfinite(rep["gate1"]["ew_cagr"])
    assert isinstance(rep["gate1"]["passed"], bool)
    assert rep["episode"]["peak"] is None
    assert rep["gate2"]["passed"] is False
    assert rep["gate2"]["applicable"] is False
    tbl = bt.value_chain_table(rep)                       # must not raise
    assert "GATE 1" in tbl and "GATE 2: n/a" in tbl


def test_value_chain_report_conditional_short_never_engages_gives_zero_carry():
    # All-False conditional-short mask -> zero-variance hedge sleeve. carry must be
    # 0.0 (a hedge that never fires has zero carry), NOT NaN auto-failing Gate 2.
    price_fn, idx = _vc_calm_uptrend_price_fn()
    fund, bl = _vc_fund_backlog()
    rep = bt.value_chain_report("2023-01-02", str(idx[-1].date()),
                                price_fn=price_fn, fund_df=fund, backlog_df=bl)
    assert rep["overlays"]["cond_short"]["carry"] == 0.0
    assert rep["gate2"]["cond_carry"] == 0.0
    assert not np.isnan(rep["gate2"]["cond_carry"])
    assert isinstance(rep["gate2"]["passed"], bool)


# ── coverage_report (handoff 2026-09-01 §8.3(a): coverage alone) ────────────

def _coverage_fund_backlog(idx):
    """9 names, 12 clean quarters (2021-Q4 through 2024-Q2) -- the extra four
    early quarters give ``coverage_change_signal``'s year-ago lookback a full
    4-quarter TTM-revenue window at the earliest rebalance in the test range,
    not just at the latest one. Rising-coverage names (even index) get a
    growing backlog against flat revenue; falling-coverage names (odd index)
    get a shrinking backlog -- and their price drift is wired to match, so a
    working tilt should beat equal-weight in this fixture."""
    from grid_equipment_basket import config
    from io import StringIO
    from grid_equipment_basket.backlog_data import load_backlog_csv

    names = sorted(config.UNIVERSE)
    qe = ["2021-12-31", "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
         "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
         "2024-03-31", "2024-06-30"]
    av = ["2022-02-01", "2022-05-01", "2022-08-01", "2022-11-01", "2023-02-01",
         "2023-05-01", "2023-08-01", "2023-11-01", "2024-02-01",
         "2024-05-01", "2024-08-01"]
    fund_rows, bl_lines = [], []
    rising_names = set()
    for k, t in enumerate(names):
        rising = (k % 2 == 0)
        if rising:
            rising_names.add(t)
        for i, (q, a) in enumerate(zip(qe, av)):
            fund_rows.append((pd.Timestamp(q), pd.Timestamp(a), 1000.0, 200.0, t))
            backlog = 1000 + (100 * i if rising else -80 * i)
            bl_lines.append(f"{t},{q},{a},{backlog},USD_million,xbrl_rpo,total,http://x,\n")
    fund = pd.DataFrame(fund_rows, columns=["quarter_end", "availability_date", "revenue", "gross_profit", "ticker"])
    header = ("ticker,quarter_end,availability_date,metric_value,metric_unit,"
             "disclosure_type,segment_scope,source_url,notes\n")
    bl = load_backlog_csv(StringIO(header + "".join(bl_lines)))
    return fund, bl, rising_names


def _coverage_price_fn(rising_names):
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(11)

    def _prices(tickers, start, end):
        data = {}
        for t in sorted(tickers):
            drift = 0.0008 if t in rising_names else 0.0002
            data[t] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(idx))))
        return pd.DataFrame(data, index=idx).loc[start:end]

    return _prices, idx


def test_coverage_report_shape_and_gate():
    fund, bl, rising_names = _coverage_fund_backlog(None)
    price_fn, idx = _coverage_price_fn(rising_names)
    rep = bt.coverage_report("2023-06-01", str(idx[-1].date()),
                             price_fn=price_fn, fund_df=fund, backlog_df=bl)
    assert set(rep) >= {"equal_weight", "coverage_tilt", "gate"}
    assert np.isfinite(rep["equal_weight"]["sharpe"])
    assert np.isfinite(rep["coverage_tilt"]["sharpe"])
    assert isinstance(rep["gate"]["passed"], bool)


def test_coverage_report_tilt_beats_equal_weight_when_signal_is_real():
    fund, bl, rising_names = _coverage_fund_backlog(None)
    price_fn, idx = _coverage_price_fn(rising_names)
    rep = bt.coverage_report("2023-06-01", str(idx[-1].date()),
                             price_fn=price_fn, fund_df=fund, backlog_df=bl)
    assert rep["gate"]["tilt_sharpe"] > rep["gate"]["ew_sharpe"]
    assert rep["gate"]["passed"] is True
