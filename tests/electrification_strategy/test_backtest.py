import numpy as np
import pandas as pd
import pytest

from electrification_strategy import backtest, config


def _cand(prices):
    from electrification_strategy import universe
    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    return prices[[c for c in cand if c in prices.columns]]


def test_core_return_runs_and_is_daily(synthetic_prices):
    spy = synthetic_prices["SPY"].pct_change()
    r = backtest.core_return(_cand(synthetic_prices), "marquee",
                             "2018-01-01", "2024-12-31", spy, use_valuation=False)
    assert isinstance(r, pd.Series)
    assert r.index.is_monotonic_increasing
    assert r.notna().mean() > 0.9
    assert r.abs().mean() < 0.1  # vol-targeted daily returns are small


def test_valuation_layer_only_reduces_or_holds_exposure(synthetic_prices):
    spy = synthetic_prices["SPY"].pct_change()
    plain = backtest.core_return(_cand(synthetic_prices), "frozen",
                                 "2018-01-01", "2024-12-31", spy, use_valuation=False)
    val = backtest.core_return(_cand(synthetic_prices), "frozen",
                               "2018-01-01", "2024-12-31", spy, use_valuation=True)
    assert val.index.equals(plain.index)
    assert val.std() <= plain.std() * 1.05


def test_episode_drawdowns_keys_and_sign():
    idx = pd.bdate_range("2019-06-01", "2026-08-31")
    lvl = pd.Series(100.0, index=idx)
    seg = lvl.loc["2022-01-03":"2022-10-03"]
    lvl.loc["2022-01-03":"2022-10-03"] = np.linspace(100, 70, len(seg))
    lvl.loc["2022-10-04":] = 72.0
    r = lvl.pct_change().dropna()
    dd = backtest.episode_drawdowns(r, config.EPISODES)
    assert set(dd) == set(config.EPISODES)
    assert dd["Rate shock 22"] < -0.15
    assert np.isnan(dd["COVID 20"])  # no data before 2019-06 peak window


def test_feared_pnl_sign():
    idx = pd.bdate_range("2021-01-01", "2021-12-31")
    plain = pd.Series(0.0, index=idx)
    plain.loc["2021-03-01":"2021-03-31"] = -0.01
    cell = plain + 0.002
    tan = pd.Series(0.0, index=idx)
    tan.loc["2021-03-01":"2021-03-31"] = 0.01
    fp = backtest.feared_pnl(cell, plain, tan)
    assert fp > 0


def test_effective_n():
    assert backtest._effective_n(pd.Series([0.25, 0.25, 0.25, 0.25])) == pytest.approx(4.0)
    assert backtest._effective_n(pd.Series([0.5, 0.5, 0.0])) == pytest.approx(2.0)


# -- Task 8: grid / winner / outputs ----------------------------------
def test_run_comparison_grid_shape(synthetic_prices, synthetic_dfii10):
    res = backtest.run_comparison("2018-01-01", "2024-12-31", synthetic_prices, synthetic_dfii10)
    assert set(res["cells"]) == {
        (con, lab) for con in ("marquee", "frozen", "thematic", "screen")
        for lab in ("plain", "+val", "+val+sleeve", "+val+sleeve+short")
    }
    for cell in res["cells"].values():
        assert set(cell) >= {"metrics", "beta", "corr_volt", "episode_dd", "feared_pnl",
                             "calm_drag", "subwindow_sharpe", "drag_vs_plain", "returns"}
        assert set(cell["episode_dd"]) == set(config.EPISODES)
    assert {"marquee", "frozen", "thematic", "screen"} == set(res["eff_n"])
    assert res["eff_n"]["screen"] > res["eff_n"]["marquee"]   # supplier screen is deeper
    assert len(res["plateau"]) == len(config.VAL_PLATEAU_SCALES)
    assert {"SPY", "XLI", "PAVE", "GRID", "VOLT"} <= set(res["benchmarks"])


def test_pick_winner_returns_a_cell_or_none(synthetic_prices, synthetic_dfii10):
    res = backtest.run_comparison("2018-01-01", "2024-12-31", synthetic_prices, synthetic_dfii10)
    win = backtest.pick_winner(res)
    assert win["winner"] is None or win["winner"] in res["cells"]
    assert 0 <= win["n_passing"] <= len(res["cells"])


def test_pick_winner_respects_maxdd_gate():
    def cell(sharpe, maxdd, feared, drag, subs):
        return {"metrics": {"sharpe": sharpe, "max_dd": maxdd, "cagr": 0.1},
                "feared_pnl": feared, "drag_vs_plain": drag,
                "subwindow_sharpe": subs, "returns": None, "beta": 0.0, "corr_volt": 0.0,
                "episode_dd": {}, "calm_drag": {}}
    res = {"cells": {
        ("frozen", "plain"): cell(2.0, -0.40, 0.0, 0.0, {"2019-22": 1.0, "2023-26": 1.0}),
        ("frozen", "+val"): cell(0.9, -0.20, 0.0, 0.02, {"2019-22": 0.9, "2023-26": 1.2}),
        ("frozen", "+val+sleeve"): cell(1.5, -0.25, -0.10, 0.03, {"2019-22": 0.8, "2023-26": 1.0}),
    }}
    win = backtest.pick_winner(res)
    assert win["winner"] == ("frozen", "+val")
    assert win["n_passing"] == 1


def test_write_outputs(tmp_path, synthetic_prices, synthetic_dfii10):
    res = backtest.run_comparison("2018-01-01", "2024-12-31", synthetic_prices, synthetic_dfii10)
    win = backtest.pick_winner(res)
    backtest.write_outputs(res, win, tmp_path, make_plot=False)
    for f in ("metrics.csv", "episode_drawdowns.csv", "plateau.json", "returns.csv"):
        assert (tmp_path / f).exists()
    assert not (tmp_path / "performance.png").exists()
