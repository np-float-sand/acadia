"""Unit tests for the grid_demand_factor triage probe (synthetic, deterministic)."""

import numpy as np
import pandas as pd
import pytest

from grid_demand_factor.nowcast import build_monthly_nowcast, seasonality_strength
from grid_demand_factor.sensitivity import (
    evaluate_gate,
    monthly_total_returns,
    quintile_spread,
    rank_ic,
    rolling_sensitivity,
)


def _daily_gsi(start="2018-01-01", periods=365 * 4):
    idx = pd.date_range(start, periods=periods, freq="D", name="date")
    # deterministic seasonal + drift signal in [0, 1]
    t = np.arange(periods)
    ercot = 0.5 + 0.2 * np.sin(2 * np.pi * t / 365.25)
    pjm = 0.5 + 0.15 * np.cos(2 * np.pi * t / 365.25) + 1e-4 * t
    return pd.DataFrame({"ERCOT": ercot, "PJM": pjm}, index=idx)


# ── nowcast ──────────────────────────────────────────────────────────────────

def test_monthly_nowcast_level_is_monthly_and_bounded():
    now = build_monthly_nowcast(_daily_gsi(), how="level", standardize=False)
    assert now.index.freqstr in ("ME", "M")
    assert now.notna().all()
    assert (now.between(0, 1)).all()
    assert 40 <= len(now) <= 49  # ~48 months


def test_monthly_nowcast_change_is_standardized_and_one_shorter():
    lvl = build_monthly_nowcast(_daily_gsi(), how="level", standardize=False)
    chg = build_monthly_nowcast(_daily_gsi(), how="change", standardize=True)
    assert len(chg) == len(lvl) - 1
    assert chg.mean() == pytest.approx(0.0, abs=1e-9)
    assert chg.std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_seasonality_strength_flags_a_pure_seasonal_series():
    # month-of-year mean should explain almost all variance
    idx = pd.date_range("2018-01-31", periods=60, freq="ME")
    pure = pd.Series(np.tile(np.arange(12), 5)[: len(idx)], index=idx, dtype=float)
    s = seasonality_strength(pure)
    assert s["month_r2"] > 0.99


def test_build_monthly_nowcast_rejects_bad_how():
    with pytest.raises(ValueError):
        build_monthly_nowcast(_daily_gsi(), how="sideways")


# ── sensitivity betas ────────────────────────────────────────────────────────

def _panel(nowcast, betas: dict, noise=0.0, seed=0):
    """Build a monthly return panel where col c has return = beta_c * nowcast."""
    rng = np.random.default_rng(seed)
    data = {}
    for c, b in betas.items():
        data[c] = b * nowcast.values + noise * rng.standard_normal(len(nowcast))
    return pd.DataFrame(data, index=nowcast.index)


def test_rolling_sensitivity_recovers_known_beta():
    idx = pd.date_range("2015-01-31", periods=72, freq="ME")
    nowcast = pd.Series(np.random.default_rng(1).standard_normal(72), index=idx, name="nowcast")
    panel = _panel(nowcast, {"A": 2.0, "B": -1.0, "C": 0.0})
    betas = rolling_sensitivity(panel, nowcast, window=36, min_periods=24)
    last = betas.dropna().iloc[-1]
    assert last["A"] == pytest.approx(2.0, abs=1e-6)
    assert last["B"] == pytest.approx(-1.0, abs=1e-6)
    assert last["C"] == pytest.approx(0.0, abs=1e-6)


def test_rolling_sensitivity_respects_min_periods():
    idx = pd.date_range("2015-01-31", periods=30, freq="ME")
    nowcast = pd.Series(np.arange(30, dtype=float), index=idx, name="nowcast")
    panel = _panel(nowcast, {"A": 1.0})
    betas = rolling_sensitivity(panel, nowcast, window=36, min_periods=24)
    assert betas["A"].iloc[:23].isna().all()   # not enough history yet
    assert betas["A"].iloc[23:].notna().all()


# ── quintile spread + rank IC ────────────────────────────────────────────────

def _wide_panel_with_dispersion(n_names=50, n_months=80, ic_sign=1.0, seed=3):
    """Panel where (a) beta is monotone in a latent b_j and (b) a static
    cross-sectional tilt rewards high b_j every month. The tilt (0.05 * b_j)
    dominates the contemporaneous nowcast term (~0.05 * b_j * nowcast) so the
    forward-return ordering is driven by the tilt, not by leakage."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    nc = rng.standard_normal(n_months)
    nc = nc - nc.mean()
    nowcast = pd.Series(nc, index=idx, name="nowcast")
    true_beta = np.linspace(-2, 2, n_names)
    cols = [f"N{i:02d}" for i in range(n_names)]
    rets = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for j, c in enumerate(cols):
        rets[c] = (0.05 * true_beta[j] * nowcast.values            # -> beta ~ 0.05*b_j
                   + ic_sign * 0.05 * true_beta[j]                  # static monotone tilt
                   + 0.003 * rng.standard_normal(n_months))
    return rets, nowcast


def test_quintile_spread_monotone_when_high_beta_outperforms():
    rets, nowcast = _wide_panel_with_dispersion(ic_sign=1.0)
    betas = rolling_sensitivity(rets, nowcast, window=36, min_periods=24)
    res = quintile_spread(betas, rets, min_names=25)
    assert res["n_months"] > 0
    qm = res["quantile_means"]
    assert qm["Q5"] > qm["Q1"]
    assert res["monotonic"]


def test_rank_ic_positive_when_high_beta_outperforms():
    rets, nowcast = _wide_panel_with_dispersion(ic_sign=1.0)
    betas = rolling_sensitivity(rets, nowcast, window=36, min_periods=24)
    res = rank_ic(betas, rets, min_names=25)
    assert res["n_months"] > 0
    assert res["mean_ic"] > 0.05


def test_rank_ic_sign_flips_with_forward_kick():
    rets, nowcast = _wide_panel_with_dispersion(ic_sign=-1.0)
    betas = rolling_sensitivity(rets, nowcast, window=36, min_periods=24)
    res = rank_ic(betas, rets, min_names=25)
    assert res["mean_ic"] < 0


# ── gate ─────────────────────────────────────────────────────────────────────

def test_gate_passes_only_when_both_conditions_met():
    strong = evaluate_gate(
        {"mean_ic": 0.05, "t_stat": 3.1},
        {"ls_sharpe_ann": 0.8, "monotonic": True},
    )
    assert strong["passed"]

    weak_ic = evaluate_gate(
        {"mean_ic": 0.01, "t_stat": 0.9},
        {"ls_sharpe_ann": 0.8, "monotonic": True},
    )
    assert not weak_ic["passed"] and not weak_ic["ic_ok"] and weak_ic["spread_ok"]

    weak_spread = evaluate_gate(
        {"mean_ic": 0.05, "t_stat": 3.1},
        {"ls_sharpe_ann": 0.1, "monotonic": False},
    )
    assert not weak_spread["passed"] and weak_spread["ic_ok"] and not weak_spread["spread_ok"]


def test_gate_handles_nan_inputs():
    res = evaluate_gate(
        {"mean_ic": float("nan"), "t_stat": float("nan")},
        {"ls_sharpe_ann": float("nan"), "monotonic": False},
    )
    assert not res["passed"]


# ── monthly_total_returns ────────────────────────────────────────────────────

def test_monthly_total_returns_from_daily_prices():
    idx = pd.date_range("2020-01-01", "2020-04-30", freq="B")
    px = pd.DataFrame({"A": np.linspace(100, 110, len(idx))}, index=idx)
    mr = monthly_total_returns(px)
    assert list(mr.index.month) == [2, 3, 4]
    assert (mr["A"] > 0).all()
