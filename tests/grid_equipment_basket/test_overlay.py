import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import overlay as ov


# ── helpers ──────────────────────────────────────────────────────────────────

def _prices_from_daily(daily: list[float], start="2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(daily) + 1)
    lvl = (1.0 + pd.Series([0.0] + daily, index=idx)).cumprod()
    return lvl


def _returns_from_prices(px: pd.Series) -> pd.Series:
    return px.pct_change().dropna()


# ── trend_gate ───────────────────────────────────────────────────────────────

def test_trend_gate_is_binary_and_flips_after_a_break():
    up = [0.002] * 220          # ~11 months up
    down = [-0.004] * 90        # ~4.5 months down -> price falls below its MA
    px = _prices_from_daily(up + down)
    g = ov.trend_gate(px, ma_days=100)
    assert set(g.dropna().unique()) <= {0.0, 1.0}
    assert g.iloc[150] == 1.0                      # deep in the uptrend
    assert g.iloc[-1] == 0.0                       # after the break


def test_trend_gate_has_no_lookahead():
    # above MA all through month M, then a one-day crater on the last bday of M
    base = [0.001] * 300
    px = _prices_from_daily(base)
    # crater the final observation
    px.iloc[-1] = px.iloc[-2] * 0.5
    g = ov.trend_gate(px, ma_days=50)
    # the gate value for the last day was decided at the prior month-end, when
    # price was still well above its MA -> must still be 1 on that day
    assert g.iloc[-1] == 1.0


def test_trend_gate_defaults_invested_before_ma_is_warm():
    px = _prices_from_daily([0.001] * 40)
    g = ov.trend_gate(px, ma_days=100)
    assert (g.iloc[:20] == 1.0).all()


# ── vol_target_scalar ────────────────────────────────────────────────────────

def test_vol_scalar_hits_cap_on_a_calm_series():
    r = pd.Series(1e-5, index=pd.bdate_range("2020-01-01", periods=300))
    s = ov.vol_target_scalar(r, lookback=20, target_vol=0.20, max_leverage=1.5)
    assert s.dropna().max() == pytest.approx(1.5)
    assert (s.dropna() <= 1.5 + 1e-9).all()


def test_vol_scalar_scales_down_on_a_spiky_series():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.04, 300), index=pd.bdate_range("2020-01-01", periods=300))
    s = ov.vol_target_scalar(r, lookback=20, target_vol=0.20, max_leverage=1.5)
    # realized vol ~0.04*sqrt(252) ~ 0.63 -> scalar ~ 0.32
    assert 0.2 < s.dropna().median() < 0.5


def test_vol_scalar_is_month_held():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 200), index=pd.bdate_range("2020-01-01", periods=200))
    s = ov.vol_target_scalar(r, lookback=20, target_vol=0.20, max_leverage=1.5)
    # within any calendar month the scalar is constant
    for _, chunk in s.dropna().groupby(s.dropna().index.to_period("M")):
        assert chunk.nunique() == 1


# ── apply_overlay ────────────────────────────────────────────────────────────

def _crash_path():
    calm_up = [0.0015] * 260
    crash = list(np.linspace(-0.002, -0.02, 50)) + [-0.015] * 30
    return _prices_from_daily(calm_up + crash)


def test_apply_overlay_cuts_drawdown_and_beats_basket_on_a_crash():
    px = _crash_path()
    r = _returns_from_prices(px)
    overlaid = ov.apply_overlay(r, px, rf_annual=0.04, ma_days=100,
                                vol_lookback=20, target_vol=0.20, max_leverage=1.5)
    base_dd = float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
    ov_dd = float(((1 + overlaid).cumprod() / (1 + overlaid).cumprod().cummax() - 1).min())
    assert ov_dd > base_dd                                   # less negative
    assert (1 + overlaid).prod() > (1 + r).prod()            # ends richer


def test_apply_overlay_levers_up_in_a_calm_uptrend():
    px = _prices_from_daily([0.0005] * 300)
    r = _returns_from_prices(px)
    overlaid = ov.apply_overlay(r, px, rf_annual=0.04, ma_days=50,
                                vol_lookback=20, target_vol=0.20, max_leverage=1.5)
    tail_base = r.iloc[-40:]
    tail_ov = overlaid.iloc[-40:]
    # exposure pinned at the 1.5 cap: overlaid ~= 1.5*r - 0.5*rf_daily
    ratio = (tail_ov + 0.5 * 0.04 / 252) / tail_base
    assert ratio.mean() == pytest.approx(1.5, abs=0.05)


def test_apply_overlay_aligns_to_input_index():
    px = _crash_path()
    r = _returns_from_prices(px)
    out = ov.apply_overlay(r, px)
    assert out.index.equals(r.index)
    assert out.notna().all()


# ── overlay_report ───────────────────────────────────────────────────────────

def _stub_price_fn():
    idx = pd.bdate_range("2022-06-01", "2026-06-30")
    rng = np.random.default_rng(3)
    tickers = ["ETN", "HUBB", "GEV", "VRT", "PWR", "MYRG", "NVT", "FLNC", "PRIM"]
    out = {}
    for j, t in enumerate(tickers):
        steps = rng.normal(0.0007, 0.02, len(idx))
        steps[400:460] = rng.normal(-0.015, 0.03, 60)      # a shared drawdown
        out[t] = 100 * (1 + pd.Series(steps, index=idx)).cumprod()
    for b in ["XLI", "SPY", "XLU", "GRID", "PAVE"]:
        out[b] = 100 * (1 + pd.Series(rng.normal(0.0004, 0.011, len(idx)), index=idx)).cumprod()

    def price_fn(names, start, end):
        return pd.DataFrame({k: v for k, v in out.items() if k in names}).loc[start:end]
    return price_fn


def test_overlay_report_has_all_sections():
    rep = ov.overlay_report("2023-01-01", "2026-06-30", price_fn=_stub_price_fn())
    for k in ("basket_only", "trend_gate", "trend_gate_voltarget"):
        assert {"cagr", "sharpe", "max_dd"} <= set(rep[k]["metrics"])
        assert isinstance(rep[k]["calendar"], pd.Series)
    m = rep["mechanics"]
    assert 0.0 <= m["avg_exposure"] <= 1.6
    assert m["min_exposure"] >= 0.0
    assert m["gate_flips"] >= 0
    grid = rep["plateau_grid"]
    assert len(grid) == 15                                   # 5 MA x 3 target-vol
    assert all(set(cell) >= {"ma_days", "target_vol", "sharpe", "max_dd"} for cell in grid)
