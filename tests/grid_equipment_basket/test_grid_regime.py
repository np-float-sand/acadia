import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import config, grid_regime as gr


# ── _daily_zone_congestion ───────────────────────────────────────────────────

def _hourly_lmp(zones, days, start="2021-01-01", cong_by_zone=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    rows = []
    for z in zones:
        c = (cong_by_zone or {}).get(z, 1.0)
        for t in idx:
            rows.append({"time": t, "location": z,
                         "lmp": 30 + rng.normal(0, 5),
                         "congestion": rng.normal(0, 0.1) - c,  # sign varies; |.| ~ c
                         "energy": 30.0, "loss": 0.1})
    return pd.DataFrame(rows)


def test_daily_zone_congestion_uses_absolute_dollars_and_daily_mean():
    df = _hourly_lmp(["DOM", "AEP"], days=5, cong_by_zone={"DOM": 8.0, "AEP": 2.0})
    out = gr._daily_zone_congestion(df)
    assert list(out.columns) == ["AEP", "DOM"]           # sorted
    assert out.index.name == "date"
    assert len(out) == 5
    # |congestion| mean should sit near the injected magnitude, DOM >> AEP
    assert out["DOM"].mean() > 5.0
    assert out["AEP"].mean() < 4.0
    assert out["DOM"].mean() > out["AEP"].mean()


def test_daily_zone_congestion_empty_frame_returns_empty():
    out = gr._daily_zone_congestion(pd.DataFrame(columns=["time", "location", "lmp", "congestion"]))
    assert out.empty


# ── _trailing_zscore ─────────────────────────────────────────────────────────

def test_trailing_zscore_matches_rolling_formula_and_winsorises():
    s = pd.Series(np.arange(1, 401, dtype=float), index=pd.bdate_range("2020-01-01", periods=400))
    z = gr._trailing_zscore(s, window=60, min_periods=30, winsor=3.0)
    m = s.rolling(60, min_periods=30).mean()
    sd = s.rolling(60, min_periods=30).std()
    expected = ((s - m) / sd).clip(-3.0, 3.0)
    pd.testing.assert_series_equal(z, expected, check_names=False)


def test_trailing_zscore_is_nan_before_min_periods():
    s = pd.Series(np.random.default_rng(0).normal(size=100),
                  index=pd.bdate_range("2020-01-01", periods=100))
    z = gr._trailing_zscore(s, window=60, min_periods=30, winsor=3.0)
    assert z.iloc[:29].isna().all()
    assert z.iloc[30:].notna().any()


def test_trailing_zscore_has_no_lookahead():
    # flat, then a permanent step up on day 200. A z computed at day 199 must be
    # unaffected by the later step (rolling uses only past+current data).
    v = np.r_[np.full(200, 10.0), np.full(200, 40.0)]
    s = pd.Series(v, index=pd.bdate_range("2020-01-01", periods=400))
    z_with_future = gr._trailing_zscore(s, window=60, min_periods=30, winsor=10.0)
    z_truncated = gr._trailing_zscore(s.iloc[:200], window=60, min_periods=30, winsor=10.0)
    pd.testing.assert_series_equal(
        z_with_future.iloc[:200].dropna(), z_truncated.dropna(), check_names=False)


# ── _combine_subsignals ──────────────────────────────────────────────────────

def test_combine_subsignals_is_weight_normalised_mean():
    idx = pd.bdate_range("2021-01-01", periods=10)
    a = pd.Series(1.0, index=idx)
    b = pd.Series(3.0, index=idx)
    out = gr._combine_subsignals({"a": (a, 0.75), "b": (b, 0.25)})
    assert out.eq(1.5).all()                              # 0.75*1 + 0.25*3


def test_combine_subsignals_renormalises_over_present_values():
    idx = pd.bdate_range("2021-01-01", periods=6)
    a = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    b = pd.Series([np.nan, np.nan, 5.0, 5.0, 5.0, 5.0], index=idx)
    out = gr._combine_subsignals({"a": (a, 0.5), "b": (b, 0.5)})
    # where b is NaN, result is just a; elsewhere the 0.5/0.5 blend
    assert out.iloc[0] == 1.0
    assert out.iloc[3] == 3.0


def test_combine_subsignals_drops_zero_weight_entries():
    idx = pd.bdate_range("2021-01-01", periods=4)
    a = pd.Series(2.0, index=idx)
    b = pd.Series(9.0, index=idx)
    out = gr._combine_subsignals({"a": (a, 1.0), "b": (b, 0.0)})
    assert out.eq(2.0).all()


# ── regime_multiplier ────────────────────────────────────────────────────────

def _composite(values, start="2021-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_regime_multiplier_discrete_three_states():
    # one full month each: tight (+1.0), neutral (0.0), loose (-1.0)
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    comp = pd.Series(0.0, index=idx)
    comp.loc["2021-01"] = 1.0
    comp.loc["2021-03"] = -1.0
    m = gr.regime_multiplier(comp, mode="discrete", thresh=0.5, hi=1.25, lo=0.6)
    # month-held: Jan's verdict applies to Feb, Feb's to Mar
    assert (m.loc["2021-02"] == 1.25).all()
    assert (m.loc["2021-03"] == 1.0).all()


def test_regime_multiplier_continuous_clips_to_band():
    comp = _composite([-10.0] * 25 + [10.0] * 25)
    # give it month boundaries so _month_hold has something to carry
    comp.index = pd.bdate_range("2021-01-01", periods=50)
    m = gr.regime_multiplier(comp, mode="continuous", k=0.35)
    assert m.min() >= 0.5 - 1e-9
    assert m.max() <= 1.5 + 1e-9


def test_regime_multiplier_is_month_held_and_lagged():
    idx = pd.bdate_range("2021-01-01", "2021-04-30")
    comp = pd.Series(0.0, index=idx)
    comp.loc["2021-02"] = 2.0                             # only February is tight
    m = gr.regime_multiplier(comp, mode="discrete", thresh=0.5, hi=1.25, lo=0.6)
    for _, chunk in m.groupby(m.index.to_period("M")):
        assert chunk.nunique() == 1                       # constant within a month
    assert (m.loc["2021-03"] == 1.25).all()              # Feb tight -> March levered
    assert (m.loc["2021-02"] != 1.25).all()              # not contemporaneous


def test_regime_multiplier_warmup_nan_is_neutral():
    comp = _composite([np.nan] * 40 + [2.0] * 40)
    comp.index = pd.bdate_range("2021-01-01", periods=80)
    m = gr.regime_multiplier(comp, mode="discrete", thresh=0.5, hi=1.25, lo=0.6)
    assert (m.iloc[:20] == 1.0).all()


# ── regime_composite (injected fetchers) ─────────────────────────────────────

def _fake_lmp_fn(cong_by_zone):
    def _fn(zones, start, end):
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        return _hourly_lmp(zones, days=days, start=start,
                           cong_by_zone={z: cong_by_zone.get(z, 1.0) for z in zones})
    return _fn


def test_regime_composite_fires_on_a_congestion_step_vs_recent_norm():
    # A trailing z-score detects "congestion unusually high vs its recent history",
    # not the level itself: flat for ~8 months, then a sustained step up. The
    # composite should be ~0 in the calm stretch and clearly positive after the step.
    step_day = pd.Timestamp("2021-09-01")

    def lmp_fn(zones, start, end):
        rng = np.random.default_rng(1)
        rows = []
        for d in pd.date_range(start, end, freq="D"):
            dom_c = 8.0 if d >= step_day else 2.0
            for h in range(0, 24, 3):
                t = d + pd.Timedelta(hours=h)
                rows.append({"time": t, "location": "DOM", "lmp": 30.0,
                             "congestion": -dom_c + rng.normal(0, 0.05), "energy": 30.0, "loss": 0.0})
                rows.append({"time": t, "location": "AEP", "lmp": 30.0,
                             "congestion": -2.0 + rng.normal(0, 0.05), "energy": 30.0, "loss": 0.0})
        return pd.DataFrame(rows)

    comp = gr.regime_composite("2021-01-01", "2021-12-31", zones=["DOM", "AEP"],
                               w_cong=1.0, w_reserve=0.0, lmp_fn=lmp_fn,
                               zscore_window=90, zscore_minp=40)
    calm = comp.loc["2021-06-01":"2021-08-15"].mean()
    # a trailing z-score flags the *transition* and then decays as the new level
    # becomes the norm, so measure right after the step
    fired = comp.loc["2021-09-02":"2021-09-30"].mean()
    assert abs(calm) < 0.6
    assert fired > calm + 0.5
    assert comp.loc["2021-09-02":"2021-10-15"].max() > config.REGIME_THRESH


# ── shipped live config (rung 1, adopted 2026-08-31, reverted 2026-09-01) ───

def test_regime_enabled_flag_is_false():
    assert config.REGIME_ENABLED is False


def test_shipped_config_is_ladder_rung_1_without_neighbours():
    cfg = gr.shipped_config()
    assert cfg["mode"] == "discrete"
    assert list(cfg["zones"]) == config.REGIME_ZONES_CORE
    assert cfg["w_cong"] == 1.0 and cfg["w_reserve"] == 0.0
    assert "neighbours" not in cfg


def test_live_multiplier_uses_frozen_rung1_params_and_is_month_held():
    seen = {}

    def composite_fn(start, end, *, zones, w_cong, w_reserve, zone_weight, **kw):
        seen.update(zones=list(zones), w_cong=w_cong, w_reserve=w_reserve)
        idx = pd.bdate_range(start, end)
        v = pd.Series(0.0, index=idx)
        v.loc["2024-01"] = 3.0            # one tight month
        v.loc["2024-03"] = -3.0           # one loose month
        return v

    m = gr.live_multiplier("2023-06-01", "2024-06-30", composite_fn=composite_fn)
    assert seen["zones"] == config.REGIME_ZONES_CORE
    assert seen["w_cong"] == 1.0 and seen["w_reserve"] == 0.0
    assert set(m.dropna().unique()) <= {config.REGIME_LO, 1.0, config.REGIME_HI}
    assert (m.loc["2024-02"] == config.REGIME_HI).all()      # Jan tight -> Feb levered
    assert (m.loc["2024-04"] == config.REGIME_LO).all()      # Mar loose -> Apr de-risked


def test_regime_composite_reserve_weight_blends_in_load_tightness():
    lmp_fn = _fake_lmp_fn({"DOM": 3.0})

    def load_fn(zones, start, end):
        idx = pd.date_range(start, end, freq="h")
        rows = [{"time": t, "zone": "DOM", "load_mw": 10000 + i}      # ramp
                for i, t in enumerate(idx)]
        return pd.DataFrame(rows)

    comp = gr.regime_composite("2021-01-01", "2021-06-30", zones=["DOM"],
                               w_cong=0.5, w_reserve=0.5, lmp_fn=lmp_fn, load_fn=load_fn,
                               zscore_window=40, zscore_minp=20)
    assert comp.notna().any()
