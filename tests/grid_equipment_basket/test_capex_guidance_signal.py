import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import capex_guidance_signal as cgs


def test_guidance_composite_empty_events_is_empty_series():
    result = cgs.guidance_composite(pd.Series(dtype=float), "2023-01-01", "2023-06-01")
    assert result.empty
    # A default RangeIndex here would blow up downstream: `timing_report`
    # resamples the composite to month-end and slices it with date STRINGS.
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.resample("ME").last().empty
    assert result.loc["2023-01-01":"2023-06-01"].empty


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


def test_guidance_composite_is_nan_through_a_zero_variance_stretch():
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


def _panel(rows):
    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def test_feasibility_gate_uses_full_panel_when_enough_usable(monkeypatch):
    monkeypatch.setattr(cgs.config, "DC_GUIDANCE_MIN_UTILITIES", 1)
    df = _panel([{"utility": "D", "report_date": "2023-06-01",
                  "capex_plan_usd_m": 1000.0, "revision_vs_prior_usd_m": 50.0}])
    panel, feas, used = cgs.feasibility_gate(df, ("2023-01-01", "2026-08-31"))
    assert used == "full"
    assert feas["n_usable"] == 1


def test_feasibility_gate_falls_back_when_not_enough_usable():
    rows = [{"utility": "D", "report_date": "2023-06-01",
            "capex_plan_usd_m": 1000.0, "revision_vs_prior_usd_m": 50.0},
           {"utility": "AEP", "report_date": "2023-06-01",
            "capex_plan_usd_m": 900.0, "revision_vs_prior_usd_m": 40.0},
           {"utility": "ZZZ", "report_date": "1999-01-01",
            "capex_plan_usd_m": 10.0, "revision_vs_prior_usd_m": np.nan}]
    df = _panel(rows)
    panel, feas, used = cgs.feasibility_gate(df, ("2023-01-01", "2026-08-31"))
    assert used == "fallback"
    assert set(panel["utility"]) == {"D", "AEP"}


def test_feasibility_gate_not_testable_when_fallback_also_empty():
    df = _panel([{"utility": "ZZZ", "report_date": "1999-01-01",
                 "capex_plan_usd_m": 10.0, "revision_vs_prior_usd_m": np.nan}])
    panel, feas, used = cgs.feasibility_gate(df, ("2023-01-01", "2026-08-31"))
    assert used == "not_testable"
    assert panel is None


def test_timing_report_combines_rank_ic_and_control_regression_pass_conditions(monkeypatch):
    idx = pd.date_range("2023-01-31", periods=12, freq="ME")
    composite = pd.Series(np.arange(12, dtype=float), index=idx)
    basket_ret = pd.Series(0.001, index=pd.date_range("2023-01-01", "2023-12-31", freq="D"))
    controls = pd.DataFrame({"d10y": 0.0, "smh": 0.0}, index=idx)

    monkeypatch.setattr(cgs, "_rank_ic", lambda signal, fwd, lag: {"ic": 0.9, "t": 5.0, "n": 10})
    monkeypatch.setattr(cgs, "_hac_ols",
                        lambda y, X, lag: {"coef": {c: 0.0 for c in X.columns},
                                           "t": {c: 5.0 for c in X.columns}, "n": 10})
    rep = cgs.timing_report({"strong": composite}, basket_ret, controls,
                            primary=("2023-01-01", "2023-12-31"),
                            holdout=("2023-10-01", "2023-12-31"), horizons=(1,))
    assert rep["strong"][1]["passed"] is True

    monkeypatch.setattr(cgs, "_rank_ic", lambda signal, fwd, lag: {"ic": 0.0, "t": 0.5, "n": 10})
    rep2 = cgs.timing_report({"strong": composite}, basket_ret, controls,
                             primary=("2023-01-01", "2023-12-31"),
                             holdout=("2023-10-01", "2023-12-31"), horizons=(1,))
    assert rep2["strong"][1]["passed"] is False

    # The converse veto (spec s5.4 is an AND, not an OR): a strong rank-IC with
    # a weak without-hyperscaler control-t must also fail.
    monkeypatch.setattr(cgs, "_rank_ic", lambda signal, fwd, lag: {"ic": 0.9, "t": 5.0, "n": 10})
    monkeypatch.setattr(cgs, "_hac_ols",
                        lambda y, X, lag: {"coef": {c: 0.0 for c in X.columns},
                                           "t": {c: 0.5 for c in X.columns}, "n": 10})
    rep3 = cgs.timing_report({"strong": composite}, basket_ret, controls,
                             primary=("2023-01-01", "2023-12-31"),
                             holdout=("2023-10-01", "2023-12-31"), horizons=(1,))
    assert rep3["strong"][1]["passed"] is False


def test_timing_report_adds_hyperscaler_control_only_when_column_present():
    idx = pd.date_range("2023-01-31", periods=6, freq="ME")
    composite = pd.Series(np.arange(6, dtype=float), index=idx)
    basket_ret = pd.Series(0.001, index=pd.date_range("2023-01-01", "2023-06-30", freq="D"))
    controls_without = pd.DataFrame({"d10y": 0.0, "smh": 0.0}, index=idx)
    rep = cgs.timing_report({"c": composite}, basket_ret, controls_without,
                            primary=("2023-01-01", "2023-06-30"),
                            holdout=("2023-05-01", "2023-06-30"), horizons=(1,))
    assert rep["c"][1]["control_with_hyperscaler"]["n"] == 0

    controls_with = controls_without.assign(bigfour=1.0)
    rep_w = cgs.timing_report({"c": composite}, basket_ret, controls_with,
                              primary=("2023-01-01", "2023-06-30"),
                              holdout=("2023-05-01", "2023-06-30"), horizons=(1,))
    assert "bigfour" in rep_w["c"][1]["control_with_hyperscaler"]["t"]


def _derisk_inputs(seed: int = 4):
    """The shared `ret`/`lvl` construction for `derisk_scaler_report` tests."""
    idx = pd.bdate_range("2021-01-01", "2026-08-31")
    rng = np.random.default_rng(seed)
    ret = pd.Series(rng.normal(0.0004, 0.02, size=len(idx)), index=idx)
    lvl = (1.0 + ret).cumprod()
    return idx, rng, ret, lvl


def test_derisk_scaler_report_builds_full_parameter_grids():
    idx, rng, ret, lvl = _derisk_inputs()
    composite = pd.Series(rng.normal(size=len(idx)), index=idx)

    rep = cgs.derisk_scaler_report(ret, lvl, composite,
                                   primary=("2023-01-01", "2026-08-31"),
                                   prior=("2021-01-01", "2022-12-31"))
    assert len(rep["derisk"]["grid"]) == 9    # 3 floor_z x 3 lo_mult
    assert len(rep["scaler"]["grid"]) == 6    # 3 k x 2 hi
    assert rep["derisk"]["verdict"] in ("PASS", "FAIL", "marginal", "knife-edge")
    assert rep["scaler"]["verdict"] in ("PASS", "FAIL", "marginal", "knife-edge")
    for key in ("buy_and_hold", "vol_target_only", "layer1_only"):
        assert key in rep["baselines"]


def test_derisk_leg_is_reported_as_not_exercised_when_composite_never_dips():
    """The live run's failure mode, pinned: the `all_usd` composite never drops
    below any `floor_z` in the grid, so `guidance_derisk_multiplier` is a flat
    1.0 everywhere and the de-risk leg tests NOTHING. The report must make that
    visible (`active_days_* == 0`) and its block must be numerically identical
    to the vol-target-only baseline -- otherwise the gate numbers read as a
    verdict on the signal when they are really a verdict on `apply_overlay_l2`
    with a constant multiplier."""
    idx, rng, ret, lvl = _derisk_inputs()
    # entirely >= 0, so it is above every floor_z in {-0.25, -0.5, -0.75}
    composite = pd.Series(np.abs(rng.normal(size=len(idx))) + 0.2, index=idx)

    rep = cgs.derisk_scaler_report(ret, lvl, composite,
                                   primary=("2023-01-01", "2026-08-31"),
                                   prior=("2021-01-01", "2022-12-31"))

    for row in rep["derisk"]["grid"]:
        assert row["active_days_primary"] == 0, row
        assert row["active_days_prior"] == 0, row

    derisk_primary = rep["derisk"]["central"]["block"]["primary"]["metrics"]
    vt_primary = rep["baselines"]["vol_target_only"]["primary"]["metrics"]
    for metric in ("sharpe", "cagr", "max_dd"):
        assert derisk_primary[metric] == pytest.approx(vt_primary[metric], abs=1e-6)

    # ... while the two-sided scaler IS exercised by the same composite (a
    # positive z still moves `clip(1+k*z, lo, hi)` away from 1.0).
    assert rep["scaler"]["central"]["active_days_primary"] > 0


def _flat_price_fn(tickers, start, end):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(5)
    data = {}
    for t in tickers:
        rets = rng.normal(0.0003, 0.01, size=len(idx))
        data[t] = 100.0 * (1.0 + pd.Series(rets, index=idx)).cumprod()
    return pd.DataFrame(data)


def _price_fn_without_controls(tickers, start, end):
    """`_flat_price_fn` with the two control tickers missing -- the shape a
    partially-failed yfinance fetch produces."""
    prices = _flat_price_fn(tickers, start, end)
    return prices.drop(columns=[c for c in ("^TNX", "SMH") if c in prices.columns])


def _fake_bigfour():
    """Mirrors `capex_signal.fetch_bigfour_capex`'s contract: `decel2` plus the
    point-in-time `known_date` (period end + 50d) the control is aligned on."""
    idx = pd.PeriodIndex(["2022Q1", "2022Q2"], freq="Q")
    return pd.DataFrame(
        {"decel2": [0.0, 0.0],
         "known_date": [p.to_timestamp(how="end").normalize() + pd.Timedelta(days=50)
                        for p in idx]},
        index=idx)


def _synthetic_panel():
    rows = []
    for i, u in enumerate(["D", "AEP", "NEE", "SO", "ETR", "XEL", "DUK", "PCG"]):
        for q in range(6):
            rows.append({
                "utility": u,
                "report_date": pd.Timestamp("2022-01-01") + pd.DateOffset(months=6 * q),
                "capex_plan_usd_m": 10000.0 + 500.0 * q + 100.0 * i,
                "revision_vs_prior_usd_m": np.nan if q == 0 else 500.0 + 20.0 * i,
                "dc_attributed_usd_m": 200.0 if (q >= 3 and i < 3) else np.nan,
                "dc_basis": "stated" if (q >= 3 and i < 3) else "none",
            })
    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    # `guidance_signal_report`'s default panel is `load_capex_guidance()` output,
    # which always carries `prior_capex_plan_usd_m` (the immediately-prior plan
    # level per utility). `_SERIES_SPECS`' three `*_pct` series read it as the
    # size-weighted denominator, so a faithful stand-in panel must supply it the
    # same way the loader does.
    df = df.sort_values(["utility", "report_date"]).reset_index(drop=True)
    df["prior_capex_plan_usd_m"] = df.groupby("utility")["capex_plan_usd_m"].shift(1)
    return df


def test_lead_lag_table_reports_pearson_corr_at_each_offset():
    idx = pd.date_range("2023-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(6)
    signal_monthly = pd.Series(rng.normal(size=24), index=idx)
    # "forward return" 2 months after date t is built to equal the signal's
    # own value from t -- i.e. the signal genuinely leads by 2 months. Given
    # this module's `shift(k)` convention (negative k = signal leads, matching
    # the VA-probe/Table-B9 doc convention "k=-1 (filings lead)"), the perfect
    # correlation must show up at k=-2, not k=+2.
    fwd_proxy = signal_monthly.shift(2)
    table = cgs.lead_lag_table(signal_monthly, fwd_proxy, ks=range(-3, 4))
    assert set(table.index) == set(range(-3, 4))
    assert table.loc[-2] == pytest.approx(1.0, abs=1e-6)


def test_guidance_signal_report_aligns_bigfour_control_on_its_known_date():
    """Point-in-time discipline (a plan Global Constraint): the hyperscaler
    control must enter the regression at `known_date` (period end + 50d), not
    at the quarter end -- regressing forward returns on a print that was not
    yet public is a 50-day look-ahead."""
    bf = _fake_bigfour()
    captured = {}

    def _capture(composites, basket_ret, controls, **kw):
        captured["index"] = controls.index
        return {}

    import grid_equipment_basket.capex_guidance_signal as mod
    real = mod.timing_report
    mod.timing_report = _capture
    try:
        mod.guidance_signal_report(
            price_fn=_flat_price_fn, panel_df=_synthetic_panel(), bigfour_fn=lambda: bf,
            primary=("2023-01-01", "2023-12-31"), prior=("2021-01-01", "2022-12-31"),
            holdout=("2023-10-01", "2023-12-31"))
    finally:
        mod.timing_report = real

    # 2022Q2 ends 2022-06-30; known_date is 2022-08-19, so the control's last
    # month-end is in August, not June.
    last_bigfour_month = captured["index"][captured["index"] <= pd.Timestamp("2022-12-31")].max()
    assert last_bigfour_month >= pd.Timestamp("2022-08-01")


def test_guidance_signal_report_survives_missing_bigfour_and_control_prices():
    """A cold `bigfour_capex.parquet` cache / failed SEC fetch (empty frame) and
    a partial price fetch (no `^TNX` / `SMH`) must degrade the affected control
    cells to n=0, not crash the whole report. Before the DatetimeIndex guards
    this raised `TypeError: '<' not supported between instances of 'Timestamp'
    and 'str'` from `timing_report`'s date-string slice."""
    for price_fn in (_flat_price_fn, _price_fn_without_controls):
        rep = cgs.guidance_signal_report(
            price_fn=price_fn, panel_df=_synthetic_panel(),
            bigfour_fn=lambda: pd.DataFrame(),
            primary=("2023-01-01", "2023-12-31"), prior=("2021-01-01", "2022-12-31"),
            holdout=("2023-10-01", "2023-12-31"))
        assert "timing_basket" in rep and "all_usd" in rep["timing_basket"]
        assert "derisk_scaler_gate" in rep
        assert rep["timing_basket"]["all_usd"][1]["control_with_hyperscaler"]["n"] == 0


def test_guidance_signal_report_runs_end_to_end():
    rep = cgs.guidance_signal_report(
        price_fn=_flat_price_fn, panel_df=_synthetic_panel(), bigfour_fn=_fake_bigfour,
        primary=("2023-01-01", "2023-12-31"), prior=("2021-01-01", "2022-12-31"),
        holdout=("2023-10-01", "2023-12-31"))
    assert rep["universe_used"] == "full"
    assert "timing_basket" in rep and "all_usd" in rep["timing_basket"]
    assert "derisk_scaler_gate" in rep
    assert rep["derisk_scaler_gate"]["derisk"]["verdict"] in ("PASS", "FAIL", "marginal", "knife-edge")
    assert "continuity_lead_lag" in rep and "all_usd" in rep["continuity_lead_lag"]


def test_guidance_signal_report_not_testable_short_circuits():
    df = _panel([{"utility": "ZZZ", "report_date": "1999-01-01",
                 "capex_plan_usd_m": 10.0, "revision_vs_prior_usd_m": np.nan}])
    rep = cgs.guidance_signal_report(price_fn=_flat_price_fn, panel_df=df,
                                     bigfour_fn=lambda: pd.DataFrame({"decel2": []}))
    assert rep["verdict"] == "not_yet_testable"
    assert "timing_basket" not in rep


def test_guidance_table_handles_not_yet_testable_report():
    rep = {"universe_used": "not_testable",
          "feasibility": {"n_usable": 0, "n_total": 3, "per_utility": {}},
          "verdict": "not_yet_testable"}
    out = cgs.guidance_table(rep)
    assert "not_yet_testable" in out or "NOT YET TESTABLE" in out.upper()


def test_guidance_table_renders_full_report():
    rep = cgs.guidance_signal_report(
        price_fn=_flat_price_fn, panel_df=_synthetic_panel(), bigfour_fn=_fake_bigfour,
        primary=("2023-01-01", "2023-12-31"), prior=("2021-01-01", "2022-12-31"),
        holdout=("2023-10-01", "2023-12-31"))
    out = cgs.guidance_table(rep)
    assert "full" in out
    assert "all_usd" in out
    assert "DERISK" in out.upper() or "SCALER" in out.upper()
    # the de-risk/scaler gate lines must carry the exercised-day count, so a
    # constant-1.0 (not-exercised) leg can never be read as a merit verdict
    assert out.count("active_days(primary)=") == 2
