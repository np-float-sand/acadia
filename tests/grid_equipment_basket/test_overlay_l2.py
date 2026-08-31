import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import overlay as ov


# ── helpers ──────────────────────────────────────────────────────────────────

def _calm_returns(n=300, mu=1e-5, start="2020-01-01"):
    return pd.Series(mu, index=pd.bdate_range(start, periods=n))


def _noisy_returns(n=300, sd=0.04, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, sd, n), index=pd.bdate_range(start, periods=n))


def _const_multiplier(like: pd.Series, value: float) -> pd.Series:
    return pd.Series(value, index=like.index, name="regime_mult")


# ── regime_exposure ──────────────────────────────────────────────────────────

def test_regime_exposure_is_regime_times_vol_scalar():
    r = _noisy_returns(seed=1)
    mult = _const_multiplier(r, 0.6)
    exp = ov.regime_exposure(r, mult, vol_lookback=20, target_vol=0.20, max_leverage=1.5)
    vscal = ov.vol_target_scalar(r, 20, 0.20, 1.5).reindex(r.index).fillna(1.0)
    expected = (0.6 * vscal).clip(upper=1.5)
    pd.testing.assert_series_equal(exp, expected.rename("exposure"), check_names=False)


def test_regime_exposure_clamps_combined_at_max_leverage():
    r = _calm_returns()                       # vol scalar pins at the 1.5 cap
    mult = _const_multiplier(r, 1.25)         # 1.25 * 1.5 = 1.875 -> must clamp to 1.5
    exp = ov.regime_exposure(r, mult, vol_lookback=20, target_vol=0.20, max_leverage=1.5)
    assert exp.max() == pytest.approx(1.5)
    assert (exp <= 1.5 + 1e-9).all()


def test_regime_exposure_nan_multiplier_days_treated_as_invested():
    r = _noisy_returns(seed=2)
    mult = _const_multiplier(r, 1.0)
    mult.iloc[50:60] = np.nan
    exp = ov.regime_exposure(r, mult, vol_lookback=20, target_vol=0.20, max_leverage=1.5)
    vscal = ov.vol_target_scalar(r, 20, 0.20, 1.5).reindex(r.index).fillna(1.0)
    # NaN multiplier -> 1.0, so exposure on those days is just the vol scalar
    assert exp.iloc[50:60].equals(vscal.iloc[50:60].clip(upper=1.5))


# ── apply_overlay_l2 ─────────────────────────────────────────────────────────

def test_flat_regime_multiplier_reproduces_vol_target_only():
    r = _noisy_returns(seed=3)
    px = (1.0 + r).cumprod()
    mult = _const_multiplier(r, 1.0)
    l2 = ov.apply_overlay_l2(r, mult, rf_annual=0.04,
                             vol_lookback=20, target_vol=0.20, max_leverage=1.5)
    # vol-target-only == apply_overlay with the trend gate disabled (huge MA)
    vt_only = ov.apply_overlay(r, px, rf_annual=0.04, ma_days=10**9,
                               vol_lookback=20, target_vol=0.20, max_leverage=1.5)
    pd.testing.assert_series_equal(l2, vt_only, check_names=False)


def test_apply_overlay_l2_return_identity():
    r = _noisy_returns(seed=4)
    mult = _const_multiplier(r, 0.6)
    mult.iloc[:120] = 1.25
    l2 = ov.apply_overlay_l2(r, mult, rf_annual=0.04,
                             vol_lookback=20, target_vol=0.20, max_leverage=1.5)
    exp = ov.exposure_series_l2(r, mult, vol_lookback=20, target_vol=0.20, max_leverage=1.5)
    rf_daily = 0.04 / 252
    expected = exp * r.fillna(0.0) + (1.0 - exp) * rf_daily
    pd.testing.assert_series_equal(l2, expected.reindex(r.index), check_names=False)


def test_apply_overlay_l2_aligns_to_input_index_and_has_no_nan():
    r = _noisy_returns(seed=5)
    mult = _const_multiplier(r, 0.6)
    out = ov.apply_overlay_l2(r, mult)
    assert out.index.equals(r.index)
    assert out.notna().all()


def test_apply_overlay_l2_cuts_exposure_when_regime_is_loose():
    r = _noisy_returns(seed=6, sd=0.02)
    loose = _const_multiplier(r, 0.6)
    neutral = _const_multiplier(r, 1.0)
    exp_loose = ov.exposure_series_l2(r, loose).mean()
    exp_neutral = ov.exposure_series_l2(r, neutral).mean()
    assert exp_loose < exp_neutral
