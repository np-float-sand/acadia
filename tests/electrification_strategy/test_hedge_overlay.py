import numpy as np
import pandas as pd
import pytest

from electrification_strategy import config, hedge_overlay


@pytest.fixture
def px():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    base = pd.Series(100.0 * (1.005) ** np.arange(len(idx)), index=idx)
    return pd.DataFrame({"GLD": base, "IEF": base * 0.5, "DLR": base * 2, "EQIX": base * 3,
                         "ENPH": base * 4, "SEDG": base * 0.25, "CHPT": base * 0.1,
                         "RUN": base * 0.3, "BLNK": base * 0.05, "STEM": base * 0.2})


def test_sleeve_returns_equal_weight(px):
    r = hedge_overlay.sleeve_returns(px)
    expect = px[["GLD", "IEF"]].pct_change().mean(axis=1).fillna(0.0)
    pd.testing.assert_series_equal(r, expect, check_names=False)


def test_short_returns_equal_weight(px):
    r = hedge_overlay.short_returns(px)
    assert r.shape[0] == px.shape[0]
    assert r.iloc[0] == 0.0  # first row NaN -> filled to 0


def test_solar_rotation_returns_equal_weight(px):
    r = hedge_overlay.solar_rotation_returns(px)
    expect = px[list(config.SOLAR_ROTATION_TICKERS)].pct_change().mean(axis=1).fillna(0.0)
    pd.testing.assert_series_equal(r, expect, check_names=False)


def test_mask_true_only_when_yield_rose(px):
    idx = pd.bdate_range("2019-01-01", "2021-06-30")
    rising = pd.Series(np.linspace(0.5, 2.5, len(idx)), index=idx, name="DFII10")
    m = hedge_overlay.real_yield_rising_mask(rising, px.index)
    assert m.reindex(px.index).fillna(False).iloc[-1]

    falling = pd.Series(np.linspace(2.5, 0.5, len(idx)), index=idx, name="DFII10")
    m2 = hedge_overlay.real_yield_rising_mask(falling, px.index)
    assert not m2.reindex(px.index).fillna(False).iloc[-1]


def test_falling_mask_true_only_when_yield_fell(px):
    idx = pd.bdate_range("2019-01-01", "2021-06-30")
    falling = pd.Series(np.linspace(2.5, 0.5, len(idx)), index=idx, name="DFII10")
    m = hedge_overlay.real_yield_falling_mask(falling, px.index)
    assert m.reindex(px.index).fillna(False).iloc[-1]

    rising = pd.Series(np.linspace(0.5, 2.5, len(idx)), index=idx, name="DFII10")
    m2 = hedge_overlay.real_yield_falling_mask(rising, px.index)
    assert not m2.reindex(px.index).fillna(False).iloc[-1]

    # rising and falling masks are mutually exclusive at any given point
    both = (hedge_overlay.real_yield_rising_mask(falling, px.index)
            & hedge_overlay.real_yield_falling_mask(falling, px.index))
    assert not both.any()


def test_apply_hedge_math(px):
    core = pd.Series(0.001, index=px.index)
    sleeve = pd.Series(0.002, index=px.index)
    short = pd.Series(0.003, index=px.index)
    mask = pd.Series(True, index=px.index)

    both = hedge_overlay.apply_hedge(core, sleeve, short, mask, use_sleeve=True, use_short=True)
    expect = (1 - 0.15) * 0.001 + 0.15 * 0.002 - 0.25 * 0.003
    assert both.iloc[0] == pytest.approx(expect)

    neither = hedge_overlay.apply_hedge(core, sleeve, short, mask, use_sleeve=False, use_short=False)
    assert neither.iloc[0] == pytest.approx(0.001)

    sleeve_only = hedge_overlay.apply_hedge(core, sleeve, short, mask, use_sleeve=True, use_short=False)
    assert sleeve_only.iloc[0] == pytest.approx((1 - 0.15) * 0.001 + 0.15 * 0.002)

    mask_off = hedge_overlay.apply_hedge(core, sleeve, short, pd.Series(False, index=px.index),
                                        use_sleeve=False, use_short=True)
    assert mask_off.iloc[0] == pytest.approx(0.001)  # short contributes nothing when mask is False


def test_apply_hedge_solar_rotation(px):
    core = pd.Series(0.001, index=px.index)
    sleeve = pd.Series(0.002, index=px.index)
    short = pd.Series(0.003, index=px.index)
    solar = pd.Series(0.02, index=px.index)
    no_mask = pd.Series(False, index=px.index)
    solar_on = pd.Series(True, index=px.index)

    # +val+sleeve+solar: layered on top of the sleeve-adjusted book, matching the
    # 2026-09-06 probe's "(1 - w*falling)*[screen +val+sleeve] + w*falling*der_r".
    both = hedge_overlay.apply_hedge(core, sleeve, short, no_mask, use_sleeve=True, use_short=False,
                                     solar_r=solar, solar_mask=solar_on, use_solar=True)
    sleeved = (1 - 0.15) * 0.001 + 0.15 * 0.002
    expect = (1 - 0.10) * sleeved + 0.10 * 0.02
    assert both.iloc[0] == pytest.approx(expect)

    # solar mask off -> identical to +val+sleeve alone
    off = hedge_overlay.apply_hedge(core, sleeve, short, no_mask, use_sleeve=True, use_short=False,
                                    solar_r=solar, solar_mask=no_mask, use_solar=True)
    assert off.iloc[0] == pytest.approx(sleeved)

    # use_solar=False ignores solar_r/solar_mask entirely (backward compatible default)
    default = hedge_overlay.apply_hedge(core, sleeve, short, no_mask, use_sleeve=True, use_short=False)
    assert default.iloc[0] == pytest.approx(sleeved)
