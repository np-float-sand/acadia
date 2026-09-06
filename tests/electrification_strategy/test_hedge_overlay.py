import numpy as np
import pandas as pd
import pytest

from electrification_strategy import config, hedge_overlay


@pytest.fixture
def px():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    base = pd.Series(100.0 * (1.005) ** np.arange(len(idx)), index=idx)
    return pd.DataFrame({"GLD": base, "IEF": base * 0.5, "DLR": base * 2, "EQIX": base * 3})


def test_sleeve_returns_equal_weight(px):
    r = hedge_overlay.sleeve_returns(px)
    expect = px[["GLD", "IEF"]].pct_change().mean(axis=1).fillna(0.0)
    pd.testing.assert_series_equal(r, expect, check_names=False)


def test_short_returns_equal_weight(px):
    r = hedge_overlay.short_returns(px)
    assert r.shape[0] == px.shape[0]
    assert r.iloc[0] == 0.0  # first row NaN -> filled to 0


def test_mask_true_only_when_yield_rose(px):
    idx = pd.bdate_range("2019-01-01", "2021-06-30")
    rising = pd.Series(np.linspace(0.5, 2.5, len(idx)), index=idx, name="DFII10")
    m = hedge_overlay.real_yield_rising_mask(rising, px.index)
    assert m.reindex(px.index).fillna(False).iloc[-1]

    falling = pd.Series(np.linspace(2.5, 0.5, len(idx)), index=idx, name="DFII10")
    m2 = hedge_overlay.real_yield_rising_mask(falling, px.index)
    assert not m2.reindex(px.index).fillna(False).iloc[-1]


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
