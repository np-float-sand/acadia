import numpy as np
import pandas as pd
import pytest

from electrification_strategy import config
from electrification_strategy._util import month_hold
from electrification_strategy.valuation_overlay import extension_multiplier


def test_month_hold_uses_prior_month_end():
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    daily = pd.Series(range(len(idx)), index=idx, dtype=float)
    held = month_hold(daily, idx, fill=1.0)
    jan_last = daily[idx.to_period("M") == pd.Period("2021-01")].iloc[-1]
    feb_days = held[idx.to_period("M") == pd.Period("2021-02")]
    assert (feb_days == jan_last).all()
    jan_days = held[idx.to_period("M") == pd.Period("2021-01")]
    assert (jan_days == 1.0).all()  # nothing before -> fill


def _flat_then_ramp(idx):
    lvl = pd.Series(100.0, index=idx)
    tail = idx[idx >= idx[-126]]
    lvl.loc[tail] = np.linspace(100.0, 180.0, len(tail))
    return lvl


def test_full_multiplier_when_calm():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    lvl = pd.Series(100.0 * (1.0 + 0.00005) ** np.arange(len(idx)), index=idx)
    spy = pd.Series(0.0003, index=idx)
    m = extension_multiplier(lvl, spy)
    assert m.iloc[-20:].eq(1.0).all()


def test_low_multiplier_when_stretched():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    lvl = _flat_then_ramp(idx)
    spy = pd.Series(0.0, index=idx)
    m = extension_multiplier(lvl, spy)
    assert m.iloc[-1] == config.VAL_MULT_LOW


def test_warmup_is_full():
    idx = pd.bdate_range("2019-01-01", "2019-09-30")   # < 200 sessions
    lvl = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    spy = pd.Series(0.0, index=idx)
    m = extension_multiplier(lvl, spy)
    assert m.eq(1.0).all()


def test_scale_widens_thresholds():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    lvl = _flat_then_ramp(idx)
    spy = pd.Series(0.0, index=idx)
    tight = extension_multiplier(lvl, spy, scale=1.0).iloc[-1]
    loose = extension_multiplier(lvl, spy, scale=1.5).iloc[-1]
    assert tight == config.VAL_MULT_LOW
    assert loose >= tight  # wider thresholds -> same or higher multiplier
