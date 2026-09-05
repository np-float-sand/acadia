import numpy as np
import pandas as pd
import pytest

from grid_equipment_basket import capex_guidance_signal as cgs


def test_guidance_composite_empty_events_is_empty_series():
    result = cgs.guidance_composite(pd.Series(dtype=float), "2023-01-01", "2023-06-01")
    assert result.empty


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


def test_guidance_composite_holds_value_between_events():
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
