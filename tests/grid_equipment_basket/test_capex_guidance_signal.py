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
    # a day between two events holds the earlier one's (already z-scored) value
    d_after_first = events.index[0] + pd.Timedelta(days=5)
    d_of_first = events.index[0]
    assert result.loc[d_after_first] == pytest.approx(result.loc[d_of_first])
