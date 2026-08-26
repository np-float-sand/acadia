"""Tests for build_gsi()'s configurable weights — isolating the energy-price
component (lmp_zscore) from congestion/reserve/event, per the price-vs-congestion
resilience thesis split."""
import numpy as np
import pandas as pd

from grid_resilience.signals.grid_stress_index import build_gsi
from grid_resilience.config import GSI_WEIGHTS, GSI_WEIGHTS_PRICE_ONLY


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["start", "end", "iso", "type", "name", "severity"])


def _flat_lmp_with_congestion_spike(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """LMP stays flat (no price stress, aside from small natural noise so the
    rolling std isn't exactly zero) but congestion_frac spikes midway — a
    scenario where blended GSI should react but a price-only GSI should not."""
    rng = np.random.default_rng(0)
    lmp_max = pd.Series(50.0 + rng.normal(0, 0.5, len(dates)), index=dates)
    congestion_frac = pd.Series(0.1 + rng.normal(0, 0.01, len(dates)), index=dates)
    spike_start = dates[len(dates) // 2]
    congestion_frac.loc[spike_start:] = 0.9 + rng.normal(0, 0.01, (congestion_frac.index >= spike_start).sum())
    return pd.DataFrame({"lmp_max": lmp_max, "congestion_frac": congestion_frac})


def test_default_weights_match_todays_behavior():
    """Omitting `weights` must reproduce today's GSI_WEIGHTS-driven output exactly —
    this flag has to be a true no-op when unused."""
    dates = pd.date_range("2024-01-01", "2024-04-30")
    daily_lmp = _flat_lmp_with_congestion_spike(dates)
    explicit = build_gsi(daily_lmp, pd.DataFrame(), _empty_events(), iso="PJM", weights=GSI_WEIGHTS)
    default = build_gsi(daily_lmp, pd.DataFrame(), _empty_events(), iso="PJM")
    pd.testing.assert_series_equal(default["gsi"], explicit["gsi"])


def test_price_only_weights_ignore_a_congestion_spike():
    """With GSI_WEIGHTS_PRICE_ONLY, a congestion spike (no price movement) should
    barely move the composite gsi, unlike the clear jump blended weights show for
    the same fixture (test_blended_weights_do_react_to_the_same_congestion_spike)."""
    dates = pd.date_range("2024-01-01", "2024-04-30")
    daily_lmp = _flat_lmp_with_congestion_spike(dates)
    result = build_gsi(
        daily_lmp, pd.DataFrame(), _empty_events(), iso="PJM", weights=GSI_WEIGHTS_PRICE_ONLY
    )
    before = result["gsi"].iloc[: len(dates) // 2 - 5].mean()
    after = result["gsi"].iloc[len(dates) // 2 + 5 :].mean()
    assert abs(after - before) < 0.1


def test_blended_weights_do_react_to_the_same_congestion_spike():
    """Sanity check the test fixture: today's blended weights *should* react to
    the congestion spike used above, confirming price-only vs blended genuinely differ."""
    dates = pd.date_range("2024-01-01", "2024-04-30")
    daily_lmp = _flat_lmp_with_congestion_spike(dates)
    result = build_gsi(daily_lmp, pd.DataFrame(), _empty_events(), iso="PJM", weights=GSI_WEIGHTS)
    before = result["gsi"].iloc[: len(dates) // 2 - 5].mean()
    after = result["gsi"].iloc[len(dates) // 2 + 5 :].mean()
    assert after > before


def test_price_only_weights_still_react_to_a_real_price_spike():
    """Price-only GSI must still detect genuine LMP spikes — it isolates the
    price component, it doesn't neuter it."""
    dates = pd.date_range("2024-01-01", "2024-04-30")
    lmp_max = pd.Series(50.0, index=dates)
    spike_start = dates[len(dates) // 2]
    lmp_max.loc[spike_start : spike_start + pd.Timedelta(days=5)] = 400.0
    daily_lmp = pd.DataFrame({"lmp_max": lmp_max})
    result = build_gsi(
        daily_lmp, pd.DataFrame(), _empty_events(), iso="ERCOT", weights=GSI_WEIGHTS_PRICE_ONLY
    )
    assert result.loc[spike_start, "gsi"] > 0.6
