"""Utility capex-guidance revision signal ("Deliverable D").

Spec: docs/superpowers/specs/2026-09-04-capex-guidance-signal-design.md

Aggregate US electric utilities revise their forward 5-yr capital-program
guidance at each earnings call / 10-K / analyst day; since ~2023 those
revisions are increasingly data-center-driven and increasingly stated as
such. This module turns the hand/web-assembled panel in
`grid_resilience.data.utility_capex_guidance` into (1) a daily z-scored
composite, (2) a one-directional de-risk multiplier and a two-sided scaler
for the grid-equipment basket's exposure, and (3) the pre-registered gate
report testing both as a timing signal and as an exposure scaler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from grid_equipment_basket import config
from grid_equipment_basket.ftr_signal import broadcast_daily
from grid_equipment_basket.grid_regime import _trailing_zscore


def guidance_composite(events: pd.Series, start: str, end: str, *,
                       zscore_window: int = config.DC_GUIDANCE_ZSCORE_WINDOW,
                       zscore_minp: int = config.DC_GUIDANCE_ZSCORE_MINP,
                       winsor: float = config.DC_GUIDANCE_ZSCORE_WINSOR) -> pd.Series:
    """`events` indexed by `report_date` (already point-in-time -- see
    `utility_capex_guidance.aggregate_revision_series`). Forward-filled onto a
    daily calendar index over `[start, end]`, then trailing-z-scored
    (`grid_regime._trailing_zscore`'s rolling window only uses
    past-and-current observations, so this composite carries the same
    no-future-leak contract as `ftr_signal.ftr_composite` /
    `grid_regime.regime_composite`)."""
    if events.empty:
        return pd.Series(dtype=float, name="guidance_composite")
    daily = broadcast_daily(events, start, end)
    return _trailing_zscore(daily, zscore_window, zscore_minp, winsor).rename("guidance_composite")


def guidance_derisk_multiplier(composite: pd.Series, *,
                               floor_z: float = config.DC_GUIDANCE_DERISK_FLOOR_Z,
                               lo_mult: float = config.DC_GUIDANCE_DERISK_LO_MULT) -> pd.Series:
    """`{lo_mult, 1.0}` -- `lo_mult` on any day the composite z-score is below
    `floor_z` (revision-flow decelerating/reversing), back to 1.0 once it
    clears. NaN days (not-yet-warm) default to 1.0. Never exceeds 1.0. Same
    one-directional shape as `capex_signal.capex_derisk_multiplier` (spec s6.1)."""
    lo = composite < floor_z
    return pd.Series(np.where(lo.fillna(False), lo_mult, 1.0),
                     index=composite.index).rename("guidance_derisk_mult")


def guidance_scaler(composite: pd.Series, *,
                    k: float = config.DC_GUIDANCE_SCALER_K,
                    lo: float = config.DC_GUIDANCE_SCALER_LO,
                    hi: float = config.DC_GUIDANCE_SCALER_HI) -> pd.Series:
    """`clip(1 + k*z, lo, hi)` -- two-sided: leans in when the composite is
    positive (revision-flow accelerating), leans out when negative. NaN days
    default to 1.0. Same shape as `grid_regime`'s continuous-mode multiplier
    (spec s6.2)."""
    z = composite.fillna(0.0)
    raw = (1.0 + k * z).clip(lower=lo, upper=hi)
    return raw.where(composite.notna(), 1.0).rename("guidance_scaler")
