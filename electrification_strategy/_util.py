"""Small shared helpers."""
from __future__ import annotations

import pandas as pd


def month_hold(daily: pd.Series, index: pd.DatetimeIndex, fill) -> pd.Series:
    """Each calendar month's last value, applied to the *following* month's days."""
    periods = index.to_period("M")
    by_month = pd.Series(daily.to_numpy(), index=index).groupby(periods).last()
    held = by_month.shift(1).reindex(periods).to_numpy()
    return pd.Series(held, index=index).astype(float).fillna(fill)
