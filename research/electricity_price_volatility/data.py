"""
Return / realized-variance construction for hubs, equities, and controls.

Variance-proxy choice (stated per the pre-registration's "state your choice"
requirement): the squared daily log return, r_t^2, is used as the realized
daily variance proxy for BOTH hub prices and equity/BTC/SPY returns. This is
the standard Merton (1980) / French-Schwert-Stambaugh proxy for one-day
realized variance from a single daily observation, and it is what a
"GARCH-style variance regression" conventionally regresses (no intraday data
is available for the equity/BTC/SPY leg, so a rolling multi-day window would
only smooth the hub leg and not the response leg -- using the same one-day
proxy on both sides keeps the comparison symmetric). No rolling smoothing is
applied on top of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


def realized_variance(returns: pd.Series) -> pd.Series:
    """Squared daily log return -- the realized-variance proxy (see module docstring)."""
    return returns ** 2


def build_hub_frame(hub_price: pd.Series) -> pd.DataFrame:
    """From a daily mean hub price series, build return + variance columns."""
    ret = log_returns(hub_price)
    var = realized_variance(ret)
    return pd.DataFrame({"hub_price": hub_price, "hub_ret": ret, "hub_var": var})


def build_asset_frame(price: pd.Series, prefix: str) -> pd.DataFrame:
    ret = log_returns(price)
    var = realized_variance(ret)
    return pd.DataFrame({f"{prefix}_price": price, f"{prefix}_ret": ret, f"{prefix}_var": var})


def align_lagged_hub_to_equity(
    hub_frame: pd.DataFrame, equity_index: pd.DatetimeIndex, lag_days: int = 1
) -> pd.DataFrame:
    """
    Align the hub's daily (calendar-day) series onto the equity trading calendar,
    lagged by `lag_days` TRADING days of the equity calendar.

    Hub markets settle every calendar day (including weekends); equities trade
    only on business days. For each equity trading date t, this returns the hub
    observation from the most recent hub calendar date at or before the equity
    trading date `lag_days` trading days earlier (merge_asof, backward) -- i.e.
    "lagged hub-price-variance" uses whatever hub information was already public
    strictly before the equity-return day being predicted.
    """
    hub_sorted = hub_frame.sort_index()
    hub_reset = hub_sorted.reset_index().rename(columns={hub_sorted.index.name or "index": "time"})
    hub_reset["time"] = pd.to_datetime(hub_reset["time"]).astype("datetime64[ns]")

    # equity_index shifted back by lag_days trading positions (using the equity
    # calendar itself, so this is a genuine trading-day lag, not a calendar-day one)
    eq_index_ns = pd.DatetimeIndex(equity_index).astype("datetime64[ns]")
    shifted_targets = pd.Series(eq_index_ns, index=equity_index)
    lagged_dates = shifted_targets.shift(lag_days)

    valid = lagged_dates.dropna()
    left = pd.DataFrame(
        {"asof": pd.DatetimeIndex(valid.values).astype("datetime64[ns]")}, index=valid.index
    ).sort_values("asof")
    merged = pd.merge_asof(
        left, hub_reset,
        left_on="asof", right_on="time", direction="backward",
    )
    merged.index = left.index
    merged = merged.reindex(equity_index)
    return merged.drop(columns=["asof", "time"], errors="ignore")
