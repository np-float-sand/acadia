"""FTR (Financial Transmission Rights) bid-implied forward-congestion signal.

PJM's actual FTR *auction-clearing* prices require a PJM membership login
(confirmed 2026-09-01 -- the historical-results page redirects to PJM SSO);
the *bid* data (`ftr_bids_mnt` on DataMiner) is public with no login. This
module builds a proxy from those bids: the MW-weighted mean price of buy-side
Obligation bids sinking at each DC-heavy zone's own zone-aggregate pnode
(config.REGIME_ZONES_CORE), which is the standard instrument a load-serving
entity uses to hedge congestion into that zone.

This is a *bid* price, not a clearing price -- it reflects what participants
were willing to pay, not what the auction actually awarded. Treat it as a
sentiment/demand proxy for forward congestion expectations, not a
reconstruction of the true market-clearing price.

The composite this module produces has the same contract as
`grid_regime.regime_composite` (a daily trailing-z-scored series) and feeds
`grid_regime.regime_multiplier` unchanged.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from grid_equipment_basket import config, overlay
from grid_equipment_basket.backtest import calendar_year_returns, compute_metrics
from grid_equipment_basket.grid_regime import (
    _trailing_zscore, final_verdict, gate_check, regime_multiplier,
)

_ANN = config.ANN_FACTOR

# Bare zone-aggregate sink pnodes confirmed present in ftr_bids_mnt (2026-09-01
# probe, Jan 2019: 240-1097 bids and 700-5000 MW per zone at these exact
# names) -- the load-hedging instrument, not individual bus-level bids.
FTR_ZONE_SINKS: dict[str, list[str]] = {
    "DOM": ["DOM", "DOMINION HUB"],
    "AEP": ["AEP"],
    "COMED": ["COMED"],
    "PPL": ["PPL"],
}

# Feed description: "This data is posted on a four month delay."
FTR_LAG_MONTHS: int = 4

_MONTH_ABBR = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
_MONTH_NAME = {v: k for k, v in _MONTH_ABBR.items()}

_MARKET_NAME_RE = re.compile(r"^([A-Z]{3})\s+(\d{4})\s+Auction$")


def parse_auction_month(market_name: str) -> pd.Timestamp:
    """``"JAN 2019 Auction"`` -> ``Timestamp("2019-01-01")``."""
    m = _MARKET_NAME_RE.match(str(market_name).strip())
    if not m or m.group(1) not in _MONTH_ABBR:
        raise ValueError(f"unrecognised FTR market_name: {market_name!r}")
    return pd.Timestamp(year=int(m.group(2)), month=_MONTH_ABBR[m.group(1)], day=1)


def market_name_for(month: pd.Timestamp) -> str:
    return f"{_MONTH_NAME[month.month]} {month.year} Auction"


def mw_weighted_price(bids: pd.DataFrame) -> float:
    """``sum(quoted_price * quoted_mw) / sum(quoted_mw)``; NaN if no rows or
    zero total MW."""
    if bids.empty:
        return float("nan")
    mw = bids["quoted_mw"].astype(float)
    total_mw = float(mw.sum())
    if total_mw <= 0:
        return float("nan")
    return float((bids["quoted_price"].astype(float) * mw).sum() / total_mw)


def zone_monthly_composite(bids: pd.DataFrame,
                           zone_sinks: dict[str, list[str]] | None = None) -> float:
    """One auction month's bids -> equal-weight mean of each zone's
    MW-weighted ``quoted_price``, over the zones in ``zone_sinks`` that have
    at least one matching row. All zones absent -> NaN."""
    zone_sinks = zone_sinks or FTR_ZONE_SINKS
    if bids.empty or "sink_pnode_name" not in bids.columns:
        return float("nan")
    per_zone = []
    for sinks in zone_sinks.values():
        val = mw_weighted_price(bids[bids["sink_pnode_name"].isin(sinks)])
        if val == val:  # not NaN
            per_zone.append(val)
    if not per_zone:
        return float("nan")
    return float(np.mean(per_zone))


def monthly_composite_series(bids_by_month: dict[str, pd.DataFrame],
                             zone_sinks: dict[str, list[str]] | None = None) -> pd.Series:
    """``{market_name: bids}`` -> a series indexed by auction month (sorted)."""
    data = {parse_auction_month(mn): zone_monthly_composite(df, zone_sinks)
            for mn, df in bids_by_month.items()}
    return pd.Series(data, dtype=float).sort_index().rename("ftr_composite_raw")


def available_series(monthly: pd.Series, lag_months: int = FTR_LAG_MONTHS) -> pd.Series:
    """Shift the index forward by ``lag_months`` -- an auction month's value
    becomes knowable ``lag_months`` after the month it describes."""
    shifted = monthly.copy()
    shifted.index = shifted.index + pd.DateOffset(months=lag_months)
    return shifted.sort_index()


def broadcast_daily(available: pd.Series, start: str, end: str) -> pd.Series:
    """Forward-fill the (already publication-lagged) monthly series onto a
    daily calendar index over ``[start, end]``; days before the first
    availability date are NaN."""
    idx = pd.date_range(start, end, freq="D")
    combined = available.reindex(available.index.union(idx)).sort_index().ffill()
    return combined.reindex(idx)


def _market_names_for_range(start: str, end: str) -> list[str]:
    """Every auction-month label from ``start``'s month through ``end``'s
    month, inclusive."""
    months = pd.period_range(pd.Timestamp(start).to_period("M"),
                             pd.Timestamp(end).to_period("M"), freq="M")
    return [market_name_for(m.to_timestamp()) for m in months]


def _default_bids_fn(market_names: list[str]) -> dict[str, pd.DataFrame]:
    from grid_resilience.data.grid_data import fetch_ftr_bids_by_month
    return {mn: fetch_ftr_bids_by_month(mn) for mn in market_names}


def ftr_composite(start: str, end: str, *,
                  lag_months: int = FTR_LAG_MONTHS,
                  zone_sinks: dict[str, list[str]] | None = None,
                  zscore_window: int = config.REGIME_ZSCORE_WINDOW,
                  zscore_minp: int = config.REGIME_ZSCORE_MINP,
                  winsor: float = config.REGIME_ZSCORE_WINSOR,
                  bids_fn=None) -> pd.Series:
    """Daily FTR-bid-implied forward-congestion composite (trailing z-score),
    matching the contract of ``grid_regime.regime_composite`` so it feeds
    ``grid_regime.regime_multiplier`` unchanged.

    ``start``/``end`` bound the auction months fetched (the caller passes a
    window that starts well before its evaluation window, same convention as
    ``regime_composite``'s ``signal_start`` warm-up). ``bids_fn(market_names)
    -> {market_name: bids_df}`` is injectable for tests; the default fetches
    from PJM DataMiner via ``grid_resilience.data.grid_data.fetch_ftr_bids_by_month``.
    """
    zone_sinks = zone_sinks or FTR_ZONE_SINKS
    bids_fn = bids_fn or _default_bids_fn

    market_names = _market_names_for_range(start, end)
    bids_by_month = bids_fn(market_names)
    monthly = monthly_composite_series(bids_by_month, zone_sinks)
    if monthly.dropna().empty:
        return pd.Series(dtype=float, name="ftr_composite")

    avail = available_series(monthly, lag_months)
    daily = broadcast_daily(avail, start, end)
    return _trailing_zscore(daily, zscore_window, zscore_minp, winsor).rename("ftr_composite")


# ── pre-registered gate report (same gate machinery as grid_regime) ────────

def ftr_signal_report(price_fn=None, bids_fn=None,
                      signal_start: str = config.REGIME_SIGNAL_START,
                      primary: tuple[str, str] = config.REGIME_PRIMARY_WINDOW,
                      prior: tuple[str, str] = config.REGIME_PRIOR_WINDOW,
                      lag_months: int = FTR_LAG_MONTHS,
                      zone_sinks: dict[str, list[str]] | None = None,
                      zscore_window: int = config.REGIME_ZSCORE_WINDOW,
                      zscore_minp: int = config.REGIME_ZSCORE_MINP,
                      winsor: float = config.REGIME_ZSCORE_WINSOR,
                      month_lookback: int = config.REGIME_MONTH_LOOKBACK) -> dict:
    """FTR-bid composite as a layer-2 exposure overlay (replaces the trend
    gate, same as `grid_regime`'s rungs): baselines (buy&hold, vol-target-only,
    layer-1-only) plus the FTR rung, evaluated with the same pre-registered
    gate (`grid_regime.gate_check`/`final_verdict`) on both the primary and
    prior windows.
    """
    price_fn = price_fn or overlay._default_price_fn
    rf = config.RISK_FREE_RATE
    full_end = primary[1]

    ret, lvl = overlay._basket_series(prior[0], full_end, price_fn)
    windows = {"primary": tuple(primary), "prior": tuple(prior)}

    def _block(series: pd.Series) -> dict:
        return {wk: {"metrics": compute_metrics(series.loc[ws:we].dropna(), rf, _ANN),
                    "calendar": calendar_year_returns(series.loc[ws:we].dropna())}
               for wk, (ws, we) in windows.items()}

    bh = _block(ret)
    vt_only = _block(overlay.apply_overlay(ret, lvl, rf, ma_days=10 ** 9))
    l1 = _block(overlay.apply_overlay(ret, lvl, rf))

    comp = ftr_composite(signal_start, full_end, lag_months=lag_months,
                         zone_sinks=zone_sinks, zscore_window=zscore_window,
                         zscore_minp=zscore_minp, winsor=winsor, bids_fn=bids_fn)
    mult = regime_multiplier(comp, mode="discrete", month_lookback=month_lookback)
    rung_block = _block(overlay.apply_overlay_l2(ret, mult, rf))

    gate = gate_check(rung_block["primary"], rung_block["prior"],
                      l1["primary"], l1["prior"], bh["primary"], bh["prior"])
    verdict = final_verdict(gate, [True])  # no parameter-neighbour plateau probe yet

    return {
        "windows": windows,
        "baselines": {"buy_and_hold": bh, "vol_target_only": vt_only, "layer1_only": l1},
        "ftr_rung": {"block": rung_block, "gate": gate},
        "verdict": verdict,
    }
