"""Layer-1 risk overlay for the equal-weight grid-equipment basket.

Two long-only, de-levering rules (plus an optional lever-up), decided at each
month-end and held for the following calendar month:

  1. Trend gate      -- hold the basket only while its own price index is at or
                        above its `ma_days` moving average; otherwise the slice
                        sits in cash at the risk-free rate.
  2. Volatility target -- scale exposure to `target_vol / trailing_realized_vol`,
                        capped at `max_leverage`; slack (or borrow) at rf.

**Honesty (carried from `docs/handoff_2026-08-29-grid-equipment-basket.md`):**
in-sample this overlay *underperforms* buy-and-hold in every calendar year
except 2025. Its entire back-tested edge is turning the one -42% DeepSeek
drawdown (Nov-2024 -> Apr-2025) into ~-9%. In a window without a -40% event it
reads as pure drag. The robust core is only "hold less when the trend is down
and volatility is high" -- every parameter beyond that is a tuned choice, which
is why `overlay_report` prints a parameter-plateau grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_equipment_basket import config
from grid_equipment_basket.backtest import calendar_year_returns, compute_metrics
from grid_equipment_basket.basket import simulate_basket

_ANN = config.ANN_FACTOR


def _month_hold(daily_value: pd.Series, index: pd.DatetimeIndex, fill) -> pd.Series:
    """Take each calendar month's last value, apply it to *next* month's days."""
    periods = index.to_period("M")
    by_month = pd.Series(daily_value.to_numpy(), index=index).groupby(periods).last()
    prev = by_month.shift(1)
    held = prev.reindex(periods).to_numpy()
    return pd.Series(held, index=index).astype(float).fillna(fill)


def trend_gate(basket_prices: pd.Series, ma_days: int = config.OVERLAY_MA_DAYS) -> pd.Series:
    """1.0 while the month-end close is >= its `ma_days` MA, else 0.0. The
    verdict is set at the prior month-end and held through the month; days
    before the MA is warm default to 1.0 (invested)."""
    close = basket_prices.dropna().astype(float).sort_index()
    ma = close.rolling(ma_days).mean()
    above = (close >= ma)
    above = above.where(ma.notna(), True)          # not-yet-warm -> invested
    return _month_hold(above.astype(float), close.index, fill=1.0)


def vol_target_scalar(basket_returns: pd.Series, lookback: int = config.OVERLAY_VOL_LOOKBACK,
                      target_vol: float = config.OVERLAY_TARGET_VOL,
                      max_leverage: float = config.OVERLAY_MAX_LEVERAGE) -> pd.Series:
    """`min(target_vol / annualised_trailing_vol, max_leverage)`, decided at the
    prior month-end and held through the month; warm-up days default to 1.0."""
    r = basket_returns.dropna().astype(float).sort_index()
    rv = r.rolling(lookback).std() * np.sqrt(_ANN)
    raw = (target_vol / rv).clip(lower=0.0, upper=max_leverage)
    return _month_hold(raw, r.index, fill=1.0)


def apply_overlay(basket_returns: pd.Series, basket_prices: pd.Series,
                  rf_annual: float = config.RISK_FREE_RATE,
                  ma_days: int = config.OVERLAY_MA_DAYS,
                  vol_lookback: int = config.OVERLAY_VOL_LOOKBACK,
                  target_vol: float = config.OVERLAY_TARGET_VOL,
                  max_leverage: float = config.OVERLAY_MAX_LEVERAGE) -> pd.Series:
    """Daily overlaid return: `exposure * basket_ret + (1 - exposure) * rf_daily`,
    with `exposure = trend_gate * vol_target_scalar`. Aligned to
    `basket_returns.index`."""
    gate = trend_gate(basket_prices, ma_days).reindex(basket_returns.index).fillna(1.0)
    vscal = vol_target_scalar(basket_returns, vol_lookback, target_vol, max_leverage)
    vscal = vscal.reindex(basket_returns.index).fillna(1.0)
    exposure = gate * vscal
    rf_daily = rf_annual / _ANN
    return (exposure * basket_returns.fillna(0.0) + (1.0 - exposure) * rf_daily).reindex(
        basket_returns.index)


def exposure_series(basket_returns: pd.Series, basket_prices: pd.Series,
                    ma_days: int = config.OVERLAY_MA_DAYS,
                    vol_lookback: int = config.OVERLAY_VOL_LOOKBACK,
                    target_vol: float = config.OVERLAY_TARGET_VOL,
                    max_leverage: float = config.OVERLAY_MAX_LEVERAGE) -> pd.Series:
    gate = trend_gate(basket_prices, ma_days).reindex(basket_returns.index).fillna(1.0)
    vscal = vol_target_scalar(basket_returns, vol_lookback, target_vol, max_leverage)
    return (gate * vscal.reindex(basket_returns.index).fillna(1.0)).rename("exposure")


# ── report ───────────────────────────────────────────────────────────────────

_PLATEAU_MA = (50, 75, 100, 125, 150)
_PLATEAU_TV = (0.15, 0.20, 0.25)


def _default_price_fn(tickers, start, end):
    from grid_equipment_basket.data.prices import fetch_prices
    return fetch_prices(tickers, start, end)


def _basket_series(start: str, end: str, price_fn) -> tuple[pd.Series, pd.Series]:
    prices = price_fn(sorted(config.UNIVERSE), start, end)
    cols = [t for t in config.UNIVERSE if t in prices.columns]
    br = simulate_basket(prices[cols], start, end, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, None)
    ret = br.returns
    lvl = (1.0 + ret).cumprod()
    return ret, lvl


def overlay_report(start: str, end: str, price_fn=None, prior_regime: bool = False) -> dict:
    """basket-only vs +trend-gate vs +trend-gate+vol-target over [start, end],
    plus a parameter-plateau grid and overlay mechanics."""
    price_fn = price_fn or _default_price_fn
    if prior_regime:
        start, end = config.PRIOR_REGIME_START, config.PRIOR_REGIME_END

    ret, lvl = _basket_series(start, end, price_fn)
    rf = config.RISK_FREE_RATE

    gate_only = apply_overlay(ret, lvl, rf, max_leverage=1.0, target_vol=1e9)  # gate on, vol-target off
    full = apply_overlay(ret, lvl, rf)

    out: dict = {
        "start": start, "end": end,
        "basket_only": {"metrics": compute_metrics(ret, rf, _ANN),
                        "calendar": calendar_year_returns(ret)},
        "trend_gate": {"metrics": compute_metrics(gate_only, rf, _ANN),
                       "calendar": calendar_year_returns(gate_only)},
        "trend_gate_voltarget": {"metrics": compute_metrics(full, rf, _ANN),
                                 "calendar": calendar_year_returns(full)},
    }

    exp = exposure_series(ret, lvl)
    flips = int((trend_gate(lvl, config.OVERLAY_MA_DAYS).reindex(ret.index).fillna(1.0)
                 .diff().abs() > 0).sum())
    out["mechanics"] = {
        "avg_exposure": round(float(exp.mean()), 3),
        "min_exposure": round(float(exp.min()), 3),
        "max_exposure": round(float(exp.max()), 3),
        "pct_days_gated_out": round(float((exp == 0).mean()), 3),
        "gate_flips": flips,
    }

    grid = []
    for ma in _PLATEAU_MA:
        for tv in _PLATEAU_TV:
            o = apply_overlay(ret, lvl, rf, ma_days=ma, target_vol=tv)
            m = compute_metrics(o, rf, _ANN)
            grid.append({"ma_days": ma, "target_vol": tv,
                         "sharpe": m["sharpe"], "max_dd": m["max_dd"], "cagr": m["cagr"]})
    out["plateau_grid"] = grid
    return out


_CAVEAT = (
    "Overlay honesty: in-sample this underperforms buy-and-hold every calendar year "
    "except 2025; its edge is converting the one -42% DeepSeek drawdown to ~-9%. "
    "Robust core = 'hold less when trend is down and vol is high' -- the rest is tuned."
)


def overlay_table(report: dict) -> str:
    lines = [f"OVERLAY REPORT  {report['start']} -> {report['end']}", ""]
    lines.append(f"  {'config':<24}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}")
    for key, label in [("basket_only", "basket only"),
                       ("trend_gate", "+ trend gate"),
                       ("trend_gate_voltarget", "+ trend gate + vol-target")]:
        m = report[key]["metrics"]
        lines.append(f"  {label:<24}{m['cagr']*100:>8.1f}%{m['sharpe']:>9.2f}{m['max_dd']*100:>8.1f}%")
    me = report["mechanics"]
    lines.append("")
    lines.append(f"  mechanics: avg exposure {me['avg_exposure']}, min {me['min_exposure']}, "
                 f"max {me['max_exposure']}, days gated-out {me['pct_days_gated_out']*100:.0f}%, "
                 f"gate flips {me['gate_flips']}")
    lines.append("")
    lines.append("  parameter plateau (Sharpe | MaxDD%):")
    lines.append(f"    {'':>8}" + "".join(f"{('tv'+str(tv)):>16}" for tv in _PLATEAU_TV))
    for ma in _PLATEAU_MA:
        cells = [c for c in report["plateau_grid"] if c["ma_days"] == ma]
        row = "".join(f"{c['sharpe']:>8.2f} |{c['max_dd']*100:>6.1f}" for c in cells)
        lines.append(f"    MA{ma:>4}  {row}")
    lines.append("")
    lines.append("  " + _CAVEAT)
    return "\n".join(lines)
