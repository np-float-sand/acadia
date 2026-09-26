"""
Build the point-in-time forward spark-spread series from the STEO vintage panel
(data/steo_panel.parquet) and turn it into the same style of daily regressor that fix #2
(docs/merchant-beta-orthogonalization-results.md) used for its trailing-LMP power trend --
so it's a drop-in, like-for-like replacement in the stress-beta orthogonalization probe.

Spark spread definition (industry-standard, gross-margin proxy for a gas-fired generator):
    spark_spread ($/MWh) = power_price ($/MWh) - HEAT_RATE (MMBtu/MWh) * gas_price ($/MMBtu)

HEAT_RATE = 7.5 MMBtu/MWh is an assumed constant (a round mid-efficiency CCGT-fleet number,
not calibrated per-ISO or per-year) -- a deliberately simple first-pass choice per the task's
"pre-register nothing elaborate" instruction. This is a LEVEL assumption; since the signal we
actually need is *how the forward view changes vintage to vintage*, a constant offset in
heat-rate does not change the sign or shape of that change, only its level.

Per vintage, the "forward view" scalar is the average of ELWHU_{PJ,TX} and NGHHUUS over all
periods that vintage's own 'Dates' sheet flagged as forecast (is_historical == 0) -- i.e. the
average of STEO's own ~12-18-month-ahead outlook as published in that release, not a single
noisy month. This is compared, vintage to vintage, purely using each vintage's own Wayback
capture date as the "known as of" anchor (never a later, revised vintage for an earlier date
-- the whole point of building this as a vintage panel instead of pulling the current API).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
PANEL_PATH = HERE / "data" / "steo_panel.parquet"
OUT_PATH = HERE / "data" / "spark_spread_vintages.parquet"

HEAT_RATE = 7.5  # MMBtu/MWh, assumed constant -- see module docstring

ISO_SERIES = {
    "PJM": "ELWHU_PJ",
    "ERCOT": "ELWHU_TX",
}


def _vintage_date(vintage_ts: str) -> pd.Timestamp:
    return pd.to_datetime(vintage_ts[:8], format="%Y%m%d")


def build_vintage_table() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    rows = []
    for vintage_ts, grp in panel.groupby("vintage_ts"):
        forecast_only = grp[grp["is_historical"] == 0]
        if forecast_only.empty:
            continue
        gas = forecast_only.loc[forecast_only["series"] == "NGHHUUS", "value"]
        if gas.empty or gas.isna().all():
            continue
        gas_fwd = gas.mean()

        rec = {
            "vintage_ts": vintage_ts,
            "vintage_date": _vintage_date(vintage_ts),
            "forecast_month": grp["forecast_month"].iloc[0],
            "n_forecast_months": forecast_only["period"].nunique(),
            "gas_fwd_hhub": gas_fwd,
        }
        any_power = False
        for iso, series in ISO_SERIES.items():
            power = forecast_only.loc[forecast_only["series"] == series, "value"]
            if power.empty or power.isna().all():
                rec[f"power_fwd_{iso}"] = np.nan
                rec[f"spark_spread_{iso}"] = np.nan
                continue
            any_power = True
            power_fwd = power.mean()
            rec[f"power_fwd_{iso}"] = power_fwd
            rec[f"spark_spread_{iso}"] = power_fwd - HEAT_RATE * gas_fwd
        if not any_power:
            continue
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("vintage_date").reset_index(drop=True)
    return df


def build_daily_trend(vintage_df: pd.DataFrame, iso: str, start: str, end: str) -> pd.Series:
    """
    Same construction as fix #2's `build_power_trend`, applied to the STEO forward
    spark-spread level instead of trailing realized LMP:
      1. forward-fill the vintage-level series onto a daily calendar (point-in-time --
         a value is only "known" from its vintage_date onward)
      2. trailing 252-calendar-day % change
      3. rolling-756-day z-score, clipped to +-3
    """
    col = f"spark_spread_{iso}"
    s = vintage_df.dropna(subset=[col]).set_index("vintage_date")[col]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    # One vintage (Dec-2021 ERCOT) carries a verified-genuine but extreme STEO forecast
    # anomaly (raw source alternates ~$35 / ~$400 month to month, confirmed against the
    # raw workbook cells -- not a parsing bug, likely a residual post-Uri scarcity-scenario
    # artifact in EIA's own ERCOT sub-model). Winsorize the vintage *level* before taking
    # trailing % change so one anomalous release doesn't dominate the regressor.
    s = s.clip(-50, 100)

    daily_index = pd.date_range(min(s.index.min(), pd.Timestamp(start)), max(s.index.max(), pd.Timestamp(end)), freq="D")
    daily = s.reindex(daily_index).ffill()
    daily = daily.loc[start:end]

    trend_raw = daily.pct_change(252)
    roll_mean = trend_raw.rolling(756, min_periods=180).mean()
    roll_std = trend_raw.rolling(756, min_periods=180).std()
    z = ((trend_raw - roll_mean) / roll_std.replace(0, np.nan)).clip(-3, 3)
    return z.dropna()


def main():
    from grid_resilience.config import BACKTEST_START, BACKTEST_END

    vdf = build_vintage_table()
    vdf.to_parquet(OUT_PATH)
    print(f"Built {len(vdf)} usable vintages (with >=1 ISO's power series present)")
    print(vdf[["vintage_ts", "forecast_month", "n_forecast_months",
               "power_fwd_PJM", "power_fwd_ERCOT", "gas_fwd_hhub",
               "spark_spread_PJM", "spark_spread_ERCOT"]].to_string())

    print(f"\nDate range: {vdf['vintage_date'].min().date()} -> {vdf['vintage_date'].max().date()}")

    for iso in ISO_SERIES:
        trend = build_daily_trend(vdf, iso, BACKTEST_START, BACKTEST_END)
        print(f"\n{iso} daily trend series: {len(trend)} obs, "
              f"{trend.index.min().date() if len(trend) else None} -> "
              f"{trend.index.max().date() if len(trend) else None}")
        trend.to_frame(name="steo_spark_trend_z").to_parquet(HERE / "data" / f"steo_trend_{iso}.parquet")


if __name__ == "__main__":
    main()
