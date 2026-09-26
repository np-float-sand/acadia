"""Point-in-time trailing-twelve-month fundamentals from EDGAR facts.

The whole point of this module is the `filed` column. A fact is only visible on
dates at or after the day it was filed with the SEC, so a backtest never sees a
number before the market did. Period-end dates are used only to order quarters
within the trailing window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from capex_intensity_strategy import config


def load_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw facts frame: typed dates, numeric values, deduplicated."""
    out = facts.copy()
    for col in ("start", "end", "filed"):
        out[col] = pd.to_datetime(out[col])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    # Synthetic/legacy frames carry no tag provenance; treat them as one series.
    if "tag" not in out.columns:
        out["tag"] = out["item"]
    if "rank" not in out.columns:
        out["rank"] = 0
    # A quarter is often restated. Keep the latest filing for each period, which
    # is what an investor would have seen most recently as of any later date.
    out = out.sort_values("filed")
    return out.reset_index(drop=True)


def derive_quarters(facts: pd.DataFrame) -> pd.DataFrame:
    """Discrete quarters from a mix of discrete and cumulative flow facts.

    Income-statement items are usually tagged as discrete three-month periods, but
    cash-flow items -- capital expenditure above all -- are tagged year-to-date: a
    10-Q reports three, six and nine months from the fiscal-year start and the 10-K
    reports twelve. Only Q1 is ever a standalone quarter. Every later quarter is
    recovered by differencing consecutive rungs of that ladder (Q4 = FY - 9M), and
    the derived quarter inherits the *later* of the two filing dates, so it becomes
    visible only once the filing that completes it exists.

    Facts whose span is already a single quarter are passed through unchanged and
    win over a derived value for the same period end.
    """
    if facts.empty:
        return facts
    f = facts.copy()
    f["span"] = (f["end"] - f["start"]).dt.days
    is_quarter = f["span"].between(config.QUARTER_MIN_DAYS, config.QUARTER_MAX_DAYS)
    direct = f[is_quarter].assign(derived=False)

    # One value per reported period: the most recently filed (load_facts sorts by filed).
    ladder = (f.sort_values("filed")
              .groupby(["ticker", "item", "tag", "rank", "start", "end"], as_index=False)
              .last()
              .sort_values(["ticker", "item", "tag", "start", "end"]))
    key = ["ticker", "item", "tag", "start"]
    prev_end = ladder.groupby(key)["end"].shift(1)
    prev_value = ladder.groupby(key)["value"].shift(1)
    prev_filed = ladder.groupby(key)["filed"].shift(1)
    derived = ladder.assign(
        start=prev_end + pd.Timedelta(days=1),
        value=ladder["value"] - prev_value,
        filed=np.maximum(ladder["filed"], prev_filed),
        derived=True,
    ).dropna(subset=["start", "value"])
    derived["span"] = (derived["end"] - derived["start"]).dt.days
    derived = derived[derived["span"].between(config.QUARTER_MIN_DAYS, config.QUARTER_MAX_DAYS)]

    out = pd.concat([direct, derived], ignore_index=True)
    # A reported quarter beats a derived one for the same period end, and the most
    # recently filed vintage of that period beats an earlier restatement.
    out = (out.sort_values(["ticker", "item", "tag", "end", "derived", "filed"],
                           ascending=[True, True, True, True, True, False])
           .groupby(["ticker", "item", "tag", "rank", "end"], as_index=False)
           .first())
    return out.sort_values("filed").reset_index(drop=True)


def ttm_panel(facts: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Trailing-twelve-month sums per (date, ticker, item), as known on each date.

    For each observation date we take only facts already *filed*, pick for each
    ticker-item the single tag with the most visible quarters (ties broken by
    `FACT_TAGS` preference order), keep the most recently filed value for each
    period end, and sum the last four quarters.

    Four guards keep a sum from being a number it is not:

    * fewer than four filed quarters -> omitted rather than annualised;
    * the four quarters must actually *span* about a year (`TTM_SPAN_*_DAYS`), so
      four Q1s from four different years are rejected;
    * the newest quarter must be recent as of the observation date
      (`MAX_FACT_STALENESS_DAYS`), so a filer that stopped tagging an item is not
      scored off years-old numbers;
    * `end_<item>` is returned so the caller can require revenue and capex to
      cover the same window.
    """
    facts = load_facts(facts)
    records = []
    for asof in dates:
        # A trailing-twelve-month window never reaches back more than the span
        # guard plus the staleness guard, so older periods are dropped before the
        # (expensive) differencing rather than after it.
        horizon = asof - pd.Timedelta(days=config.TTM_SPAN_MAX_DAYS
                                      + config.MAX_FACT_STALENESS_DAYS + 200)
        visible = facts[(facts["filed"] <= asof) & (facts["end"] >= horizon)]
        if visible.empty:
            continue
        # Differencing is done on what was filed *by this date*, so a quarter is
        # built from the vintages an investor actually had, never a later restatement.
        visible = derive_quarters(visible)
        if visible.empty:
            continue
        for (ticker, item), group in visible.groupby(["ticker", "item"], sort=False):
            # Try tags newest-series-first, then best-covered, then in FACT_TAGS
            # preference order, and keep the first that yields a valid TTM. A filer
            # that abandoned a tag years ago (DUK's SalesRevenueNet) falls through to
            # the one it actually files under rather than being dropped.
            order = (group.groupby(["tag", "rank"])
                     .agg(latest=("end", "max"), n=("end", "nunique"))
                     .reset_index()
                     .sort_values(["latest", "n", "rank"], ascending=[False, False, True]))
            for tag in order["tag"]:
                tail = (group[group["tag"] == tag]
                        .groupby("end", as_index=False).last()
                        .sort_values("end").tail(config.TTM_QUARTERS))
                if len(tail) < config.TTM_QUARTERS:
                    continue
                span = (tail["end"].max() - tail["start"].min()).days
                if not config.TTM_SPAN_MIN_DAYS <= span <= config.TTM_SPAN_MAX_DAYS:
                    continue
                if (asof - tail["end"].max()).days > config.MAX_FACT_STALENESS_DAYS:
                    continue
                records.append((asof, ticker, item, float(tail["value"].sum()),
                                tail["end"].max()))
                break
    if not records:
        return pd.DataFrame(columns=["date", "ticker", "revenue", "capex"])
    tidy = pd.DataFrame(records, columns=["date", "ticker", "item", "ttm", "end"])
    wide = tidy.pivot_table(index=["date", "ticker"], columns="item",
                            values="ttm").reset_index()
    ends = tidy.pivot_table(index=["date", "ticker"], columns="item",
                            values="end", aggfunc="last").reset_index()
    wide.columns.name = None
    ends.columns.name = None
    items = sorted(set(tidy["item"]) | {"revenue", "capex"})
    for col in items:
        if col not in wide.columns:
            wide[col] = np.nan
        if col not in ends.columns:
            ends[col] = pd.NaT
    ends = ends.rename(columns={c: f"end_{c}" for c in items})
    wide = wide.merge(ends[["date", "ticker"] + [f"end_{c}" for c in items]],
                      on=["date", "ticker"], how="left")
    return wide[["date", "ticker"] + items + [f"end_{c}" for c in items]]


def capex_intensity(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide date x ticker frame of TTM capex / TTM revenue.

    This is the score. Low = asset-light, sells into the capital cycle.
    High = asset-heavy, *is* the capital cycle. Non-positive revenue yields NaN
    rather than a nonsense ratio, and so does a capex window that does not end on
    the same quarter as the revenue window -- dividing a current-year revenue TTM
    by a years-old capex TTM is what put two regulated utilities in the long leg.
    """
    if panel.empty:
        return pd.DataFrame()
    out = panel.copy()
    revenue = out["revenue"].where(out["revenue"] > 0)
    out["score"] = out["capex"] / revenue
    if {"end_revenue", "end_capex"} <= set(out.columns):
        aligned = out["end_revenue"].eq(out["end_capex"])
        out["score"] = out["score"].where(aligned)
    # dropna=False keeps a ticker whose score is entirely NaN as an explicit NaN
    # column, so it is excluded by the rank step rather than silently vanishing.
    wide = out.pivot_table(index="date", columns="ticker", values="score", dropna=False)
    wide.columns.name = None
    return wide.sort_index()
