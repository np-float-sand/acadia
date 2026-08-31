"""Big-four hyperscaler capex-deceleration signal (probe, 2026-08-31).

Idea: an equity-side demand signal for the grid-equipment basket. Aggregate
quarterly capex of MSFT + Alphabet + Amazon + Meta (from SEC XBRL, discrete
quarters derived from YTD where not tagged directly), used ONE-DIRECTIONALLY:
when big-four capex *growth* rolls over, scale the basket back; never lean in on
it (capex acceleration is the single most-priced datapoint in this theme).

Point-in-time: a quarter's capex is only "known" ~50 days after quarter end
(the last of the four 10-Qs). The signal is lagged accordingly and held until
the next known print.

See docs/triage_2026-08-31-differentiation-ideas.md for why the direct
demand-data ideas were considered weak; this tests the one with a pulse.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from grid_equipment_basket import config, overlay
from grid_equipment_basket.backtest import calendar_year_returns, compute_metrics

_ANN = config.ANN_FACTOR
_CIK = {"MSFT": 789019, "GOOGL": 1652044, "AMZN": 1018724, "META": 1326801}
_CONCEPTS = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
_HDR = {"User-Agent": "acadia-research research@example.com"}
_CACHE = config.CACHE_DIR / "bigfour_capex.parquet"
KNOWN_LAG_DAYS = 50


def _company_quarters(cik: int) -> pd.Series:
    """Discrete calendar-quarter capex for one company. GOOGL/META tag only Q1
    as a discrete 3-month period; the rest of their year is YTD-only. So: for
    each distinct XBRL period `start`, walk the YTD ladder (~90/180/270/365-day
    end points) and take consecutive differences that span ~one quarter.
    Earliest filing kept per (start, end) for point-in-time. Verified against
    known annual totals (GOOGL 2024 ~$53B, META ~$37B, AMZN ~$83B, MSFT ~$56B)."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    facts = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=_HDR), timeout=60))
    rows = []
    for c in _CONCEPTS:
        try:
            us = facts["facts"]["us-gaap"][c]["units"]["USD"]
        except KeyError:
            continue
        for r in us:
            if not r.get("start"):
                continue
            rows.append((pd.Timestamp(r["start"]), pd.Timestamp(r["end"]), float(r["val"]),
                         pd.Timestamp(r["filed"]), c))
    df = pd.DataFrame(rows, columns=["start", "end", "val", "filed", "concept"])
    if df.empty:
        return pd.Series(dtype=float)
    df = df.sort_values(["concept", "filed"]).drop_duplicates(["start", "end"], keep="first")
    df["days"] = (df["end"] - df["start"]).dt.days

    out = {}
    for _, g in df.groupby("start"):
        g = g[g["days"].between(80, 100) | g["days"].between(170, 196)
              | g["days"].between(260, 286) | g["days"].between(350, 381)].sort_values("end")
        prev_v, prev_d = 0.0, 0
        for r in g.itertuples():
            step, q = r.days - prev_d, r.val - prev_v
            if 80 <= step <= 100 and q > 0:
                out[r.end] = q
            prev_v, prev_d = r.val, r.days
    s = pd.Series(out)
    s.index = pd.to_datetime(s.index)
    s = s[s > 0]
    return s.groupby(s.index.to_period("Q")).sum()


def fetch_bigfour_capex(use_cache: bool = True) -> pd.DataFrame:
    """Quarterly big-four aggregate capex ($), with `yoy` growth, `decel`
    (2-quarter change in yoy) and `known_date` (period end + 50d)."""
    if use_cache and _CACHE.exists():
        return pd.read_parquet(_CACHE)
    cols = {t: _company_quarters(cik) for t, cik in _CIK.items()}
    cx = pd.DataFrame(cols)
    agg = cx.sum(axis=1, min_count=1).rename("capex").to_frame()
    agg = agg.loc["2018Q1":].dropna()
    agg["yoy"] = agg["capex"] / agg["capex"].shift(4) - 1.0
    agg["decel2"] = agg["yoy"] - agg["yoy"].shift(2)
    agg["known_date"] = [p.to_timestamp(how="end").normalize() + pd.Timedelta(days=KNOWN_LAG_DAYS)
                         for p in agg.index]
    if use_cache:
        agg.to_parquet(_CACHE)
    return agg


def capex_derisk_multiplier(index: pd.DatetimeIndex, capex: pd.DataFrame, *,
                            floor_yoy: float = 0.15, require_decel: bool = True,
                            lo_mult: float = 0.5) -> pd.Series:
    """Daily exposure multiplier in {lo_mult, 1.0}, aligned to `index`.
    `lo_mult` applies once the most recently *known* big-four capex print shows
    yoy growth below `floor_yoy` (and, if `require_decel`, a negative 2-quarter
    `decel2`); back to 1.0 when the next known print clears the floor. One-way:
    it never exceeds 1.0."""
    idx = pd.DatetimeIndex(index)
    lo = (capex["yoy"] < floor_yoy)
    if require_decel:
        lo = lo & (capex["decel2"] < 0)
    known = pd.Series(np.where(lo.to_numpy(), lo_mult, 1.0), index=capex["known_date"].to_numpy())
    known = known[~known.index.duplicated(keep="last")].sort_index()
    daily = known.reindex(idx, method="ffill")
    return daily.fillna(1.0).rename("capex_mult")


def capex_report(price_fn=None, capex_df=None,
                 windows=None) -> dict:
    price_fn = price_fn or overlay._default_price_fn
    capex_df = fetch_bigfour_capex() if capex_df is None else capex_df
    rf = config.RISK_FREE_RATE
    windows = windows or {"full": ("2020-01-01", "2026-08-31"),
                          "deepseek_episode": ("2024-11-01", "2025-05-31"),
                          "2022_ratespike": ("2022-01-01", "2022-12-31")}
    end = max(b for _, b in windows.values())
    ret, lvl = overlay._basket_series("2020-01-01", end, price_fn)

    cmult = capex_derisk_multiplier(ret.index, capex_df)
    l1 = overlay.apply_overlay(ret, lvl, rf)
    capex_only = overlay.apply_overlay_l2(ret, cmult, rf)          # de-risk only, vol-target on
    both = overlay.apply_overlay_l2(overlay.apply_overlay(ret, lvl, rf).rename("r"), cmult, rf) \
        if False else None
    # "both" = price gate AND capex de-risk, then one vol target: multiply the two gates
    gate = overlay.trend_gate(lvl, config.OVERLAY_MA_DAYS).reindex(ret.index).fillna(1.0)
    both = overlay.apply_overlay_l2(ret, (gate * cmult).clip(upper=1.0), rf)

    def block(s):
        return {wk: {"metrics": compute_metrics(s.loc[a:b].dropna(), rf, _ANN),
                     "calendar": calendar_year_returns(s.loc[a:b])}
                for wk, (a, b) in windows.items()}

    return {"windows": windows,
            "capex_known": capex_df[["capex", "yoy", "decel2", "known_date"]].assign(
                capex=(capex_df["capex"] / 1e9).round(1)).rename_axis("period")
                .reset_index().assign(period=lambda d: d["period"].astype(str)).to_dict("records"),
            "series": {"buy_and_hold": block(ret), "price_gate": block(l1),
                       "capex_derisk": block(capex_only), "price_gate_plus_capex": block(both)},
            "mult_mechanics": {"pct_derisked": round(float((cmult < 1.0).mean()), 3),
                               "derisk_starts": [str(d.date()) for d in
                                                 cmult.index[(cmult.diff() < 0)]]}}


def capex_table(rep: dict) -> str:
    L = ["BIG-FOUR CAPEX-DECELERATION DE-RISK  (one-directional)"]
    kk = rep["capex_known"]
    L.append("  known capex prints (yoy growth / 2q decel):")
    for r in kk:
        if pd.Timestamp(r["known_date"]) < pd.Timestamp("2020-01-01"):
            continue
        L.append(f"    {r['period']:<7} ${r['capex']:>6.1f}B  yoy {r['yoy']*100:>+6.1f}%  "
                 f"decel2 {r['decel2']*100:>+6.1f}pp   known {str(pd.Timestamp(r['known_date']).date())}")
    L.append("")
    for wk in rep["windows"]:
        L.append(f"  [{wk}]")
        L.append(f"    {'':<24}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}")
        for key, lab in [("buy_and_hold", "buy & hold"), ("price_gate", "price gate + VT"),
                         ("capex_derisk", "capex de-risk + VT"),
                         ("price_gate_plus_capex", "price gate + capex")]:
            m = rep["series"][key][wk]["metrics"]
            L.append(f"    {lab:<24}{_p(m['cagr']):>9}{_f(m['sharpe']):>8}{_p(m['max_dd']):>9}")
        L.append("")
    me = rep["mult_mechanics"]
    L.append(f"  de-risked {me['pct_derisked']*100:.0f}% of days; de-risk onsets: {me['derisk_starts']}")
    return "\n".join(L)


def _p(x):
    return "n/a" if x != x else f"{x*100:.1f}%"


def _f(x):
    return "n/a" if x != x else f"{x:.2f}"
