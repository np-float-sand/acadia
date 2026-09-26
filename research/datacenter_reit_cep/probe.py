"""
PROBE — commercial electricity price (CEP) as a data-center-REIT signal.

Replicates (own-horizon, reduced controls) Khaliq & Boufous (2026), "Do energy price shocks
affect U.S. data center REIT returns?": higher CEP -> lower DC-REIT returns.

Pre-registered design: docs/datacenter-reit-cep-preregistration.md (read before touching output).

Target: equal-weighted DLR+EQIX+IRM monthly return (time-series signal, not cross-sectional --
n=months, not n=names, per the handoff's statistical-power note).

Gates (both |t| >= 2.0, fixed before running):
  G1 linear    : OLS portfolio_ret ~ cep_chg + hh_chg + wti_chg + vix_chg + spy_ret + gs10_chg,
                 HAC(maxlags=6). Sign expected negative on cep_chg.
  G2 nonlinear : time-series tercile spread (top-CEP-chg months minus bottom-CEP-chg months),
                 one-sample t-test.
Both reported over 3 windows: paper's Dec2015-Dec2025, 2023-01-01-present, 2026-YTD (context only).

Throwaway. Findings -> docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from electrification_strategy.fred import fetch_series  # noqa: E402

EIA_KEY = os.environ.get("EIA_API_KEY", "")
TICKERS = ["DLR", "EQIX", "IRM"]

WINDOWS = {
    "paper_2015-2025": ("2015-12-01", "2025-12-31"),
    "2023-onward":     ("2023-01-01", "2026-12-31"),
    "2026-YTD":        ("2026-01-01", "2026-12-31"),
}
GATE_T = 2.0


# ----------------------------- data -----------------------------
def eia_national_cep() -> pd.Series:
    """Monthly national commercial retail electricity price, cents/kWh."""
    url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
    params = [("api_key", EIA_KEY), ("frequency", "monthly"), ("data[0]", "price"),
              ("facets[sectorid][]", "COM"), ("facets[stateid][]", "US"),
              ("start", "2015-01"), ("end", "2026-12"),
              ("sort[0][column]", "period"), ("sort[0][direction]", "asc"), ("length", "500")]
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["response"]["data"])
    d["period"] = pd.PeriodIndex(d["period"], freq="M")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    return d.set_index("period")["price"].sort_index()


def monthly_yf(tickers, start="2015-01-01", end="2026-12-31") -> pd.DataFrame:
    px = yf.download(tickers, start=start, end=end, interval="1mo",
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0] if isinstance(tickers, list) else tickers)
    px.index = pd.PeriodIndex(px.index, freq="M")
    return px


def build_panel() -> pd.DataFrame:
    cep = eia_national_cep()
    cep_chg = cep.pct_change().rename("cep_chg")

    reit_px = monthly_yf(TICKERS)
    reit_ret = reit_px.pct_change().mean(axis=1).rename("portfolio_ret")

    ng = monthly_yf(["NG=F"])["NG=F"]
    hh_chg = ng.pct_change().rename("hh_chg")

    spy = monthly_yf(["SPY"])["SPY"]
    spy_ret = spy.pct_change().rename("spy_ret")

    wti = fetch_series("DCOILWTICO", "2014-01-01", "2026-12-31")
    wti_m = wti.resample("ME").last()
    wti_m.index = pd.PeriodIndex(wti_m.index, freq="M")
    wti_chg = wti_m.pct_change().rename("wti_chg")

    vix = fetch_series("VIXCLS", "2014-01-01", "2026-12-31")
    vix_m = vix.resample("ME").last()
    vix_m.index = pd.PeriodIndex(vix_m.index, freq="M")
    vix_chg = vix_m.pct_change().rename("vix_chg")

    gs10 = fetch_series("GS10", "2014-01-01", "2026-12-31")
    gs10_m = gs10.resample("ME").last()
    gs10_m.index = pd.PeriodIndex(gs10_m.index, freq="M")
    gs10_chg = gs10_m.diff().rename("gs10_chg")

    panel = pd.concat([reit_ret, cep_chg, hh_chg, wti_chg, vix_chg, spy_ret, gs10_chg], axis=1)
    return panel.dropna()


# ----------------------------- gates -----------------------------
def g1_linear(panel: pd.DataFrame, label: str):
    if len(panel) < 12:
        print(f"  G1 [{label}]: n={len(panel)} too small, skipped")
        return
    X = sm.add_constant(panel[["cep_chg", "hh_chg", "wti_chg", "vix_chg", "spy_ret", "gs10_chg"]])
    y = panel["portfolio_ret"]
    md = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    t = md.tvalues["cep_chg"]
    coef = md.params["cep_chg"]
    verdict = "PASS" if abs(t) >= GATE_T else "FAIL"
    sign = "expected(-)" if coef < 0 else "unexpected(+)"
    print(f"  G1 [{label}] n={len(panel):3d}  cep_chg coef={coef:+.4f}  t={t:+.2f}  "
          f"{sign}  -> {verdict}")


def g2_tercile(panel: pd.DataFrame, label: str):
    if len(panel) < 12:
        print(f"  G2 [{label}]: n={len(panel)} too small, skipped")
        return
    d = panel[["cep_chg", "portfolio_ret"]].dropna()
    terc = pd.qcut(d["cep_chg"], 3, labels=["low", "mid", "high"], duplicates="drop")
    top = d.loc[terc == "high", "portfolio_ret"]
    bot = d.loc[terc == "low", "portfolio_ret"]
    if len(top) < 3 or len(bot) < 3:
        print(f"  G2 [{label}]: too few per tercile, skipped")
        return
    from scipy import stats
    tstat, pval = stats.ttest_ind(top, bot, equal_var=False)
    verdict = "PASS" if abs(tstat) >= GATE_T else "FAIL"
    print(f"  G2 [{label}] n_top={len(top)} n_bot={len(bot)}  "
          f"mean(top)={top.mean():+.4f} mean(bot)={bot.mean():+.4f}  "
          f"spread={top.mean()-bot.mean():+.4f}  t={tstat:+.2f}  -> {verdict}")


def secondary_forward_ic(panel: pd.DataFrame, label: str):
    """Secondary: forward 1m/3m rank-corr of cep_chg vs portfolio return (context only --
    n=1 portfolio series, so this is a pooled Spearman corr, not a Fama-MacBeth t-stat)."""
    for fwd in (1, 3):
        fwd_ret = panel["portfolio_ret"].shift(-fwd)
        m = pd.concat([panel["cep_chg"], fwd_ret.rename("fwd")], axis=1).dropna()
        if len(m) < 12:
            continue
        ic = m["cep_chg"].corr(m["fwd"], method="spearman")
        print(f"  secondary fwd-{fwd}m rank-corr(cep_chg, ret) [{label}]: {ic:+.3f} (n={len(m)}, "
              f"context only)")


def main():
    print("=== CEP -> data-center-REIT probe (DLR/EQIX/IRM equal-weight portfolio) ===\n")
    panel = build_panel()
    print(f"Full panel: {len(panel)} months, {panel.index.min()}..{panel.index.max()}\n")

    for label, (start, end) in WINDOWS.items():
        sub = panel.loc[pd.Period(start, "M"):pd.Period(end, "M")]
        print(f"--- window {label} ({start}..{end}): n={len(sub)} months ---")
        g1_linear(sub, label)
        g2_tercile(sub, label)
        secondary_forward_ic(sub, label)
        print()

    print("(Gate: |t| >= 2.0 on G1 cep_chg coef and G2 tercile spread, in >=2 of the 3 windows "
          "excluding 2026-YTD context-only window, per pre-registration Decision 7/8.)")


if __name__ == "__main__":
    main()
