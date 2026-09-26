"""
RESEARCH-LOG open item 23 -- industrial electricity usage growth as a broad-market
(SPY) timing signal (Da, Huang & Yun 2017, JFQA).

Pre-registered spec: docs/electricity-usage-spy-timing-preregistration.md
  - series: EIA electricity/retail-sales, sectorid=IND, stateid=US, data=sales (monthly)
  - regressor g_t: YoY log growth of a trailing-3-month average of sales (+ raw-YoY
    robustness variant)
  - lag: sales month t treated as first knowable at month-end (t+2); signal used from
    the following month (mirrors grid_equipment_basket/overlay.py's month-end-decided,
    next-month-held convention)
  - target: SPY forward 12m return (primary, matches the paper's own horizon) and
    forward 1m return (secondary, the horizon this project's overlay stack can consume)
  - fit window 2001-01..2014-12 (effective-month basis), holdout 2015-01..present
  - PRIMARY GATE: |HAC t-stat| >= 2.0 AND correct (negative) sign on the fit-window,
    fwd-12m, smoothed-regressor coefficient. Fails -> stop, no portfolio-overlay build.

Throwaway. Findings -> docs/electricity-usage-spy-timing-results.md
(docs/RESEARCH-LOG.md updated separately by the orchestrating session, not this script).
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import statsmodels.api as sm

EIA_KEY = os.environ.get("EIA_API_KEY")
if not EIA_KEY:
    for candidate in ("/Users/sandhyapersad/acadia/.env",
                      os.path.join(os.path.dirname(__file__), "..", "..", ".env")):
        if os.path.exists(candidate):
            with open(candidate) as f:
                for line in f:
                    if line.startswith("EIA_API_KEY"):
                        EIA_KEY = line.strip().split("=", 1)[1]
            if EIA_KEY:
                break

_HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(_HERE, "data")
os.makedirs(CACHE_DIR, exist_ok=True)

LAG_MONTHS = 2          # sales month t -> first knowable at month-end (t + LAG_MONTHS)
FIT_START = "2001-01"
FIT_END = "2014-12"
HOLDOUT_START = "2015-01"
GATE_T = 2.0


# --------------------------------------------------------------------------- data ----
def fetch_industrial_sales(use_cache: bool = True) -> pd.Series:
    """Monthly US industrial electricity sales (million kWh), national aggregate."""
    cache = os.path.join(CACHE_DIR, "eia_industrial_sales_us.parquet")
    if use_cache and os.path.exists(cache):
        return pd.read_parquet(cache)["sales"]
    params = {
        "api_key": EIA_KEY,
        "frequency": "monthly",
        "data[0]": "sales",
        "facets[sectorid][]": "IND",
        "facets[stateid][]": "US",
        "start": "2001-01",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": "5000",
    }
    url = "https://api.eia.gov/v2/electricity/retail-sales/data/?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    df = pd.DataFrame(d["response"]["data"])
    df["period"] = pd.PeriodIndex(df["period"], freq="M")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    s = df.set_index("period")["sales"].sort_index()
    s.to_frame().to_parquet(cache)
    return s


def fetch_spy_monthly(use_cache: bool = True) -> pd.Series:
    """SPY monthly auto-adjusted close, indexed by PeriodIndex[M]."""
    cache = os.path.join(CACHE_DIR, "spy_monthly.parquet")
    if use_cache and os.path.exists(cache):
        return pd.read_parquet(cache)["close"]
    import yfinance as yf
    px = yf.download("SPY", start="2000-06-01", interval="1mo", auto_adjust=True,
                     progress=False)["Close"]
    if hasattr(px, "columns"):
        px = px.iloc[:, 0]
    px.index = pd.PeriodIndex(px.index, freq="M")
    s = px.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "close"
    s.to_frame().to_parquet(cache)
    return s


# ---------------------------------------------------------------------- transform ----
def build_regressors(sales: pd.Series) -> pd.DataFrame:
    """g_smooth / g_raw indexed by *effective month-end* (sales period + LAG_MONTHS),
    matching the pre-registered "first knowable at t+2" lag treatment."""
    roll3 = sales.rolling(3).mean()
    g_smooth = np.log(roll3) - np.log(roll3.shift(12))
    g_raw = np.log(sales) - np.log(sales.shift(12))
    out = pd.DataFrame({"g_smooth": g_smooth, "g_raw": g_raw})
    out.index = out.index + LAG_MONTHS
    out.index.name = "effective_month_end"
    return out


def fwd_returns(spy: pd.Series, anchor_index: pd.PeriodIndex) -> pd.DataFrame:
    """fwd1 / fwd12 SPY returns entered the month AFTER each anchor month-end."""
    fwd1, fwd12 = {}, {}
    for a in anchor_index:
        p0 = spy.get(a)
        p1 = spy.get(a + 1)
        p12 = spy.get(a + 12)
        fwd1[a] = (p1 / p0 - 1.0) if (p0 is not None and p1 is not None
                                       and pd.notna(p0) and pd.notna(p1)) else np.nan
        fwd12[a] = (p12 / p0 - 1.0) if (p0 is not None and p12 is not None
                                         and pd.notna(p0) and pd.notna(p12)) else np.nan
    return pd.DataFrame({"fwd1": pd.Series(fwd1), "fwd12": pd.Series(fwd12)})


# ------------------------------------------------------------------------- stats -----
def hac_ols(y: pd.Series, x: pd.Series, maxlags: int):
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(d) < 8:
        return None
    X = sm.add_constant(d["x"])
    model = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "n": int(len(d)),
        "b": float(model.params["x"]),
        "t": float(model.tvalues["x"]),
        "p": float(model.pvalues["x"]),
        "r2": float(model.rsquared),
    }


def fmt(r):
    if r is None:
        return "  n<8, skipped"
    sign = "neg" if r["b"] < 0 else "pos"
    return (f"b={r['b']:+.4f} ({sign})  t={r['t']:+.2f}  R2={r['r2']:.3f}  n={r['n']}")


# --------------------------------------------------------------------------- main ----
def main():
    sales = fetch_industrial_sales()
    print(f"EIA industrial sales: {sales.index.min()}..{sales.index.max()} "
          f"({len(sales)} months, US total, million kWh)")

    spy = fetch_spy_monthly()
    print(f"SPY monthly close: {spy.index.min()}..{spy.index.max()} ({len(spy)} months)\n")

    reg = build_regressors(sales)
    fwd = fwd_returns(spy, reg.index)
    panel = reg.join(fwd).dropna(subset=["g_smooth", "g_raw"], how="all")
    panel = panel[panel.index >= "2001-01"]

    fit = panel.loc[(panel.index >= FIT_START) & (panel.index <= FIT_END)]
    last_fwd12_ok = panel["fwd12"].last_valid_index()
    holdout = panel.loc[(panel.index >= HOLDOUT_START) & (panel.index <= last_fwd12_ok)] \
        if last_fwd12_ok is not None else panel.iloc[0:0]

    print(f"panel: {panel.index.min()}..{panel.index.max()}  n={len(panel)} "
          f"(effective-month basis, lag={LAG_MONTHS}m)")
    print(f"fit window   {FIT_START}..{FIT_END}: n={len(fit)}")
    print(f"holdout window {HOLDOUT_START}..{last_fwd12_ok}: n={len(holdout)}\n")

    # ---- PRIMARY GATE ----
    print("=" * 78)
    print("PRIMARY GATE: fit-window, fwd-12m, smoothed regressor, |t|>=2.0, sign<0")
    print("=" * 78)
    r_primary = hac_ols(fit["fwd12"], fit["g_smooth"], maxlags=11)
    print("fit fwd12 ~ g_smooth :", fmt(r_primary))
    gate_pass = (r_primary is not None and abs(r_primary["t"]) >= GATE_T and r_primary["b"] < 0)
    print(f"\n>>> PRIMARY GATE: {'PASS' if gate_pass else 'FAIL'} <<<\n")

    # ---- always report these for the record, regardless of gate outcome ----
    print("-" * 78)
    print("full-sample (all available months, descriptive only -- NOT the gate):")
    print("-" * 78)
    r_full12 = hac_ols(panel["fwd12"], panel["g_smooth"], maxlags=11)
    r_full1 = hac_ols(panel["fwd1"], panel["g_smooth"], maxlags=1)
    print("full  fwd12 ~ g_smooth:", fmt(r_full12))
    print("full  fwd1  ~ g_smooth:", fmt(r_full1))

    out = {
        "sales_range": [str(sales.index.min()), str(sales.index.max())],
        "spy_range": [str(spy.index.min()), str(spy.index.max())],
        "panel_n": len(panel),
        "fit_n": len(fit),
        "holdout_n": len(holdout),
        "primary_gate": {"result": r_primary, "pass": bool(gate_pass)},
        "full_sample": {"fwd12_g_smooth": r_full12, "fwd1_g_smooth": r_full1},
    }

    if gate_pass:
        print("\n" + "=" * 78)
        print("GATE PASSED -- running secondary/descriptive checks")
        print("=" * 78)

        # fwd-1m fit-window (own check, the horizon the overlay stack can use)
        r_fit_fwd1 = hac_ols(fit["fwd1"], fit["g_smooth"], maxlags=1)
        print("fit   fwd1  ~ g_smooth:", fmt(r_fit_fwd1))

        # raw-YoY robustness
        r_fit_raw12 = hac_ols(fit["fwd12"], fit["g_raw"], maxlags=11)
        print("fit   fwd12 ~ g_raw   :", fmt(r_fit_raw12))

        # holdout refit (sign-consistency check)
        r_hold12 = hac_ols(holdout["fwd12"], holdout["g_smooth"], maxlags=11)
        print("holdout fwd12 ~ g_smooth (refit):", fmt(r_hold12))

        # out-of-sample prediction using fit-estimated (a,b) applied to holdout g_t
        if r_primary is not None:
            a_hat = None
            # refit to get intercept too (hac_ols doesn't return it -- quick redo)
            d = pd.concat([fit["fwd12"].rename("y"), fit["g_smooth"].rename("x")],
                          axis=1).dropna()
            X = sm.add_constant(d["x"])
            m = sm.OLS(d["y"], X).fit()
            a_hat, b_hat = float(m.params["const"]), float(m.params["x"])
            hd = pd.concat([holdout["fwd12"].rename("y"), holdout["g_smooth"].rename("x")],
                           axis=1).dropna()
            if len(hd) >= 8:
                pred = a_hat + b_hat * hd["x"]
                oos_corr = float(np.corrcoef(pred, hd["y"])[0, 1])
                ss_res = float(((hd["y"] - pred) ** 2).sum())
                ss_tot = float(((hd["y"] - hd["y"].mean()) ** 2).sum())
                oos_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                print(f"OOS holdout: corr(pred,actual)={oos_corr:+.3f}  "
                      f"OOS R2={oos_r2:+.3f}  n={len(hd)}")
                out["holdout_oos"] = {"corr": oos_corr, "r2": oos_r2, "n": len(hd)}

        # descriptive tercile split (full sample, smoothed regressor, fwd12)
        gp = panel.dropna(subset=["g_smooth", "fwd12"])
        terc = pd.qcut(gp["g_smooth"], 3, labels=["low", "mid", "high"])
        tmeans = gp.groupby(terc, observed=True)["fwd12"].mean()
        print("\ntercile means, fwd12 return by g_smooth tercile (descriptive, full sample):")
        print(tmeans.to_string())
        out["tercile_fwd12_means"] = tmeans.to_dict()

        out["secondary"] = {
            "fit_fwd1_g_smooth": r_fit_fwd1,
            "fit_fwd12_g_raw": r_fit_raw12,
            "holdout_fwd12_g_smooth_refit": r_hold12,
        }
    else:
        print("\nGate failed on the pre-registered primary spec -- per pre-registration,")
        print("stopping here. No holdout interpretation, no portfolio-overlay build.")

    with open(os.path.join(CACHE_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    panel.to_csv(os.path.join(CACHE_DIR, "panel.csv"))
    print(f"\nwrote {CACHE_DIR}/results.json, {CACHE_DIR}/panel.csv")


if __name__ == "__main__":
    main()
