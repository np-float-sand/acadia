"""
PROBE B — electricity-cost exposure cross-section.

signal_i(q) = - intensity_i * pct_change( weighted state industrial power price )_i(q)
  (rising local power cost -> negative signal -> expect margin pressure / weak return)

Research tree (pre-registered gates):
  G1 fundamentals : rank-IC( signal(q), d operating-margin(q -> q+1) ) , |t| >= 2
  G2 returns      : rank-IC( signal, fwd 1m & 3m return ) , |t| >= 2  (Fama-MacBeth)
  G3 neutralized  : signal coef survives FF5+MOM+sector panel regression, |t| >= 2
  G4 breadth      : signal defined for >= 15 names
  G5 regimes      : G2 sign-stable in 2018-2021 AND 2022-2025
  G6 cost         : quarterly turnover -> implied cost drag (report only)

Throwaway. Findings -> docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from edgar import quarterly_flow, quarterly_gross_margin   # noqa: E402
from universe import UNIVERSE, ALL_STATES                   # noqa: E402

EIA_KEY = os.environ.get("EIA_API_KEY", "")
TCK = [u["ticker"] for u in UNIVERSE]


# ----------------------------- data -----------------------------
def eia_state_industrial_price() -> pd.DataFrame:
    """Monthly industrial retail electricity price (cents/kWh) by state -> quarterly."""
    frames = []
    url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
    for i in range(0, len(ALL_STATES), 8):
        chunk = ALL_STATES[i:i + 8]
        params = [("api_key", EIA_KEY), ("frequency", "monthly"), ("data[0]", "price"),
                  ("facets[sectorid][]", "IND"), ("start", "2016-01"), ("end", "2026-06"),
                  ("sort[0][column]", "period"), ("sort[0][direction]", "asc"),
                  ("length", "5000")]
        for s in chunk:
            params.append(("facets[stateid][]", s))
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        frames.append(pd.DataFrame(r.json()["response"]["data"]))
    d = pd.concat(frames, ignore_index=True)
    d["period"] = pd.PeriodIndex(d["period"], freq="M")
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    piv = d.pivot_table(index="period", columns="stateid", values="price")
    q = piv.groupby(piv.index.asfreq("Q")).mean()
    return q  # index: PeriodIndex[Q], cols: state


import glob as _glob
_CACHE = "grid_resilience/data/cache"
_PJM_STATES = {"PA", "OH", "IN", "MI", "VA", "NC", "KY", "MD", "NJ", "IL", "WV", "TN"}


def _wholesale_hub_quarterly() -> pd.DataFrame:
    """Quarterly mean DA LMP for ERCOT + PJM from the local cache, as PeriodIndex[Q]."""
    e = pd.concat([pd.read_parquet(f) for f in
                   sorted(_glob.glob(f"{_CACHE}/ercot_lmp_settlement point_*.parquet"))],
                  ignore_index=True)
    e = e[e["location"].astype(str).str.startswith("HB_")]
    e["time"] = pd.to_datetime(e["time"])
    erc = e.groupby(pd.PeriodIndex(e["time"], freq="Q"))["lmp"].mean()
    pp = []
    for f in sorted(_glob.glob(f"{_CACHE}/pjm_lmp_DAY_AHEAD_HOURLY_*.parquet")):
        d = pd.read_parquet(f)
        if d.empty:
            continue
        pp.append(d[d["location"].astype(str).str.upper().eq("WESTERN HUB")][["time", "lmp"]])
    p = pd.concat(pp, ignore_index=True)
    p["time"] = pd.to_datetime(p["time"], utc=True).dt.tz_localize(None)
    pjm = p.groupby(pd.PeriodIndex(p["time"], freq="Q"))["lmp"].mean()
    return pd.DataFrame({"ERCOT": erc, "PJM": pjm})


def company_power_price(qprice: pd.DataFrame, wholesale: bool = False) -> pd.DataFrame:
    """Weighted power price per company per quarter.
    wholesale=True: substitute ERCOT/PJM DA hub price for TX / PJM-state weights
    (the merchant-exposed names actually pay something closer to wholesale)."""
    hub = _wholesale_hub_quarterly() if wholesale else None
    out = {}
    for u in UNIVERSE:
        w = pd.Series(u["exposure"]); w = w / w.sum()
        parts = []
        for st, wt in w.items():
            if wholesale and st == "TX" and "ERCOT" in hub:
                s = hub["ERCOT"]
            elif wholesale and st in _PJM_STATES:
                s = hub["PJM"]
            elif st in qprice.columns:
                s = qprice[st]
            else:
                continue
            parts.append(s * wt)
        if parts:
            out[u["ticker"]] = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
    return pd.DataFrame(out)


def signal_panel(qprice: pd.DataFrame, wholesale: bool = False) -> pd.DataFrame:
    cpp = company_power_price(qprice, wholesale=wholesale)
    pct = cpp.pct_change()                          # QoQ % change in power price
    intens = pd.Series({u["ticker"]: u["intensity"] for u in UNIVERSE})
    raw = -(pct * intens)                           # higher cost -> more negative
    # cross-sectional z each quarter
    z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0)
    return z


def edgar_margins() -> pd.DataFrame:
    """Quarterly gross margin (primary G1 target — power cost sits in COGS) plus
    operating margin where tagged."""
    rows = []
    for u in UNIVERSE:
        gm = quarterly_gross_margin(u["cik"])
        oil = quarterly_flow(u["cik"], ["OperatingIncomeLoss"])
        if gm.empty or gm["gross_margin"].notna().sum() < 6:
            print(f"  [{u['ticker']}] EDGAR gross-margin miss (n={gm['gross_margin'].notna().sum() if not gm.empty else 0})")
            continue
        m = gm[["quarter_end", "revenue", "gross_margin"]].copy()
        if not oil.empty:
            m = m.merge(oil.rename(columns={"val": "oi"})[["quarter_end", "oi"]],
                        on="quarter_end", how="left")
            m["op_margin"] = m["oi"] / m["revenue"]
        else:
            m["op_margin"] = np.nan
        m["ticker"] = u["ticker"]
        rows.append(m[["ticker", "quarter_end", "gross_margin", "op_margin", "revenue"]])
    df = pd.concat(rows, ignore_index=True)
    df["q"] = df["quarter_end"].dt.to_period("Q")
    # clip absurd margins from tag mismatches
    for c in ("gross_margin", "op_margin"):
        df.loc[df[c].abs() > 2, c] = np.nan
    return df


def monthly_returns() -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(TCK, start="2016-01-01", end="2026-07-01", interval="1mo",
                     auto_adjust=True, progress=False)["Close"]
    return px.pct_change().iloc[1:]


def ff_factors() -> pd.DataFrame:
    def _load(url, name):
        z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=60).content))
        raw = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        rows = []
        for ln in raw:
            p = ln.split(",")
            if len(p) >= 2 and p[0].strip().isdigit() and len(p[0].strip()) == 6:
                rows.append([x.strip() for x in p])
        df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
        df = df.rename(columns={0: "ym"}).set_index("ym")
        df.index = pd.PeriodIndex(df.index.astype(int).astype(str), freq="M")
        return df
    f5 = _load("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
               "F-F_Research_Data_5_Factors_2x3_CSV.zip", "f5")
    f5.columns = ["MktRF", "SMB", "HML", "RMW", "CMA", "RF"][:f5.shape[1]]
    mom = _load("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                "F-F_Momentum_Factor_CSV.zip", "mom")
    mom.columns = ["MOM"][:mom.shape[1]]
    out = f5.join(mom, how="inner") / 100.0
    return out


# ----------------------------- stats -----------------------------
def _rank_ic(a, b):
    m = pd.concat([a, b], axis=1).dropna()
    if len(m) < 5:
        return np.nan
    return m.iloc[:, 0].rank().corr(m.iloc[:, 1].rank())


def fama_macbeth(ic_series: pd.Series):
    ic = ic_series.dropna()
    if len(ic) < 6:
        return np.nan, np.nan, len(ic)
    t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
    return ic.mean(), t, len(ic)


def main():
    wholesale = "--wholesale" in sys.argv
    print(f"=== PROBE B: electricity-cost exposure cross-section "
          f"({'WHOLESALE hub' if wholesale else 'state retail'} price) ===\n")
    qprice = eia_state_industrial_price()
    sig = signal_panel(qprice, wholesale=wholesale)  # index Q, cols ticker  (z-score)
    print(f"signal panel: {sig.shape[0]} quarters {sig.index.min()}..{sig.index.max()}, "
          f"{sig.notna().any().sum()} names with data")

    # ---------- G1: fundamentals ----------
    fm = edgar_margins()
    fm = fm.sort_values(["ticker", "q"])
    for base in ["gross_margin", "op_margin"]:
        fm[f"d_{base}_next"] = fm.groupby("ticker")[base].shift(-1) - fm[base]
        fm[f"d_{base}_now"] = fm[base] - fm.groupby("ticker")[base].shift(1)
    slong = sig.stack().rename("signal").reset_index()
    slong.columns = ["q", "ticker", "signal"]
    g1 = fm.merge(slong, on=["q", "ticker"], how="inner")
    print(f"G1 panel: {g1['ticker'].nunique()} names, {g1['q'].nunique()} quarters")
    for tgt in ["d_gross_margin_now", "d_gross_margin_next",
                "d_op_margin_now", "d_op_margin_next"]:
        ics = g1.groupby("q").apply(
            lambda d: _rank_ic(d["signal"], d[tgt]), include_groups=False)
        mean, t, n = fama_macbeth(ics)
        print(f"G1 {tgt:20s}: mean rank-IC {mean:+.3f}  FM t={t:+.2f}  ({n} quarters)")

    # ---------- returns panel ----------
    rets = monthly_returns()                          # monthly, index Timestamp
    rets.index = pd.PeriodIndex(rets.index, freq="M")
    # signal known ~1 quarter after quarter-end (10-Q lag): map quarter Q -> effective
    # from the 2nd month of the *next* quarter, then hold.
    sig_m = pd.DataFrame(index=rets.index, columns=sig.columns, dtype=float)
    for q in sig.index:
        eff_start = (q + 1).asfreq("M", "s") + 1     # ~2nd month after quarter end
        sig_m.loc[sig_m.index >= eff_start] = sig.loc[q].values
    sig_m = sig_m.reindex(rets.index)

    def ic_by_month(fwd):
        out = {}
        for m in rets.index:
            s = sig_m.loc[m]
            fut = rets.loc[(rets.index > m) & (rets.index <= m + fwd)]
            if fut.empty:
                continue
            rr = (1 + fut).prod() - 1                 # forward fwd-month return
            out[m] = _rank_ic(s, rr)
        return pd.Series(out)

    print()
    for fwd in (1, 3):
        ics = ic_by_month(fwd)
        mean, t, n = fama_macbeth(ics)
        print(f"G2 fwd-{fwd}m return : mean rank-IC {mean:+.3f}  FM t={t:+.2f}  ({n} months)")
        # G5 regime split
        for lo, hi, lab in [("2016", "2021", "2016-2021"), ("2022", "2026", "2022-2026")]:
            sub = ics[(ics.index >= lo) & (ics.index <= hi)]
            m2, t2, n2 = fama_macbeth(sub)
            print(f"    {lab}: mean rank-IC {m2:+.3f}  t={t2:+.2f}  ({n2})")

    # ---------- G3: factor + sector neutralization ----------
    ff = ff_factors()
    sec = {u["ticker"]: u["sector"] for u in UNIVERSE}
    long = rets.stack().rename("ret").reset_index()
    long.columns = ["ym", "ticker", "ret"]
    long["signal"] = [sig_m.loc[r.ym, r.ticker] if r.ticker in sig_m.columns else np.nan
                      for r in long.itertuples()]
    long = long.merge(ff.reset_index().rename(columns={"index": "ym"}), on="ym", how="left")
    long["sector"] = long["ticker"].map(sec)
    long = long.dropna(subset=["ret", "signal", "MktRF"])
    long["ret_ex"] = long["ret"] - long["RF"]

    import statsmodels.formula.api as smf
    md = smf.ols("ret_ex ~ signal + MktRF + SMB + HML + RMW + CMA + MOM + C(sector)",
                 data=long).fit(cov_type="cluster", cov_kwds={"groups": long["ym"]})
    print(f"\nG3 neutralized panel OLS (cluster by month, n={int(md.nobs)}):")
    print(f"    signal coef {md.params['signal']:+.5f}  t={md.tvalues['signal']:+.2f}  "
          f"p={md.pvalues['signal']:.3f}")

    print(f"\nG4 breadth: {sig.notna().any().sum()} names")
    print("\n(Interpret against the 6-gate tree; any |t|<2 on G1/G2/G3 -> FAIL.)")


if __name__ == "__main__":
    main()
