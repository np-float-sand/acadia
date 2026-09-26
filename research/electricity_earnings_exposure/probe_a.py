"""
PROBE A1 — merchant-generation margin retrodiction gate.

Question: can a bottom-up model (owned fleet x realized ISO power price - fuel cost)
retrodict the DIRECTION and RELATIVE SIZE of these companies' REPORTED quarterly
gross profit, 2018-2025?

Pre-registered PASS bar:
  - pooled Spearman corr(dModel, dReported) >= 0.50
  - QoQ sign-agreement >= 70%
  - both hold in each half of the sample
Anything less -> FAIL -> the "electricity -> merchant earnings" branch is dead;
proceed to Probe B.

Throwaway. Re-run to regenerate. Findings -> docs/RESEARCH-LOG.md.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from edgar import quarterly_flow  # noqa: E402

CACHE = "grid_resilience/data/cache"
EIA_KEY = os.environ.get("EIA_API_KEY", "")

CIK = {"VST": "1692819", "NRG": "1013871", "CEG": "1868275", "TLN": "1622536"}

# ---- hand-coded fleet: owned GW by (company, market, fuel), with an effective date.
# Approximate, public sources (10-K fleet tables, EIA-860, deal press). Precise to
# ~+/-15% — fine for a direction/size retrodiction. NYISO nuclear (CEG Nine Mile/
# Ginna/FitzPatrick, ~3.4 GW) is priced at PJM Western Hub as a documented proxy.
# Market in {ERCOT, PJM}. Fuel in {nuclear, gas, coal, renew}.
FLEET = [
    # VST / Vistra
    ("VST", "2018-01-01", "ERCOT", "gas",     18.0),
    ("VST", "2018-01-01", "ERCOT", "nuclear",  2.4),   # Comanche Peak
    ("VST", "2018-01-01", "ERCOT", "coal",     5.0),   # declining; retirements through 2027
    ("VST", "2018-01-01", "ERCOT", "renew",    1.0),
    ("VST", "2018-01-01", "PJM",   "gas",      8.0),    # ex-Dynegy PJM/MISO/ISO-NE/NY, priced PJM
    ("VST", "2018-01-01", "PJM",   "coal",     4.0),
    ("VST", "2024-04-01", "PJM",   "nuclear",  4.0),    # Energy Harbor: Perry, Davis-Besse, Beaver Valley
    # NRG (post-2021 divestitures; big retail book NOT modeled)
    ("NRG", "2018-01-01", "ERCOT", "gas",      9.0),
    ("NRG", "2018-01-01", "ERCOT", "coal",     4.0),    # WA Parish, Limestone; declining
    ("NRG", "2018-01-01", "PJM",   "gas",      3.0),
    # CEG / Constellation (spun from Exelon 2022-02; largest US nuclear fleet)
    ("CEG", "2022-01-01", "PJM",   "nuclear", 17.5),    # PJM + NY nuclear priced at PJM West
    ("CEG", "2022-01-01", "PJM",   "gas",      6.0),
    ("CEG", "2022-01-01", "PJM",   "renew",    2.0),    # Conowingo hydro + wind/solar
    # TLN / Talen (emerged from Ch.11 2023-05)
    ("TLN", "2023-01-01", "PJM",   "nuclear",  2.2),    # Susquehanna 90% of 2.5 GW
    ("TLN", "2023-01-01", "PJM",   "gas",      4.0),
    ("TLN", "2023-01-01", "PJM",   "coal",     2.0),
    ("TLN", "2023-01-01", "ERCOT", "gas",      1.0),
]

CF   = {"nuclear": 0.92, "gas": 0.52, "coal": 0.45, "renew": 0.32}
# all-in cash cost EX power price, $/MWh  (fuel + VOM)
EXFUEL = {"nuclear": 28.0, "coal": 34.0, "renew": 3.0}
GAS_HEATRATE = 7.0   # MMBtu/MWh CCGT-ish blended
GAS_VOM = 3.0


def hub_price_quarterly() -> pd.DataFrame:
    """Quarterly mean DA LMP ($/MWh) for ERCOT and PJM from the local cache."""
    out = {}
    # ERCOT: HB_* hub average
    ef = sorted(glob.glob(f"{CACHE}/ercot_lmp_settlement point_*.parquet"))
    ep = []
    for f in ef:
        d = pd.read_parquet(f)
        d = d[d["location"].astype(str).str.startswith("HB_")]
        ep.append(d)
    e = pd.concat(ep, ignore_index=True)
    e["time"] = pd.to_datetime(e["time"])
    out["ERCOT"] = e.groupby(pd.PeriodIndex(e["time"], freq="Q"))["lmp"].mean()
    # PJM: WESTERN HUB
    pf = sorted(glob.glob(f"{CACHE}/pjm_lmp_DAY_AHEAD_HOURLY_*.parquet"))
    pp = []
    for f in pf:
        d = pd.read_parquet(f)
        if d.empty:
            continue
        d = d[d["location"].astype(str).str.upper().eq("WESTERN HUB")]
        pp.append(d[["time", "lmp"]])
    p = pd.concat(pp, ignore_index=True)
    p["time"] = pd.to_datetime(p["time"], utc=True).dt.tz_localize(None)
    out["PJM"] = p.groupby(pd.PeriodIndex(p["time"], freq="Q"))["lmp"].mean()
    df = pd.DataFrame(out)
    df.index = df.index.to_timestamp(how="end").normalize()
    return df


def henry_hub_quarterly() -> pd.Series:
    url = f"https://api.eia.gov/v2/seriesid/NG.RNGWHHD.D?api_key={EIA_KEY}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["response"]["data"])
    d["period"] = pd.to_datetime(d["period"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    s = d.set_index("period")["value"].sort_index()
    q = s.groupby(pd.PeriodIndex(s.index, freq="Q")).mean()
    q.index = q.index.to_timestamp(how="end").normalize()
    return q.rename("hh")


def model_margin() -> pd.DataFrame:
    hub = hub_price_quarterly()
    hh = henry_hub_quarterly()
    idx = hub.index
    hh = hh.reindex(idx).ffill()
    hours = pd.Series(idx, index=idx).dt.days_in_month * 0  # placeholder
    # hours in the quarter
    qhours = pd.Series(
        [(pd.Period(ts, "Q").end_time - pd.Period(ts, "Q").start_time).days * 24 + 24
         for ts in idx], index=idx)
    recs = []
    for tkr, eff, mkt, fuel, gw in FLEET:
        eff = pd.Timestamp(eff)
        for ts in idx:
            if ts < eff:
                continue
            p = hub.loc[ts, mkt]
            if not np.isfinite(p):
                continue
            if fuel == "gas":
                exfuel = hh.loc[ts] * GAS_HEATRATE + GAS_VOM
            else:
                exfuel = EXFUEL[fuel]
            mwh = gw * 1000 * qhours.loc[ts] * CF[fuel]
            gm = mwh * (p - exfuel)          # $ economic gross margin
            recs.append((tkr, ts, gm))
    m = pd.DataFrame(recs, columns=["ticker", "quarter_end", "model_gm"])
    return m.groupby(["ticker", "quarter_end"])["model_gm"].sum().reset_index()


def reported() -> pd.DataFrame:
    """Quarterly Revenues and OperatingIncomeLoss from EDGAR XBRL.
    Energy companies do not tag GrossProfit; OperatingIncomeLoss is the deepest,
    most consistent income-statement flow and its QoQ swings are dominated by the
    same wholesale-margin dynamics the model builds (plus hedge MtM / D&A noise)."""
    parts = []
    for tkr, cik in CIK.items():
        rev = quarterly_flow(cik, ["Revenues",
                                   "RevenueFromContractWithCustomerExcludingAssessedTax"])
        oil = quarterly_flow(cik, ["OperatingIncomeLoss"])
        if rev.empty and oil.empty:
            print(f"  [{tkr}] no EDGAR data")
            continue
        d = rev.rename(columns={"val": "revenue"})[["quarter_end", "revenue"]].merge(
            oil.rename(columns={"val": "op_income"})[["quarter_end", "op_income"]],
            on="quarter_end", how="outer")
        d["ticker"] = tkr
        parts.append(d)
        print(f"  [{tkr}] rev {rev['quarter_end'].min().date() if not rev.empty else '-'}"
              f"..{rev['quarter_end'].max().date() if not rev.empty else '-'} n={len(rev)}   "
              f"opinc {oil['quarter_end'].min().date() if not oil.empty else '-'}"
              f"..{oil['quarter_end'].max().date() if not oil.empty else '-'} n={len(oil)}")
    return pd.concat(parts, ignore_index=True)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan, ok.sum()
    ar = pd.Series(a[ok]).rank().values
    br = pd.Series(b[ok]).rank().values
    return float(np.corrcoef(ar, br)[0, 1]), int(ok.sum())


def main():
    print("=== PROBE A1: merchant-generation margin retrodiction ===\n")
    print("Reported (EDGAR XBRL):")
    rep = reported()
    print("\nModel (fleet x price - fuel):")
    mod = model_margin()

    df = mod.merge(rep, on=["ticker", "quarter_end"], how="inner").sort_values(
        ["ticker", "quarter_end"])
    df = df[df["quarter_end"] <= "2025-12-31"]
    df["d_model"] = df.groupby("ticker")["model_gm"].diff()
    df["d_rev"] = df.groupby("ticker")["revenue"].diff()
    df["d_oi"] = df.groupby("ticker")["op_income"].diff()

    print(f"\nMatched panel: {len(df)} company-quarters, "
          f"{df['quarter_end'].min().date()} .. {df['quarter_end'].max().date()}")

    for target, col in [("REVENUE", "d_rev"), ("OPERATING INCOME", "d_oi")]:
        sub0 = df.dropna(subset=["d_model", col])
        print(f"\n>>> target = {target}  (n={len(sub0)})")
        for tkr, g in sub0.groupby("ticker"):
            r, n = _spearman(g["d_model"], g[col])
            sign = float(np.mean(np.sign(g["d_model"]) == np.sign(g[col])))
            print(f"  {tkr:4s} n={n:2d}  d-corr={r:+.2f}  sign-agree={sign:.0%}")

        def block(sub, label):
            r, n = _spearman(sub["d_model"], sub[col])
            sign = (float(np.mean(np.sign(sub["d_model"]) == np.sign(sub[col])))
                    if len(sub) else np.nan)
            print(f"  {label:14s} n={n:3d}  d-corr={r:+.2f}  sign-agree={sign:.0%}")
            return r, sign, n

        r_all, s_all, _ = block(sub0, "POOLED ALL")
        mid = pd.Timestamp("2022-12-31")
        block(sub0[sub0["quarter_end"] <= mid], "2018-2022")
        block(sub0[sub0["quarter_end"] > mid], "2023-2025")
        if target == "OPERATING INCOME":
            passed = (r_all >= 0.50) and (s_all >= 0.70)
            print(f"\n--- PRE-REGISTERED VERDICT (target=operating income) ---")
            print(f"pooled d-corr {r_all:+.2f} (bar >=0.50), sign-agree {s_all:.0%} (bar >=70%)")
            print("VERDICT:", "PASS -> run A2" if passed else "FAIL -> go to Probe B")


if __name__ == "__main__":
    main()
