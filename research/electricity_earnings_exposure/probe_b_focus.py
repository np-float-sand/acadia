"""
Confirmatory test of the single surviving cell from probe_b:
  2022-2026, fwd-3m return rank-IC = +0.118, Fama-MacBeth t = 1.96  (state-retail price)

Pre-specified: restrict to 2022-01..2026-06, DROP the BTC miners (BTC-driven equity
noise), sector-neutralise, use Newey-West t (3-mo overlap), and look at an actual
tertile long/short portfolio. Both price proxies.

Bar to be "worth a proper look": |t| >= 2 on ALL of
  (a) NW-adjusted fwd-3m rank-IC
  (b) sector-neutral NW rank-IC
  (c) FF5+MOM+sector panel OLS signal coef (2022-26 only)
Anything less -> the 1.96 was noise.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from universe import UNIVERSE
import probe_b as pb

CRYPTO = {"MARA", "RIOT", "CLSK", "WULF", "CIFR", "IREN", "BTDR"}
KEEP = [u["ticker"] for u in UNIVERSE if u["ticker"] not in CRYPTO]
SECT = {u["ticker"]: u["sector"] for u in UNIVERSE}
LO, HI = pd.Period("2022-01", "M"), pd.Period("2026-06", "M")


def nw_t(x: pd.Series, lag: int = 3) -> float:
    x = x.dropna().values
    n = len(x)
    if n < 8:
        return np.nan
    xc = x - x.mean()
    g0 = (xc @ xc) / n
    s = g0
    for k in range(1, lag + 1):
        gk = (xc[k:] @ xc[:-k]) / n
        s += 2 * (1 - k / (lag + 1)) * gk
    se = np.sqrt(s / n)
    return x.mean() / se


def run(wholesale: bool):
    tag = "WHOLESALE hub" if wholesale else "state retail"
    print(f"\n================  price = {tag}  ================")
    qprice = pb.eia_state_industrial_price()
    sig = pb.signal_panel(qprice, wholesale=wholesale)[KEEP]

    rets = pb.monthly_returns()[KEEP]
    rets.index = pd.PeriodIndex(rets.index, freq="M")
    sig_m = pd.DataFrame(index=rets.index, columns=sig.columns, dtype=float)
    for q in sig.index:
        eff = (q + 1).asfreq("M", "s") + 1
        sig_m.loc[sig_m.index >= eff] = sig.loc[q].values
    sig_m = sig_m.reindex(rets.index)

    months = [m for m in rets.index if LO <= m <= HI]
    raw_ic, sn_ic = {}, {}
    for m in months:
        s = sig_m.loc[m]
        fut = rets.loc[(rets.index > m) & (rets.index <= m + 3)]
        if fut.empty or s.notna().sum() < 6:
            continue
        r = (1 + fut).prod() - 1
        d = pd.concat([s.rename("s"), r.rename("r")], axis=1).dropna()
        d["sec"] = d.index.map(SECT)
        raw_ic[m] = d["s"].rank().corr(d["r"].rank())
        # sector-neutral: demean within sector
        d["sn_s"] = d["s"] - d.groupby("sec")["s"].transform("mean")
        d["sn_r"] = d["r"] - d.groupby("sec")["r"].transform("mean")
        d2 = d.dropna(subset=["sn_s", "sn_r"])
        if len(d2) >= 6:
            sn_ic[m] = d2["sn_s"].rank().corr(d2["sn_r"].rank())
    raw_ic, sn_ic = pd.Series(raw_ic), pd.Series(sn_ic)
    print(f"(a) fwd-3m rank-IC      mean {raw_ic.mean():+.3f}  plain t {raw_ic.mean()/(raw_ic.std()/np.sqrt(len(raw_ic))):+.2f}  NW-t {nw_t(raw_ic):+.2f}  (n={len(raw_ic)})")
    print(f"(b) sector-neutral IC   mean {sn_ic.mean():+.3f}  NW-t {nw_t(sn_ic):+.2f}  (n={len(sn_ic)})")

    # (c) FF5+MOM+sector panel OLS, 2022-26 only
    ff = pb.ff_factors()
    long = rets.stack().rename("ret").reset_index()
    long.columns = ["ym", "ticker", "ret"]
    long = long[(long.ym >= LO) & (long.ym <= HI)]
    long["signal"] = [sig_m.loc[r.ym, r.ticker] for r in long.itertuples()]
    long = long.merge(ff.reset_index().rename(columns={"index": "ym"}), on="ym", how="left")
    long["sector"] = long["ticker"].map(SECT)
    long = long.dropna(subset=["ret", "signal", "MktRF"])
    long["ret_ex"] = long["ret"] - long["RF"]
    import statsmodels.formula.api as smf
    md = smf.ols("ret_ex ~ signal + MktRF+SMB+HML+RMW+CMA+MOM + C(sector)",
                 data=long).fit(cov_type="cluster", cov_kwds={"groups": long["ym"]})
    print(f"(c) panel OLS signal    coef {md.params['signal']:+.5f}  t {md.tvalues['signal']:+.2f}  p {md.pvalues['signal']:.3f}  (n={int(md.nobs)})")

    # (d) tertile L/S, monthly formed, 3-mo hold (avg of 3 overlapping sleeves)
    pnl = {}
    for m in months:
        s = sig_m.loc[m].dropna()
        if len(s) < 6:
            continue
        hi_ = s[s >= s.quantile(2/3)].index
        lo_ = s[s <= s.quantile(1/3)].index
        nxt = m + 1
        if nxt not in rets.index:
            continue
        pnl[nxt] = rets.loc[nxt, hi_].mean() - rets.loc[nxt, lo_].mean()
    pnl = pd.Series(pnl)
    if len(pnl) > 6:
        ann = (1 + pnl.mean())**12 - 1
        shp = pnl.mean() / pnl.std() * np.sqrt(12)
        print(f"(d) tertile L/S 1-mo    ann {ann:+.1%}  Sharpe {shp:+.2f}  t {pnl.mean()/(pnl.std()/np.sqrt(len(pnl))):+.2f}  (n={len(pnl)})")

    a = nw_t(raw_ic); b = nw_t(sn_ic); c = md.tvalues["signal"]
    verdict = (abs(a) >= 2) and (abs(b) >= 2) and (abs(c) >= 2)
    print(f"--> {'WORTH A LOOK' if verdict else 'NOISE — dead'}  (need |t|>=2 on a,b,c; got {a:+.2f}, {b:+.2f}, {c:+.2f})")


if __name__ == "__main__":
    print("=== CONFIRMATORY TEST — 2022-2026, ex-crypto, sector-neutral, NW-t ===")
    print(f"universe: {KEEP}")
    for w in (False, True):
        run(w)
