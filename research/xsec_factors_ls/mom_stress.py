"""Stress-test cross-sectional 12-1 momentum on the wide electrification universe:
beta-neutrality, IS/OOS within 2023-26, turnover/cost, full history, + layer-1 overlay."""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "/Users/sandhyapersad/acadia")
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket import config

UNIV = sorted(set("""
ETN HUBB GEV VRT NVT POWL ATKR AYI ENS AZZ ROK EMR AME ITT PH DOV BMI
PWR MYRG PRIM MTZ EME IESC FIX DY GVA STRL ROAD ACM TPC
FLNC STEM SHLS ARRY NXT FSLR ENPH SEDG
NX BDC HAYW GNRC WCC CNM
""".split()))
ANN = 12
px = fetch_prices(UNIV, "2015-06-01", "2026-08-31")
px = px[[c for c in UNIV if c in px.columns]].sort_index()
px = px[px.columns[px.notna().sum() > 350]].ffill(limit=5)
mpx = px.resample("ME").last(); mret = mpx.pct_change()
spy = fetch_prices(["SPY"], "2015-06-01", "2026-08-31")["SPY"].resample("ME").last().pct_change()

mom = np.log(mpx).diff(12) - np.log(mpx).diff(1)
ewret = mret.mean(1)

def terc_ls(score, q=3, cost_bps=0.0):
    fwd = mret.shift(-1); ls = []; prev_top = prev_bot = set()
    tno = []
    for dt in score.index:
        row = score.loc[dt].dropna()
        if len(row) < 12: ls.append(np.nan); tno.append(np.nan); continue
        k = max(3, len(row) // q)
        top = set(row.nlargest(k).index); bot = set(row.nsmallest(k).index)
        f = fwd.loc[dt]
        turn = (len(top ^ prev_top) + len(bot ^ prev_bot)) / (2 * k)
        gross = f[list(top)].mean() - f[list(bot)].mean()
        ls.append(gross - cost_bps / 1e4 * turn * 2)
        tno.append(turn); prev_top, prev_bot = top, bot
    return pd.Series(ls, index=score.index), np.nanmean(tno)

def st(r):
    r = r.dropna()
    if len(r) < 5: return (np.nan,)*3
    cg = (1+r).prod()**(ANN/len(r))-1; sh = r.mean()/r.std()*np.sqrt(ANN)
    dd = ((1+r).cumprod()/(1+r).cumprod().cummax()-1).min()
    return cg, sh, dd

WINS = {"2017-18":("2017-01","2018-12"), "2019-22":("2019-01","2022-12"),
        "2023-24":("2023-01","2024-12"), "2025-26":("2025-01","2026-08"),
        "2023-26":("2023-01","2026-08"), "full":("2017-01","2026-08")}

ls3, tno = terc_ls(mom, 3)
ls5, _ = terc_ls(mom, 5)
ls3c, _ = terc_ls(mom, 3, cost_bps=20)

print("="*82)
print(f"12-1 MOMENTUM L/S on {mpx.shape[1]}-name electrification universe  (avg 1-way turnover {tno:.2f})")
print(f"  {'window':10}{'tercile L/S':>14}{'quintile L/S':>14}{'terc L/S -20bp':>16}{'EW univ':>10}")
for wl,(a,b) in WINS.items():
    r=lambda s: st(s.loc[a:b])[1]
    print(f"  {wl:10}{r(ls3):>14.2f}{r(ls5):>14.2f}{r(ls3c):>16.2f}{st(ewret.loc[a:b])[1]:>10.2f}")

# beta-neutrality: regress tercile L/S on EW-universe and SPY
print("\nBETA CHECK -- regress tercile L/S monthly ret on EW-universe + SPY:")
for wl,(a,b) in [("2019-22",WINS["2019-22"]),("2023-26",WINS["2023-26"])]:
    d = pd.concat([ls3.rename("ls"), ewret.rename("ew"), spy.rename("spy")], axis=1).loc[a:b].dropna()
    X = np.column_stack([np.ones(len(d)), d["ew"], d["spy"]])
    bta,*_ = np.linalg.lstsq(X, d["ls"].values, rcond=None)
    res = d["ls"].values - X@bta
    se = np.sqrt((res@res)/(len(d)-3)*np.linalg.inv(X.T@X).diagonal())
    ir = res.mean()/res.std()*np.sqrt(ANN)
    print(f"  {wl}: alpha {bta[0]*100:+.2f}%/mo (t={bta[0]/se[0]:+.2f})  beta_ew {bta[1]:+.2f}  beta_spy {bta[2]:+.2f}"
          f"  resid IR {ir:+.2f}  n={len(d)}")

# long-only tilt (top tercile overweight) + layer-1 style trend gate on the tilt
fwd = mret.shift(-1)
def long_tilt(score):
    out=[]
    for dt in score.index:
        row=score.loc[dt].dropna()
        if len(row)<12: out.append(np.nan); continue
        k=max(3,len(row)//3); n=len(row)
        w=pd.Series(1.0/n,index=row.index); w[row.nlargest(k).index]*=1.75; w[row.nsmallest(k).index]*=0.35
        w/=w.sum(); out.append((w*fwd.loc[dt].reindex(w.index)).sum())
    return pd.Series(out,index=score.index)
lt = long_tilt(mom)
# trend gate: hold tilt only if its own 6m MA rising, else EW
ltl = (1+lt).cumprod(); gate = (ltl > ltl.rolling(4).mean()).shift(1).fillna(True)
lt_gated = lt.where(gate, ewret)
print("\nLONG-ONLY MOMENTUM TILT (top terc x1.75 / bot x0.35):")
print(f"  {'window':10}{'tilt Sh':>10}{'tilt+gate Sh':>14}{'EW univ Sh':>12}{'marquee-9 Sh':>14}")
m9 = mret[[c for c in sorted(config.UNIVERSE) if c in mret.columns]].mean(1)
for wl,(a,b) in WINS.items():
    print(f"  {wl:10}{st(lt.loc[a:b])[1]:>10.2f}{st(lt_gated.loc[a:b])[1]:>14.2f}"
          f"{st(ewret.loc[a:b])[1]:>12.2f}{st(m9.loc[a:b])[1]:>14.2f}")

# is momentum just long-equipment / short-contractor? tag names
EQUIP=set("ETN HUBB GEV VRT NVT POWL ATKR AYI ENS AZZ ROK EMR AME ITT PH DOV BMI NX BDC HAYW GNRC WCC CNM".split())
EPC=set("PWR MYRG PRIM MTZ EME IESC FIX DY GVA STRL ROAD ACM TPC".split())
DER=set("FLNC STEM SHLS ARRY NXT FSLR ENPH SEDG".split())
print("\nGROUP TILT of the momentum long-tercile over 2023-26 (avg # names from each group in top tercile):")
cnt={"EQUIP":[],"EPC":[],"DER":[]}
for dt in mom.loc["2023-01":"2026-08"].index:
    row=mom.loc[dt].dropna()
    if len(row)<12: continue
    k=max(3,len(row)//3); top=set(row.nlargest(k).index)
    for g,names in [("EQUIP",EQUIP),("EPC",EPC),("DER",DER)]:
        cnt[g].append(len(top & names))
for g in cnt: print(f"  {g}: {np.mean(cnt[g]):.1f}  (universe has {len({'EQUIP':EQUIP,'EPC':EPC,'DER':DER}[g] & set(mom.columns))})")
