import pandas as pd, numpy as np, warnings
warnings.simplefilter("ignore")
RF=0.04;A=252
P=pd.read_parquet('research/capex_cycle_pair_classifier/_cache/policy_prices.parquet').astype(float)
LONG={
 "AI/DC buildout":['VRT','ANET','CIEN','APH','FIX','EME','IESC','CARR'],
 "Semicap":['AMAT','LRCX','KLAC','ONTO','ACLS','KLIC','AEIS','ENTG'],
 "Aero/defense supply":['HEI','TDG','HWM','CW','WWD','TXT','HXL'],
 "Reshoring E&C/automation":['J','STRL','PWR','ROK','EMR','NDSN','PH','DOV','AME'],
 "Water infra":['XYL','BMI','MWA','FELE','PNR','AOS','ROP'],
 "Grid equip":['SIEGY','ETN','HUBB','POWL','NVT','WCC','PRYMY'],
 "Electrification/thermal":['CR','SPXC','FLS','ITT','GGG'],
}
ALL_LONG=sorted({t for v in LONG.values() for t in v})
NONPOL=['MMM','ITW','GWW','FAST','ODFL','JBHT','EXPD','CHRW','PCAR','GPC','SNA','SWK','WSO']

def ew(cols,a,b):
    h=[c for c in cols if c in P.columns and P[c].loc[a:b].notna().sum()>20]
    return P.loc[a:b,h].ffill().pct_change().clip(-.35,.35).mean(axis=1)
def stat(r):
    r=r.dropna()
    if len(r)<50: return (np.nan,np.nan,np.nan)
    v=r.std()*np.sqrt(A); s=(r.mean()*A-RF)/v
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return round(s,2),f"{(1+r).prod()-1:+.0%}",f"{dd:.0%}"
def bhvt(L,S,tgt=.15,cap=1.5,bw=60,vw=20):
    d=pd.concat([L.rename('L'),S.rename('S')],axis=1).dropna()
    b=(d['L'].rolling(bw).cov(d['S'])/d['S'].rolling(bw).var()).clip(.5,2).shift(1).fillna(1)
    spr=d['L']-b*d['S']; rv=spr.rolling(vw).std()*np.sqrt(A)
    return ((tgt/rv).clip(0,cap).shift(1).fillna(0)*spr).dropna()

W=[('2023-01-01','2026-08-31','2023-26'),('2021-01-01','2022-12-31','2021-22'),('2019-06-01','2026-08-31','full')]

print("############ per-bucket EW long, Sharpe / total / maxDD ############")
for a,b,wl in W:
    print(f"\n-- {wl} --")
    for th,lst in LONG.items():
        s,t,dd=stat(ew(lst,a,b)); print(f"  {th:26s} {s!s:>6}  {t:>7}  {dd}")
    s,t,dd=stat(ew(ALL_LONG,a,b)); print(f"  {'>>> ALL policy-long (46)':26s} {s!s:>6}  {t:>7}  {dd}")
    s,t,dd=stat(ew(NONPOL,a,b)); print(f"  {'non-policy industrials (13)':26s} {s!s:>6}  {t:>7}  {dd}")
    s,t,dd=stat(P['XLI'].loc[a:b].pct_change()); print(f"  {'XLI':26s} {s!s:>6}  {t:>7}  {dd}")
    s,t,dd=stat(P['PAVE'].loc[a:b].pct_change()); print(f"  {'PAVE':26s} {s!s:>6}  {t:>7}  {dd}")

print("\n############ LONG/SHORT: policy-long vs non-policy short (beta-hedge + 15% vol) ############")
for a,b,wl in W:
    L=ew(ALL_LONG,a,b)
    for shname,shc in [('XLI',None),('non-policy basket',NONPOL)]:
        S=P['XLI'].loc[a:b].pct_change() if shc is None else ew(shc,a,b)
        s,t,dd=stat(bhvt(L,S))
        print(f"  {wl:8s} policy-long / {shname:18s}  Sharpe {s!s:>6}  tot {t:>7}  maxDD {dd}")
    # also raw long-only excess vs the short (is there a tilt at all)
    exc=(L - ew(NONPOL,a,b)).dropna()
    s,_,_=stat(exc)
    print(f"           [raw daily (policy-long minus non-policy), Sharpe {s}]")

print("\n############ per-name Sharpe 2023-26 (long candidates) ############")
rows=[]
for t in ALL_LONG:
    s,tt,dd=stat(P[t].loc['2023-01-01':'2026-08-31'].pct_change().clip(-.35,.35))
    rows.append((t,s,tt))
for t,s,tt in sorted(rows,key=lambda x:-(x[1] if pd.notna(x[1]) else -9)):
    print(f"  {t:6s} {s!s:>6}  {tt}")
