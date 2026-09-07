import pandas as pd, numpy as np, glob
RF=0.04; A=252; CAP=0.35
fs=sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))
p=pd.concat([pd.read_parquet(f) for f in fs]).sort_index(); p=p[~p.index.duplicated(keep='last')].astype(float)
extra=pd.read_parquet('research/capex_cycle_pair_classifier/_cache/der_extra_prices.parquet')
for c in extra.columns:
    if c not in p.columns or p[c].dropna().empty: p[c]=extra[c]
p=p.sort_index()

# (pureplay, customer_LONG?, profit_LONG?, duration_LONG?, policy_LONG?)   [ENPH profit fixed -> 1]
C={
 "ETN":(1,1,1,1,1),"HUBB":(1,1,1,1,1),"NVT":(1,1,1,1,1),"VRT":(1,1,1,1,1),"GEV":(1,1,1,1,1),
 "PWR":(1,1,1,1,1),"MYRG":(1,1,1,1,1),"PRIM":(1,1,1,1,1),"POWL":(1,1,1,1,1),"ATKR":(1,1,1,1,1),
 "WCC":(1,1,1,1,1),"ABBNY":(1,1,1,1,1),"SBGSY":(1,1,1,1,1),"PRYMY":(1,1,1,1,1),"HTHIY":(1,1,1,1,1),
 "NXT":(1,1,1,1,0),"FSLR":(1,1,1,1,0),
 "ENPH":(1,0,1,0,0),"SEDG":(1,0,0,0,0),"RUN":(1,0,0,0,0),"CHPT":(1,0,0,0,0),"BLNK":(1,0,0,0,0),
 "EVGO":(1,0,0,0,0),"WBX":(1,0,0,0,0),"STEM":(1,0,0,0,0),
 "FLNC":(1,1,0,0,0),"FTCI":(1,1,0,0,0),"ARRY":(1,1,0,1,0),"SHLS":(1,1,1,0,0),"GNRC":(1,0,1,1,0),
}
def sleeves(strict):
    L,S=[],[]
    for t,(pp,cu,pr,du,po) in C.items():
        lc=pr+du+po; sc=(1-pr)+(1-du)+(1-po)
        need=3 if strict else 2
        if pp and cu==1 and lc>=need: L.append(t)
        elif pp and cu==0 and sc>=need: S.append(t)
    return sorted(L),sorted(S)

def rets(cols,a,b):
    px=p.loc[a:b,[c for c in cols if c in p.columns]].ffill()
    return px.pct_change().clip(-CAP,CAP).mean(axis=1)
def stats(r):
    r=r.dropna()
    if len(r)<40: return {}
    cum=(1+r).prod()-1; vol=r.std()*np.sqrt(A); sh=(r.mean()*A-RF)/vol
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return dict(tot=f"{cum:+.0%}",vol=f"{vol:.0%}",sharpe=round(sh,2),maxdd=f"{dd:.0%}")
def bh_vt(L,S,tgt=.20,cap=1.5,bw=60,vw=20):
    d=pd.concat([L.rename('L'),S.rename('S')],axis=1).dropna()
    b=(d['L'].rolling(bw).cov(d['S'])/d['S'].rolling(bw).var()).clip(.5,2).shift(1).fillna(1)
    spr=d['L']-b*d['S']; rv=spr.rolling(vw).std()*np.sqrt(A)
    return ((tgt/rv).clip(0,cap).shift(1).fillna(0)*spr).dropna()

Ll,Sl=sleeves(False); Ls,Ss=sleeves(True)
print("LOOSE  long :",Ll)
print("LOOSE  short:",Sl)
print("STRICT long :",Ls)
print("STRICT short:",Ss)
print("  -> strict drops from long:",sorted(set(Ll)-set(Ls)),"| from short:",sorted(set(Sl)-set(Ss)))

for a,b,wl in [('2019-06-01','2022-12-31','pre-AI 19-22'),('2023-01-01','2026-08-31','primary 23-26'),('2019-06-01','2026-08-31','full')]:
    out={}
    for nm,Lc,Sc in [('LOOSE  rules L/S',Ll,Sl),('STRICT rules L/S',Ls,Ss)]:
        L=rets(Lc,a,b);S=rets(Sc,a,b)
        out[nm+'  DN']=stats(L-S); out[nm+'  BH+VT']=stats(bh_vt(L,S))
    print(f"\n==== {wl} ====")
    print(pd.DataFrame(out).T.to_string())
