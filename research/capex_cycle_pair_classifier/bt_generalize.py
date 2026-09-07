import pandas as pd, numpy as np, glob
RF=0.04; A=252; CAP=0.35

g=pd.read_parquet('research/capex_cycle_pair_classifier/_cache/gen_prices.parquet')
grid=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))]).sort_index()
grid=grid[~grid.index.duplicated(keep='last')].astype(float)
gext=pd.read_parquet('research/capex_cycle_pair_classifier/_cache/der_extra_prices.parquet')
p=g.copy()
for src in (grid,gext):
    for c in src.columns:
        if c not in p.columns: p[c]=src[c]
p=p.sort_index().astype(float)

# classification per theme: ticker -> (pureplay, customer_LONG?, profit_LONG?, duration_LONG?, policy_LONG?)
THEMES={
 "GRID / DER":{
   "ETN":(1,1,1,1,1),"HUBB":(1,1,1,1,1),"NVT":(1,1,1,1,1),"VRT":(1,1,1,1,1),"GEV":(1,1,1,1,1),
   "PWR":(1,1,1,1,1),"MYRG":(1,1,1,1,1),"PRIM":(1,1,1,1,1),"POWL":(1,1,1,1,1),"ATKR":(1,1,1,1,1),
   "WCC":(1,1,1,1,1),"ABBNY":(1,1,1,1,1),"SBGSY":(1,1,1,1,1),"PRYMY":(1,1,1,1,1),"HTHIY":(1,1,1,1,1),
   "NXT":(1,1,1,1,0),"FSLR":(1,1,1,1,0),
   "ENPH":(1,0,1,0,0),"SEDG":(1,0,0,0,0),"RUN":(1,0,0,0,0),"CHPT":(1,0,0,0,0),"BLNK":(1,0,0,0,0),
   "EVGO":(1,0,0,0,0),"WBX":(1,0,0,0,0),"STEM":(1,0,0,0,0),
 },
 "EV SUPPLY CHAIN":{
   "BWA":(1,1,1,1,1),"APTV":(1,1,1,1,1),"VC":(1,1,1,1,1),"LEA":(1,1,1,1,1),"ST":(1,1,1,1,1),
   "ALSN":(1,1,1,1,1),"MGA":(1,1,1,1,1),"DAN":(1,1,1,1,1),
   "RIVN":(1,0,0,0,0),"LCID":(1,0,0,0,0),"WKHS":(1,0,0,0,0),"PSNY":(1,0,0,0,0),"NIO":(1,0,0,0,0),
   "XPEV":(1,0,0,0,0),"LI":(1,0,1,0,0),
 },
 "NUCLEAR / SMR":{
   "BWXT":(1,1,1,1,1),"CCJ":(1,1,1,1,1),"LEU":(1,1,1,1,0),
   "SMR":(1,0,0,0,0),"OKLO":(1,0,0,0,0),"NNE":(1,0,0,0,0),"LTBR":(1,0,0,0,0),"ASPI":(1,0,0,0,0),
 },
 "SPACE":{
   "IRDM":(1,1,1,1,1),"RKLB":(1,1,0,0,1),
   "ASTS":(1,0,0,0,0),"PL":(1,0,0,0,0),"BKSY":(1,0,0,0,0),"RDW":(1,0,0,0,0),"SPCE":(1,0,0,0,0),"GSAT":(1,0,0,0,0),
 },
}
def sleeves(cls,need=2):
    L,S=[],[]
    for t,(pp,cu,pr,du,po) in cls.items():
        lc=pr+du+po; sc=3-lc
        if pp and cu==1 and lc>=need: L.append(t)
        elif pp and cu==0 and sc>=need: S.append(t)
    return sorted(L),sorted(S)
def rets(cols,a,b):
    have=[c for c in cols if c in p.columns and p[c].loc[a:b].notna().sum()>20]
    px=p.loc[a:b,have].ffill()
    return px.pct_change().clip(-CAP,CAP).mean(axis=1), have
def st(r):
    r=r.dropna()
    if len(r)<60: return {"n":len(r)}
    cum=(1+r).prod()-1; vol=r.std()*np.sqrt(A); sh=(r.mean()*A-RF)/vol
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return dict(n=len(r),tot=f"{cum:+.0%}",vol=f"{vol:.0%}",sharpe=round(sh,2),maxdd=f"{dd:.0%}")
def bhvt(L,S,tgt=.20,cap=1.5,bw=60,vw=20):
    d=pd.concat([L.rename('L'),S.rename('S')],axis=1).dropna()
    if len(d)<80: return pd.Series(dtype=float)
    b=(d['L'].rolling(bw).cov(d['S'])/d['S'].rolling(bw).var()).clip(.5,2).shift(1).fillna(1)
    spr=d['L']-b*d['S']; rv=spr.rolling(vw).std()*np.sqrt(A)
    return ((tgt/rv).clip(0,cap).shift(1).fillna(0)*spr).dropna()

WIN={"GRID / DER":[('2019-06-01','2022-12-31','pre-AI'),('2023-01-01','2026-08-31','post/normalise'),('2019-06-01','2026-08-31','full')],
     "EV SUPPLY CHAIN":[('2020-06-01','2021-12-31','mania'),('2022-01-01','2026-08-31','post/bust'),('2020-06-01','2026-08-31','full')],
     "NUCLEAR / SMR":[('2022-03-01','2023-12-31','pre-rally'),('2024-01-01','2026-08-31','AI-power rally'),('2022-03-01','2026-08-31','full')],
     "SPACE":[('2020-11-01','2021-12-31','mania'),('2022-01-01','2026-08-31','post/bust'),('2020-11-01','2026-08-31','full')]}

for th,cls in THEMES.items():
    Lc,Sc=sleeves(cls)
    print(f"\n########## {th} ##########")
    print(f"  LONG ({len(Lc)}): {Lc}")
    print(f"  SHORT({len(Sc)}): {Sc}")
    if len(Lc)<3 or len(Sc)<3:
        print(f"  --> RULE SELF-REJECTS: sleeve too thin (min 3/side).")
        continue
    for a,b,wl in WIN[th]:
        L,lh=rets(Lc,a,b); S,sh=rets(Sc,a,b)
        dn=st(L-S); bv=st(bhvt(L,S))
        print(f"  {wl:16s} | DN {dn.get('sharpe','?')!s:>6} (dd {dn.get('maxdd','?')})   BH+VT {bv.get('sharpe','?')!s:>6} (dd {bv.get('maxdd','?')})  [L={len(lh)} S={len(sh)} n={dn.get('n')}]")
