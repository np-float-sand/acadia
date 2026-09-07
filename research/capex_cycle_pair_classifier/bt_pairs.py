import pandas as pd, numpy as np, glob
RF=0.04; A=252
fs=sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))
p=pd.concat([pd.read_parquet(f) for f in fs]).sort_index(); p=p[~p.index.duplicated(keep='last')].astype(float)
extra=pd.read_parquet('research/capex_cycle_pair_classifier/_cache/der_extra_prices.parquet')
for c in extra.columns:
    if c not in p.columns or p[c].dropna().empty: p[c]=extra[c]
p=p.sort_index()

HP_L=['ETN','HUBB','GEV','VRT','PWR','MYRG','NVT','FLNC','PRIM']
HP_S=['ENPH','SEDG','CHPT','RUN','BLNK','STEM']
R_L =['ETN','HUBB','NVT','VRT','GEV','PWR','MYRG','PRIM','POWL','ATKR','WCC','ABBNY','SBGSY','PRYMY','HTHIY','NXT','FSLR']
R_S =['BLNK','CHPT','ENPH','EVGO','RUN','SEDG','STEM','WBX']

def ew(cols,a,b):
    px=p.loc[a:b,[c for c in cols if c in p.columns]].ffill()
    return px.pct_change().mean(axis=1)

def stats(r):
    r=r.dropna()
    if len(r)<40: return dict(n=len(r))
    cum=(1+r).prod()-1; yrs=len(r)/A
    cagr=(1+cum)**(1/yrs)-1 if cum>-1 else np.nan
    vol=r.std()*np.sqrt(A); sh=(r.mean()*A-RF)/vol
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return dict(n=len(r),tot=f"{cum:+.0%}",cagr=f"{cagr:+.1%}",vol=f"{vol:.0%}",sharpe=round(sh,2),maxdd=f"{dd:.0%}")

def betahedge_voltgt(L,S,tgt=0.20,cap=1.5,bwin=60,vwin=20):
    df=pd.concat([L.rename('L'),S.rename('S')],axis=1).dropna()
    beta=(df['L'].rolling(bwin).cov(df['S'])/df['S'].rolling(bwin).var()).clip(0.5,2.0).shift(1).fillna(1.0)
    spr=df['L']-beta*df['S']
    rv=spr.rolling(vwin).std()*np.sqrt(A)
    lev=(tgt/rv).clip(0,cap).shift(1).fillna(0.0)
    return (lev*spr).dropna()

wins=[('2019-06-01','2022-12-31','pre-AI 19H2-22'),
      ('2023-01-01','2026-08-31','primary 23-26'),
      ('2019-06-01','2026-08-31','full')]
pairs=[('HP long / HP short',HP_L,HP_S),
       ('HP long / RULES short',HP_L,R_S),
       ('RULES long / RULES short',R_L,R_S)]

for a,b,wl in wins:
    print(f"\n================= {wl}  ({a}..{b}) =================")
    out={}
    for name,Lc,Sc in pairs:
        L=ew(Lc,a,b); S=ew(Sc,a,b)
        out[name+'  [dollar-neutral]']=stats((L-S))
        out[name+'  [betahedge+voltgt]']=stats(betahedge_voltgt(L,S))
    print(pd.DataFrame(out).T.to_string())

# show which short names actually have data in the pre-AI window
print("\nshort-name data coverage:")
for t in sorted(set(HP_S+R_S)):
    s=p[t].dropna() if t in p.columns else pd.Series(dtype=float)
    print(f"  {t:5s} {s.index.min().date() if len(s) else 'NONE':>10}  n={len(s)}")
