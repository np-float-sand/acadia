import pandas as pd, numpy as np, glob, warnings
warnings.simplefilter("ignore")
RF=0.04; A=252; CAP=0.35

def load():
    parts=[pd.read_parquet('research/capex_cycle_pair_classifier/_cache/hist_prices.parquet'),
           pd.read_parquet('research/capex_cycle_pair_classifier/_cache/gen_prices.parquet'),
           pd.read_parquet('research/capex_cycle_pair_classifier/_cache/der_extra_prices.parquet')]
    parts+=[pd.read_parquet(f) for f in sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))]
    p=pd.concat(parts,axis=0)
    p=p.groupby(level=0).last().sort_index()
    # collapse dup columns
    p=p.loc[:,~p.columns.duplicated()]
    return p.astype(float)
P=load()

INSTANCES={
 "grid/DER 2023":        dict(L=['ETN','HUBB','NVT','VRT','GEV','PWR','MYRG','PRIM','POWL','ATKR','WCC','ABBNY','SBGSY','PRYMY','HTHIY','NXT','FSLR'],
                              S=['BLNK','CHPT','ENPH','EVGO','RUN','SEDG','STEM','WBX'],
                              mania=('2019-06-01','2022-12-31'), post=('2023-01-01','2026-08-31')),
 "EV 2022":             dict(L=['ALSN','APTV','BWA','DAN','LEA','MGA','ST','VC'],
                              S=['LCID','LI','NIO','PSNY','RIVN','WKHS','XPEV'],
                              mania=('2020-06-01','2021-12-31'), post=('2022-01-01','2026-08-31')),
 "nuclear/SMR 2024":    dict(L=['BWXT','CCJ','LEU'], S=['SMR','OKLO','NNE','LTBR','ASPI'],
                              mania=('2022-03-01','2023-12-31'), post=('2024-01-01','2026-08-31')),
 "hydrogen 2000":       dict(L=['LIN','APD','CMI'], S=['PLUG','FCEL','BLDP'],
                              mania=('1999-06-01','2000-12-31'), post=('2001-01-01','2004-12-31')),
 "hydrogen 2020":       dict(L=['LIN','APD','CMI'], S=['PLUG','FCEL','BLDP','BE'],
                              mania=('2020-06-01','2021-01-31'), post=('2021-02-01','2026-08-31')),
 "genomics 2021":       dict(L=['ILMN','TMO','DHR'], S=['CRSP','NTLA','BEAM','EDIT'],
                              mania=('2020-06-01','2021-02-28'), post=('2021-03-01','2026-08-31')),
 "space 2021":          dict(L=['IRDM'], S=['ASTS','PL','BKSY','RDW','SPCE','GSAT'],
                              mania=('2020-11-01','2021-12-31'), post=('2022-01-01','2026-08-31')),
 "cannabis 2018":       dict(L=['SMG','IIPR'], S=['TLRY','CGC','ACB','CRON','SNDL','OGI'],
                              mania=('2018-06-01','2018-12-31'), post=('2019-01-01','2026-08-31')),
}

def ew(cols,a,b):
    have=[c for c in cols if c in P.columns and P[c].loc[a:b].notna().sum()>20]
    r=P.loc[a:b,have].ffill().pct_change().clip(-CAP,CAP)
    return r.mean(axis=1), have
def sharpe(r):
    r=r.dropna()
    if len(r)<60: return None,len(r),None
    vol=r.std()*np.sqrt(A); sh=(r.mean()*A-RF)/vol
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return round(sh,2),len(r),f"{dd:.0%}"
def bhvt(L,S,tgt=.20,cap=1.5,bw=60,vw=20):
    d=pd.concat([L.rename('L'),S.rename('S')],axis=1).dropna()
    if len(d)<90: return pd.Series(dtype=float)
    b=(d['L'].rolling(bw).cov(d['S'])/d['S'].rolling(bw).var()).clip(.5,2).shift(1).fillna(1)
    spr=d['L']-b*d['S']; rv=spr.rolling(vw).std()*np.sqrt(A)
    return ((tgt/rv).clip(0,cap).shift(1).fillna(0)*spr).dropna()
def break_gate(Sidx):
    # short-sleeve EW price index; gate = index < 200d MA AND MA falling over 20d
    lvl=(1+Sidx.fillna(0)).cumprod()
    ma=lvl.rolling(200).mean()
    g=((lvl<ma) & (ma.diff(20)<0)).astype(float).shift(1).fillna(0)
    return g

print(f"{'instance':18s} {'L':>2} {'S':>2} | {'mania BH+VT':>12} | {'post BH+VT':>11} | {'post +break-gate':>16}")
print("-"*82)
for nm,d in INSTANCES.items():
    (a0,a1),(b0,b1)=d['mania'],d['post']
    Lm,lh=ew(d['L'],a0,a1); Sm,sh=ew(d['S'],a0,a1)
    Lp,lhp=ew(d['L'],b0,b1); Sp,shp=ew(d['S'],b0,b1)
    if len(lhp)<3 or len(shp)<3:
        print(f"{nm:18s} {len(lhp):>2} {len(shp):>2} | RULE SELF-REJECTS (need >=3/side)")
        continue
    shm,_,_=sharpe(bhvt(Lm,Sm))
    shp_,_,ddp=sharpe(bhvt(Lp,Sp))
    # break-gated over the FULL available span (mania start -> post end)
    Lf,_=ew(d['L'],a0,b1); Sf,_=ew(d['S'],a0,b1)
    g=break_gate(Sf).reindex((Lf.index)).fillna(0)
    raw=bhvt(Lf,Sf)
    gg=g.reindex(raw.index).fillna(0)
    shg,ng,ddg=sharpe(raw*gg)
    print(f"{nm:18s} {len(lhp):>2} {len(shp):>2} | {str(shm):>12} | {str(shp_):>11} ({ddp}) | {str(shg):>7} (dd {ddg}, {int(gg.sum())}d on)")
