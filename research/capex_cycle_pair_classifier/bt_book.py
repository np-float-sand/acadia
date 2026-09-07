import pandas as pd, numpy as np, glob, warnings
warnings.simplefilter("ignore")
RF=0.04; A=252; CAP=0.35

parts=[pd.read_parquet(f) for f in [
 'research/capex_cycle_pair_classifier/_cache/hist_prices.parquet',
 'research/capex_cycle_pair_classifier/_cache/gen_prices.parquet',
 'research/capex_cycle_pair_classifier/_cache/der_extra_prices.parquet',
 'research/capex_cycle_pair_classifier/_cache/live_prices.parquet']]
parts+=[pd.read_parquet(f) for f in sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))]
P=pd.concat(parts,axis=0).groupby(level=0).last().sort_index()
P=P.loc[:,~P.columns.duplicated()].astype(float)

PAIRS={
 "grid/DER":  dict(L=['ETN','HUBB','NVT','VRT','GEV','PWR','MYRG','PRIM','POWL','ATKR','WCC','ABBNY','SBGSY','PRYMY','HTHIY','NXT','FSLR'],
                   S=['BLNK','CHPT','ENPH','EVGO','RUN','SEDG','STEM','WBX'], live='2022-06-01'),
 "EV":        dict(L=['ALSN','APTV','BWA','DAN','LEA','MGA','ST','VC'],
                   S=['LCID','LI','NIO','PSNY','RIVN','WKHS','XPEV'], live='2022-06-01'),
 "hydrogen":  dict(L=['LIN','APD','CMI'], S=['PLUG','FCEL','BLDP','BE'], live='2022-06-01'),
 "eVTOL":     dict(L=['HON','TDG','HEI','TXT','GRMN'], S=['JOBY','ACHR','EVTL'], live='2022-06-01'),
 "battery-tech":dict(L=['ALB','SQM'], S=['QS','MVST','SLDP','ENVX','SES'], live='2022-06-01'),
}
def ew(cols,a,b):
    have=[c for c in cols if c in P.columns and P[c].loc[a:b].notna().sum()>20]
    r=P.loc[a:b,have].ffill().pct_change().clip(-CAP,CAP)
    return r.mean(axis=1),have
def bhvt(L,S,tgt=.15,cap=1.5,bw=60,vw=20):
    d=pd.concat([L.rename('L'),S.rename('S')],axis=1).dropna()
    b=(d['L'].rolling(bw).cov(d['S'])/d['S'].rolling(bw).var()).clip(.5,2).shift(1).fillna(1)
    spr=d['L']-b*d['S']; rv=spr.rolling(vw).std()*np.sqrt(A)
    return ((tgt/rv).clip(0,cap).shift(1).fillna(0)*spr).dropna()
def sh(r):
    r=r.dropna()
    if len(r)<60: return None
    v=r.std()*np.sqrt(A); s=(r.mean()*A-RF)/v
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return dict(sharpe=round(s,2),vol=f"{v:.0%}",maxdd=f"{dd:.0%}",tot=f"{(1+r).prod()-1:+.0%}")

END='2026-08-31'
streams={}
print("=== individual pairs, live-date -> 2026-08, vol-targeted 15% ===")
for nm,d in PAIRS.items():
    L,lh=ew(d['L'],d['live'],END); S,shn=ew(d['S'],d['live'],END)
    if len(lh)<2 or len(shn)<2: print(f"{nm:12s} skip (L={len(lh)} S={len(shn)})"); continue
    r=bhvt(L,S); streams[nm]=r
    st=sh(r); print(f"{nm:12s} L={len(lh)} S={len(shn)}  Sharpe {st['sharpe']:>5}  vol {st['vol']}  maxDD {st['maxdd']}  tot {st['tot']}")

M=pd.DataFrame(streams).dropna()
print("\n=== pairwise correlation of pair-return streams (common window) ===")
print(M.corr().round(2).to_string())

for combo in [["grid/DER","EV"],["grid/DER","EV","hydrogen"],["grid/DER","EV","hydrogen","eVTOL"],
              ["grid/DER","EV","hydrogen","eVTOL","battery-tech"]]:
    sub=M[combo].dropna()
    book=sub.mean(axis=1)   # equal weight, each already ~15% vol
    st=sh(book)
    print(f"\nBOOK {combo}  n={len(sub)}  ->  Sharpe {st['sharpe']}  vol {st['vol']}  maxDD {st['maxdd']}  tot {st['tot']}")
