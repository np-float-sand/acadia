import pandas as pd, numpy as np, glob, warnings
warnings.simplefilter("ignore")
P=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))]
            +[pd.read_parquet('research/capex_cycle_pair_classifier/_cache/policy_prices.parquet')],axis=0
           ).groupby(level=0).last().sort_index().loc[:,lambda d:~d.columns.duplicated()].astype(float)
L=['ETN','HUBB','NVT','VRT','GEV','PWR','MYRG','PRIM','POWL','ATKR','WCC','ABBNY','SBGSY','PRYMY','HTHIY','NXT','FSLR']
# hand-assigned ex-ante observables (known in 2022, not from returns)
SUBSECTOR={'VRT':'DCpower','GEV':'DCpower','POWL':'DCpower','ETN':'DCpower',   # meaningful data-center power exposure
           'HUBB':'TD','NVT':'TD','PRYMY':'TD','HTHIY':'TD','ABBNY':'TD','SBGSY':'TD','WCC':'TD',
           'PWR':'EPC','MYRG':'EPC','PRIM':'EPC',
           'NXT':'solar','FSLR':'solar','ATKR':'other'}
# rough 2022-year-end market cap ($B), for a small-cap-tilt test
MKTCAP_2022={'ETN':63,'HUBB':13,'NVT':7,'VRT':5,'GEV':None,'PWR':21,'MYRG':1.9,'PRIM':1.2,'POWL':0.3,
             'ATKR':4.6,'WCC':6.7,'ABBNY':60,'SBGSY':78,'PRYMY':9,'HTHIY':60,'NXT':4,'FSLR':16}

px=P[L].ffill()
q_ends=pd.date_range('2022-12-31','2026-06-30',freq='QE')
rows=[]
for i in range(len(q_ends)-1):
    a,b=q_ends[i],q_ends[i+1]
    fwd=px.loc[a:b].iloc[[0,-1]]
    fret=(fwd.iloc[-1]/fwd.iloc[0]-1).dropna()
    if len(fret)<8: continue
    # ex-ante features measured at quarter start `a`
    trail=px.loc[:a].tail(63)
    vol=trail.pct_change().std()*np.sqrt(252)
    prev_q=px.loc[q_ends[i-1]:a].iloc[[0,-1]] if i>0 else None
    pret=(prev_q.iloc[-1]/prev_q.iloc[0]-1) if prev_q is not None else pd.Series(index=L,dtype=float)
    smh=P['SMH'].loc[:a].tail(126).pct_change() if 'SMH' in P.columns else None
    for t in fret.index:
        rows.append(dict(q=a.date(), t=t, fret=fret[t],
                         subsector=SUBSECTOR[t],
                         mktcap=MKTCAP_2022.get(t),
                         trail_vol=vol.get(t),
                         prev_qret=pret.get(t)))
d=pd.DataFrame(rows)
d['fwd_rank']=d.groupby('q')['fret'].rank(pct=True)

print("=== avg forward-quarter return by ex-ante subsector ===")
print(d.groupby('subsector')['fret'].agg(['mean','count']).round(3))
print("\n=== quarters where DCpower sub-sector led (mean fret DCpower vs rest) ===")
for q,g in d.groupby('q'):
    dc=g[g.subsector=='DCpower']['fret'].mean(); rest=g[g.subsector!='DCpower']['fret'].mean()
    print(f"  {q}: DCpower {dc:+.1%}  rest {rest:+.1%}  {'DC WINS' if dc>rest else ''}")

print("\n=== rank correlation of ex-ante features with forward-quarter return rank (pooled) ===")
dd=d.dropna(subset=['mktcap','trail_vol'])
for f in ['mktcap','trail_vol','prev_qret']:
    c=dd[['fwd_rank',f]].dropna().corr(method='spearman').iloc[0,1]
    print(f"  {f:12s} spearman vs fwd_rank: {c:+.2f}  (n={dd[[f,'fwd_rank']].dropna().shape[0]})")
c_dc=d.assign(is_dc=(d.subsector=='DCpower').astype(int))[['fwd_rank','is_dc']].corr(method='spearman').iloc[0,1]
print(f"  is_DCpower   spearman vs fwd_rank: {c_dc:+.2f}")
