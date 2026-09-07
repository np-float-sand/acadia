import pandas as pd, numpy as np, glob
pd.set_option('display.width',200)

# ---- price panel ----
fs=sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))
panel=pd.concat([pd.read_parquet(f) for f in fs]).sort_index()
panel=panel[~panel.index.duplicated(keep='last')].astype(float)
US9=['ETN','HUBB','GEV','VRT','PWR','MYRG','NVT','FLNC','PRIM']
FOR=['ABBNY','SBGSY','PRYMY','HTHIY']
DER=['ENPH','SEDG','CHPT','RUN','BLNK','STEM']

def cum(cols,a,b):
    px=panel.loc[a:b,[c for c in cols if c in panel.columns]].ffill()
    r=px.pct_change().mean(axis=1)
    return (1+r.fillna(0)).prod()-1

# ---- PJM RTO large-load adjustment, fixed target year 2030, by vintage ----
b9=pd.read_csv('grid_resilience/data/seed/pjm_large_load_b9_vintages.csv')
rto=b9[(b9.zone=='PJM RTO')].pivot(index='vintage',columns='target_year',values='mw_adj')
order=['2021 LF','2022 LF','2023 LF','2024 LF','2025 LF','2026 LF']
rto=rto.loc[order]

# signal dates: report published ~mid-Jan of the vintage year; LAS 'preliminary' ~late Nov prior
pub={'2021 LF':'2021-01-15','2022 LF':'2022-01-15','2023 LF':'2023-01-15',
     '2024 LF':'2024-01-15','2025 LF':'2025-01-15','2026 LF':'2026-01-15'}

rows=[]
for i,v in enumerate(order):
    for ty in (2028,2030,2032):
        cur=rto.loc[v,ty]
        prev=rto.loc[order[i-1],ty] if i>0 else np.nan
        rev=cur-prev
    d0=pd.Timestamp(pub[v]); d12=d0+pd.DateOffset(months=12)
    d12=min(d12,panel.index[-1])
    bl=cum(US9+FOR,d0,d12); sp=cum(['SPY'],d0,d12); der=cum(DER,d0,d12)
    rows.append(dict(vintage=v,
                     rto2030=int(rto.loc[v,2030]),
                     rev2030=(None if i==0 else int(rto.loc[v,2030]-rto.loc[order[i-1],2030])),
                     rev2028=(None if i==0 else int(rto.loc[v,2028]-rto.loc[order[i-1],2028])),
                     win=f"{d0.date()}->{d12.date()}",
                     basket_fwd=round(bl,3), spy_fwd=round(sp,3),
                     basket_ex_spy=round(bl-sp,3), der_fwd=round(der,3),
                     long_short=round(bl-der,3)))
res=pd.DataFrame(rows)
print("=== PJM RTO large-load adj (2030 target) revision  vs  buildout-basket forward 12m ===")
print(res.to_string(index=False))

r=res.dropna(subset=['rev2030'])
print("\ncorr(rev2030, basket_ex_spy_fwd) n=%d:"%len(r), round(r['rev2030'].astype(float).corr(r['basket_ex_spy']),2))
print("corr(rev2030, long_short_fwd):", round(r['rev2030'].astype(float).corr(r['long_short']),2))

# ---- ERCOT 11-month momentum vs basket fwd 1m ----
e=pd.read_csv('/Users/sandhyapersad/acadia/grid_resilience/data/seed/ercot_large_load_monthly.csv')
e['snapshot_date']=pd.to_datetime(e['snapshot_date']); e=e.sort_values('snapshot_date')
e['mom']=e['total_mw'].pct_change()
fwd=[]
for _,row in e.iterrows():
    d0=row['snapshot_date']; d1=d0+pd.DateOffset(months=1)
    if d1>panel.index[-1]: fwd.append(np.nan); continue
    fwd.append(cum(US9+FOR,d0,d1)-cum(['SPY'],d0,d1))
e['basket_ex_spy_fwd1m']=fwd
print("\n=== ERCOT large-load MoM growth vs basket fwd-1m excess ===")
print(e[['snapshot_date','total_mw','mom','basket_ex_spy_fwd1m']].round(3).to_string(index=False))
ee=e.dropna()
print("corr(ERCOT mom, basket fwd1m excess) n=%d:"%len(ee), round(ee['mom'].corr(ee['basket_ex_spy_fwd1m']),2))
