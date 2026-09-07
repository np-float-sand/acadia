import pandas as pd, glob, numpy as np
fs = sorted(glob.glob('grid_equipment_basket/data/cache/prices_*.parquet'))
panel = pd.concat([pd.read_parquet(f) for f in fs]).sort_index()
panel = panel[~panel.index.duplicated(keep='last')].astype(float)

US9 = ['ETN','HUBB','GEV','VRT','PWR','MYRG','NVT','FLNC','PRIM']
DER = ['ENPH','SEDG','CHPT','RUN','BLNK','STEM']

def ew_index(cols, start='2022-06-01', end='2024-01-31'):
    px = panel.loc[start:end, cols].dropna(axis=1, how='all').ffill()
    rets = px.pct_change().fillna(0.0)
    port = rets.mean(axis=1)               # daily equal-weight, rebalanced daily
    return (1+port).cumprod()

basket = ew_index(US9)
spy    = ew_index(['SPY'])
der    = ew_index(DER)

# monthly snapshots
me = basket.resample('ME').last()
mS = spy.resample('ME').last()
mD = der.resample('ME').last()

# doc normalises so that basket ~1.0 near 2022-12 / SPY ~0.86 there. Reproduce that scale:
# doc: 2022-12 basket 0.975, SPY 0.863.  -> base = value on 2022-12 / 0.975 etc. Actually doc looks
# indexed to some future point. Just divide by 2023-12 value to compare *shape*, then also show ratio.
def show(lbl, s):
    s = s / s.loc['2022-12-31':'2022-12-31'].iloc[0] * 0.975 if lbl=='basket' else s
    return s

print('=== Monthly cumulative (raw, base=2022-06 first day = 1.0) ===')
tab = pd.DataFrame({'basket':me, 'SPY':mS, 'DER':mD})
tab['basket/SPY'] = tab.basket/tab.SPY
tab['US9-DER spread(long-short cum)'] = tab.basket - tab.DER
print(tab.round(3).to_string())

print()
print('=== basket/SPY ratio, month over month change ===')
r = (me/mS)
print((r).round(3).to_string())
print()
print('rel-perf 2022-06..2022-11:', round(r.loc[:"2022-11-30"].iloc[-1]/r.iloc[0]-1,3))
print('rel-perf 2022-11..2023-05:', round(r.loc[:"2023-05-31"].iloc[-1]/r.loc[:"2022-11-30"].iloc[-1]-1,3))
print('rel-perf 2023-05..2023-09:', round(r.loc[:"2023-09-30"].iloc[-1]/r.loc[:"2023-05-31"].iloc[-1]-1,3))

# Long/short US9 vs DER over the *pre-AI* window to test regime dependence
print()
print('=== US9 long / DER short, dollar-neutral daily-rebal, by period ===')
def ls_stats(start,end):
    px = panel.loc[start:end]
    L = px[US9].dropna(axis=1,how='all').ffill().pct_change().mean(axis=1)
    S = px[DER].dropna(axis=1,how='all').ffill().pct_change().mean(axis=1)
    sp = (L - S).dropna()
    ann = sp.mean()*252; vol = sp.std()*np.sqrt(252)
    return dict(days=len(sp), ann_ret=round(ann,3), ann_vol=round(vol,3), sharpe=round(ann/vol,2),
               cum=round((1+sp).prod()-1,3))
for a,b,lbl in [('2019-06-01','2022-12-31','pre-AI 19H2-22'),
                ('2023-01-01','2026-08-31','AI era 23-26'),
                ('2022-11-30','2023-12-31','Nov22 filing -> end23')]:
    print(f'{lbl:24s}', ls_stats(a,b))
