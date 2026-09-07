import pandas as pd, numpy as np, glob, warnings
warnings.simplefilter("ignore")
A=252
parts=[pd.read_parquet('research/capex_cycle_pair_classifier/_cache/der_extra_prices.parquet')]
parts+=[pd.read_parquet(f) for f in sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))]
P=pd.concat(parts,axis=0).groupby(level=0).last().sort_index()
P=P.loc[:,~P.columns.duplicated()].astype(float)

L=['ETN','HUBB','NVT','VRT','GEV','PWR','MYRG','PRIM','POWL','ATKR','WCC','ABBNY','SBGSY','PRYMY','HTHIY','NXT','FSLR']
S=['ENPH','SEDG','CHPT','RUN','BLNK','STEM','EVGO','WBX']
def ew(c): 
    h=[x for x in c if x in P.columns]
    return P[h].ffill().pct_change().clip(-.35,.35).mean(axis=1)
def r(t): return P[t].ffill().pct_change().clip(-.35,.35)
def shp(x):
    x=x.dropna(); 
    return (x.mean()*A)/(x.std()*np.sqrt(A)) if len(x)>40 else np.nan

book = ew(L)-ew(S)                    # OUR dollar-neutral book
pave_tan = r('PAVE')-r('TAN')         # the naive pitch = the ETF pair
grid_tan = r('GRID')-r('TAN')         # concentrated-grid ETF vs solar ETF

for lbl,a,b in [('2023-01→2026-08','2023-01-01','2026-08-31'),
                ('2019-06→2022-12','2019-06-01','2022-12-31'),
                ('full 2019-06→2026-08','2019-06-01','2026-08-31')]:
    d=pd.concat({'book':book,'pave_tan':pave_tan,'grid_tan':grid_tan,
                 'PAVE':r('PAVE'),'TAN':r('TAN'),'SPY':r('SPY')},axis=1).loc[a:b].dropna()
    print(f"\n===== {lbl} =====")
    print(f"  OUR book Sharpe        {shp(d.book):+.2f}")
    print(f"  naive pitch PAVE-TAN   {shp(d.pave_tan):+.2f}   corr(book,PAVE-TAN)={d.book.corr(d.pave_tan):+.2f}")
    print(f"  GRID-TAN               {shp(d.grid_tan):+.2f}   corr(book,GRID-TAN)={d.book.corr(d.grid_tan):+.2f}")
    # hedge our book against SPY + PAVE + TAN; report alpha t-stat and the hedged stream Sharpe
    X=np.column_stack([np.ones(len(d)),d.SPY,d.PAVE,d.TAN]); Y=d.book.values
    bta,_,_,_=np.linalg.lstsq(X,Y,rcond=None)
    resid=Y-X@bta
    se=np.sqrt(((resid**2).sum()/(len(d)-4))*np.linalg.inv(X.T@X)[0,0])
    tstat=bta[0]/se
    hedged = d.book - (bta[1]*d.SPY + bta[2]*d.PAVE + bta[3]*d.TAN)   # keep intercept in
    ss=1-(resid**2).sum()/((Y-Y.mean())**2).sum()
    print(f"  book ~ SPY+PAVE+TAN :  R²={ss:.2f}  ann_alpha={bta[0]*A:+.0%} (t={tstat:.1f})  loadings SPY={bta[1]:+.2f} PAVE={bta[2]:+.2f} TAN={bta[3]:+.2f}")
    print(f"  --> book HEDGED of SPY/PAVE/TAN: Sharpe {shp(hedged):+.2f}   (this is what the stock selection adds beyond the pitch)")
