import pandas as pd, numpy as np, glob, warnings
warnings.simplefilter("ignore")
A=252
parts=[pd.read_parquet(f) for f in [
 'research/capex_cycle_pair_classifier/_cache/gen_prices.parquet',
 'research/capex_cycle_pair_classifier/_cache/der_extra_prices.parquet',
 'research/capex_cycle_pair_classifier/_cache/live_prices.parquet']]
parts+=[pd.read_parquet(f) for f in sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))]
P=pd.concat(parts,axis=0).groupby(level=0).last().sort_index()
P=P.loc[:,~P.columns.duplicated()].astype(float)

L=['ETN','HUBB','NVT','VRT','GEV','PWR','MYRG','PRIM','POWL','ATKR','WCC','ABBNY','SBGSY','PRYMY','HTHIY','NXT','FSLR']
S=['BLNK','CHPT','ENPH','EVGO','RUN','SEDG','STEM','WBX']
def ew(c): 
    h=[x for x in c if x in P.columns]
    return P[h].ffill().pct_change().clip(-.35,.35).mean(axis=1)
rL,rS=ew(L),ew(S)
spread=(rL-rS)                      # dollar-neutral grid/DER

def r(t): return P[t].ffill().pct_change() if t in P.columns else None
mkt=r('SPY')
qual=r('COWZ')-r('SPY')            # cash-cows minus market ~ quality/profitability
lowvol=r('USMV')-r('SPY')          # defensive
dur=r('TLT')                        # long-duration Treasuries = rates/duration factor
sect=r('PAVE')-r('TAN')            # infrastructure minus solar = the sector pair
mom=r('XLI')-r('ICLN')            # industrials minus clean energy (another sector cut)

for lbl,a,b in [('2023-01→2026-08','2023-01-01','2026-08-31'),('2021-01→2022-12','2021-01-01','2022-12-31')]:
    df=pd.concat({'y':spread,'mkt':mkt,'qual':qual,'lowvol':lowvol,'dur':dur,'sect':sect},axis=1).loc[a:b].dropna()
    Y=df['y'].values
    def reg(cols):
        X=np.column_stack([np.ones(len(df))]+[df[c].values for c in cols])
        bta,_,_,_=np.linalg.lstsq(X,Y,rcond=None)
        pred=X@bta; ss=1-((Y-pred)**2).sum()/((Y-Y.mean())**2).sum()
        resid=Y-pred
        al=bta[0]*A; sr_res=(resid.mean()*A)/(resid.std()*np.sqrt(A))
        return bta,ss,al,sr_res
    print(f"\n===== {lbl}  (grid/DER dollar-neutral spread; raw Sharpe {(df['y'].mean()*A)/(df['y'].std()*np.sqrt(A)):.2f}) =====")
    for cols in [['mkt'],['mkt','qual'],['mkt','qual','dur'],['mkt','sect'],['mkt','qual','dur','sect'],['mkt','qual','lowvol','dur','sect']]:
        bta,ss,al,srr=reg(cols)
        coefs=", ".join(f"{c}={bta[i+1]:+.2f}" for i,c in enumerate(cols))
        print(f"  ~ {'+'.join(cols):28s}  R²={ss:0.2f}  ann_alpha={al:+.1%}  resid_Sharpe={srr:+.2f}   [{coefs}]")
