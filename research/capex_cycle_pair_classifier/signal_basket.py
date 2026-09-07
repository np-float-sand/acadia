import pandas as pd, numpy as np, glob
pd.set_option('display.width',220)
RF=0.04; A=252

fs=sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))
panel=pd.concat([pd.read_parquet(f) for f in fs]).sort_index()
panel=panel[~panel.index.duplicated(keep='last')].astype(float)

US9=['ETN','HUBB','GEV','VRT','PWR','MYRG','NVT','FLNC','PRIM']
ADR=['ABBNY','SBGSY','PRYMY','HTHIY']

def basket_ret(cols):
    px=panel[[c for c in cols if c in panel.columns]].ffill()
    return px.pct_change().mean(axis=1)          # daily EW, rebal daily

def stats(r):
    r=r.dropna()
    if len(r)<20: return {}
    cum=(1+r).prod()-1
    yrs=len(r)/A
    cagr=(1+cum)**(1/yrs)-1
    vol=r.std()*np.sqrt(A)
    sh=(r.mean()*A-RF)/vol
    eq=(1+r).cumprod(); dd=(eq/eq.cummax()-1).min()
    return dict(tot=f"{cum:+.1%}", cagr=f"{cagr:+.1%}", vol=f"{vol:.1%}", sharpe=round(sh,2), maxdd=f"{dd:.1%}")

# --- signal: PJM data-center MW revision.  Only ever fires once in-sample: STALL at the 2026-LF
#     preliminary (2025-11-24), near-term @2028 revised -3,078 MW.  Everywhere else = fully invested.
SIG_DATE_PRELIM = pd.Timestamp('2025-11-24')
SIG_DATE_PUBLISH= pd.Timestamp('2026-01-15')

def gated(r, sig_date, exposure_when_stalled):
    e=pd.Series(1.0, index=r.index)
    e.loc[r.index>=sig_date]=exposure_when_stalled
    return e*r + (1-e)*(RF/A)

for name,cols in [('US-9 + 4 ADRs', US9+ADR), ('US-9 only', US9)]:
    r=basket_ret(cols)
    for w0,w1,wl in [('2021-01-15','2026-08-31','FULL 2021-01 -> 2026-08'),
                     ('2023-01-01','2026-08-31','2023-01 -> 2026-08'),
                     ('2025-11-24','2026-08-31','SIGNAL WINDOW 2025-11-24 -> 2026-08')]:
        seg=r.loc[w0:w1]
        print(f"\n===== {name} | {wl} =====")
        rows={
          'plain basket (no signal)'          : stats(seg),
          'signal, stall->50%, @LAS prelim'   : stats(gated(seg, SIG_DATE_PRELIM, 0.5)),
          'signal, stall->0%,  @LAS prelim'   : stats(gated(seg, SIG_DATE_PRELIM, 0.0)),
          'signal, stall->50%, @Jan publish'  : stats(gated(seg, SIG_DATE_PUBLISH, 0.5)),
          'signal, stall->0%,  @Jan publish'  : stats(gated(seg, SIG_DATE_PUBLISH, 0.0)),
        }
        print(pd.DataFrame(rows).T.to_string())
