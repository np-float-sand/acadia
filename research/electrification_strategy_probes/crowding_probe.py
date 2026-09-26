"""THROWAWAY (2026-09-06) -- crowding fixes: (1) quality-broadened universe vs the
marquee 9, (2) a valuation/extension de-risk overlay. Pre-registered bars in comments."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.basket import simulate_basket
from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket import hedges
START,END="2017-06-01","2026-08-31"; RF,AF=0.04,252
WIN={"full":("2017-06-01","2026-08-31"),"2019-22":("2019-01-01","2022-12-31"),"2023-26":("2023-01-01","2026-08-31")}
EP={"COVID":(("2020-02-01","2020-02-25"),"2020-04-30"),"Rate22":(("2021-11-01","2022-01-31"),"2022-12-31"),
    "DeepSeek":(("2024-11-01","2025-01-15"),"2025-05-31"),"Tariff":(("2025-03-15","2025-04-03"),"2025-04-30"),
    "Selloff26":(("2026-05-01","2026-06-20"),"2026-08-31")}
MARQUEE=["ETN","HUBB","GEV","VRT","PWR","MYRG","NVT","FLNC","PRIM"]
# broadened: drop FLNC (chronic losses -> fails quality), add established profitable
# electrical / power-equipment / electrical-EPC / data-center-thermal names
BROAD=["ETN","HUBB","GEV","VRT","PWR","MYRG","NVT","PRIM",
       "EMR","AME","RRX","POWL","ATKR","AEIS","GNRC","EME","FIX","MTZ","TT","AYI"]
def vt(r,t=0.20,lb=21,c=1.5):
    r=r.dropna();rv=r.rolling(lb).std()*np.sqrt(AF);lev=(t/rv).clip(upper=c).shift(1);return (lev*r+(1-lev)*RF/AF).dropna()
def mult_hold(daily,idx):
    p=idx.to_period("M");bm=pd.Series(daily.to_numpy(),index=idx).groupby(p).last()
    return pd.Series(bm.shift(1).reindex(p).to_numpy(),index=idx).astype(float)
def cg(r):return compute_metrics(r,RF,AF)["cagr"]
def sh(r):return compute_metrics(r,RF,AF)["sharpe"]
def dd(r):return compute_metrics(r,RF,AF)["max_dd"]
def eff_n(w):  # effective number of names from last weight row
    w=w[w>0]; return 1.0/ (w**2).sum() * (w.sum()**2)

allt=sorted(set(MARQUEE+BROAD+["SPY","VOLT","PAVE"]))
px=fetch_prices(allt,START,END)
print("missing:",[t for t in BROAD if t not in px.columns])
spy=px["SPY"].pct_change()

def book(names):
    cols=[t for t in names if t in px.columns]
    br=simulate_basket(px[cols],START,END,42,0.25)
    return br
brM,brB=book(MARQUEE),book(BROAD)
lbM,lbB=vt(brM.returns),vt(brB.returns)

print("\n===== (1) QUALITY BROADENING =====")
print(f"marquee names used: {list(brM.weights.columns)}")
print(f"broadened names used ({len(brB.weights.columns)}): {list(brB.weights.columns)}")
for nm,(lb,br) in [("marquee-9",(lbM,brM)),("broadened-20",(lbB,brB))]:
    m=compute_metrics(lb,RF,AF)
    wl=br.weights.iloc[-1]; en=eff_n(wl.values)
    vcommon=px["VOLT"].pct_change().dropna().index.intersection(lb.index)
    cV=lb.reindex(vcommon).corr(px["VOLT"].pct_change().reindex(vcommon))
    b=np.cov(pd.concat([lb.rename('a'),spy.rename('b')],axis=1).dropna().T)[0,1]/spy.reindex(lb.index).var()
    eps={k:hedges.episode_drawdown(lb,*hedges.find_drawdown_episode(lb,pw,te)) for k,(pw,te) in EP.items()}
    print(f"\n{nm}: CAGR {m['cagr']*100:.1f}%  vol {m['ann_vol']*100:.1f}%  Sharpe {m['sharpe']:.2f}  "
          f"MaxDD {m['max_dd']*100:.1f}%  SPYbeta {b:.2f}")
    print(f"  top weight {wl.max()*100:.0f}%   effective #names {en:.1f}/{len(wl[wl>0])}   corr vs VOLT {cV:.2f}")
    print("  episode DDs: "+"  ".join(f"{k} {v*100:.0f}%" for k,v in eps.items()))
    for w,(a,b2) in WIN.items():
        print(f"    {w:8s} Sharpe {sh(lb.loc[a:b2]):+.2f}  CAGR {cg(lb.loc[a:b2])*100:+.1f}%")

print("\n===== (2) VALUATION / EXTENSION DE-RISK OVERLAY =====")
# rule (pre-registered, no optimisation): monthly, known next month.
#   ext = P/MA200 - 1 ;  rs12 = book_252d_ret - SPY_252d_ret
#   m = 1.0 if ext<=0.15 and rs12<=0.25 ;  0.6 if ext>0.35 or rs12>0.50 ;  else 0.8
# bar: full-window MaxDD improves >=3pp, Sharpe within -0.05, CAGR drag <=4%/yr;
#      no sub-window Sharpe drop >0.15 ; avg multiplier >=0.75.  +/-20% threshold plateau.
def val_overlay(lb, lo_e,hi_e,lo_r,hi_r):
    lvl=(1+lb).cumprod()
    ext=lvl/lvl.rolling(200).mean()-1
    r12=lvl.pct_change(252); s12=(1+spy.reindex(lvl.index)).cumprod().pct_change(252)
    rs=r12-s12
    m=pd.Series(0.8,index=lvl.index)
    m[(ext<=lo_e)&(rs<=lo_r)]=1.0
    m[(ext>hi_e)|(rs>hi_r)]=0.6
    mh=mult_hold(m,lvl.index).fillna(1.0)
    return (mh*lb+(1-mh)*RF/AF).reindex(lb.index), mh
for lb,nm in [(lbM,"marquee-9"),(lbB,"broadened-20")]:
    base=compute_metrics(lb,RF,AF)
    print(f"\n--- {nm} base: CAGR {base['cagr']*100:.1f}%  Sharpe {base['sharpe']:.2f}  MaxDD {base['max_dd']*100:.1f}% ---")
    for scale in (0.8,1.0,1.2):
        lo_e,hi_e,lo_r,hi_r=0.15*scale,0.35*scale,0.25*scale,0.50*scale
        o,mh=val_overlay(lb,lo_e,hi_e,lo_r,hi_r)
        m=compute_metrics(o,RF,AF)
        sub=" ".join(f"{w}:{sh(o.loc[a:b])-sh(lb.loc[a:b]):+.2f}" for w,(a,b) in WIN.items() if w!="full")
        print(f"  thr x{scale}: CAGR {m['cagr']*100:5.1f}% (drag {(base['cagr']-m['cagr'])*100:+.1f})  "
              f"Sharpe {m['sharpe']:.2f} ({m['sharpe']-base['sharpe']:+.2f})  MaxDD {m['max_dd']*100:6.1f}% "
              f"({(abs(base['max_dd'])-abs(m['max_dd']))*100:+.1f}pp)  avgMult {mh.mean():.2f}  subSharpeΔ[{sub}]")

print("\n===== (3) FULL STACK: broadened + valuation overlay + hedge overlay =====")
# hedge overlay from the prior probe: +0.15 GLD/IEF sleeve - 0.25x (DLR+EQIX) gated on 6-mo real-yield rise
def fred(s):
    f=pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd=2000-01-01",parse_dates=[0],index_col=0)
    return pd.to_numeric(f.iloc[:,0],errors="coerce").dropna()
px2=fetch_prices(["DLR","EQIX","GLD","IEF","TAN"],START,END)
dcr=simulate_basket(px2[["DLR","EQIX"]],START,END,42,1.0).returns.reindex(lbB.index).fillna(0)
gi=simulate_basket(px2[["GLD","IEF"]],START,END,42,1.0).returns.reindex(lbB.index).fillna(0)
dfii=fred("DFII10").reindex(pd.date_range(START,END)).ffill()
s1d=((dfii-dfii.shift(126))>0).shift(5).fillna(False)
S1=mult_hold(s1d.reindex(lbB.index).ffill().fillna(False).astype(float),lbB.index).fillna(0)>0.5
tanm=(1+px2["TAN"].pct_change().reindex(lbB.index)).resample("ME").prod()-1
o_val,_=val_overlay(lbB,0.15,0.35,0.25,0.50)
full=0.85*o_val + 0.15*gi - 0.25*(dcr*S1.astype(float))
for nm,r in [("marquee-9 plain (today)",lbM),("broadened-20 plain",lbB),
             ("broadened + valuation overlay",o_val),("broadened + valuation + hedge overlay",full.dropna())]:
    m=compute_metrics(r,RF,AF)
    b=np.cov(pd.concat([r.rename('a'),spy.rename('b')],axis=1).dropna().T)[0,1]/spy.reindex(r.index).var()
    eps={k:hedges.episode_drawdown(r,*hedges.find_drawdown_episode(lbB,pw,te)) for k,(pw,te) in EP.items()}
    rm=(1+r).resample("ME").prod()-1; lbm=(1+lbB).resample("ME").prod()-1
    fear=((lbm<0)&(tanm>0)).reindex(rm.index).fillna(False)
    fp=float((rm[fear]-lbm.reindex(rm.index)[fear]).sum())
    print(f"\n{nm}:\n  CAGR {m['cagr']*100:.1f}%  Sharpe {m['sharpe']:.2f}  MaxDD {m['max_dd']*100:.1f}%  SPYbeta {b:.2f}  fearP&L {fp*100:+.0f}%")
    print("  episode DDs: "+"  ".join(f"{k} {v*100:.0f}%" for k,v in eps.items()))
