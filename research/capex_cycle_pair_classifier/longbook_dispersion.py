import pandas as pd, numpy as np, glob, warnings
warnings.simplefilter("ignore")
A=252; RF=0.04
parts=[pd.read_parquet(f) for f in sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))]
parts+=[pd.read_parquet('research/capex_cycle_pair_classifier/_cache/policy_prices.parquet')]
P=pd.concat(parts,axis=0).groupby(level=0).last().sort_index().loc[:,lambda d:~d.columns.duplicated()].astype(float)
L=['ETN','HUBB','NVT','VRT','GEV','PWR','MYRG','PRIM','POWL','ATKR','WCC','ABBNY','SBGSY','PRYMY','HTHIY','NXT','FSLR']

def tr(t,a,b):
    s=P[t].loc[a:b].ffill().dropna()
    return s.iloc[-1]/s.iloc[0]-1 if len(s)>15 else np.nan

print("=== per-name total return by calendar period ===")
print(f"{'tkr':6s} {'2023':>8} {'2024':>8} {'2025':>8} {'2026YTD':>9} {'23-26':>9}")
for t in L:
    r23,r24,r25,r26 = tr(t,'2023-01-01','2023-12-31'),tr(t,'2024-01-01','2024-12-31'),tr(t,'2025-01-01','2025-12-31'),tr(t,'2026-01-01','2026-08-31')
    rall=tr(t,'2023-01-01','2026-08-31')
    f=lambda x: f"{x:+.0%}" if pd.notna(x) else "  —"
    print(f"{t:6s} {f(r23):>8} {f(r24):>8} {f(r25):>8} {f(r26):>9} {f(rall):>9}")

# --- cross-sectional momentum: monthly rank by trailing 126d, hold top-K equal-weight, vs EW all ---
px=P[L].loc['2022-06-01':'2026-08-31'].ffill()
rets=px.pct_change().clip(-.35,.35)
me=px.resample('ME').last()
mom=me.pct_change(6)                      # trailing ~6m at month ends
def bt(weights_fn,lbl):
    monthly_w=[]
    for dt in me.index:
        avail=me.loc[dt].dropna().index
        w=weights_fn(mom.loc[dt].dropna(), avail)
        monthly_w.append((dt,w))
    # apply next-month daily returns
    port=pd.Series(0.0,index=rets.index)
    for i,(dt,w) in enumerate(monthly_w):
        nxt = me.index[i+1] if i+1<len(me.index) else rets.index[-1]
        seg=rets.loc[dt:nxt]
        port.loc[seg.index]= (seg[w.index]*w).sum(axis=1)
    port=port.loc['2023-01-01':]
    s=(port.mean()*A-RF)/(port.std()*np.sqrt(A))
    eq=(1+port).cumprod(); dd=(eq/eq.cummax()-1).min()
    print(f"  {lbl:28s} Sharpe {s:+.2f}  tot {(1+port).prod()-1:+.0%}  maxDD {dd:.0%}")

def ew_all(sig,avail): 
    return pd.Series(1/len(avail),index=avail)
def top_k(k):
    def f(sig,avail):
        s=sig.reindex(avail).dropna()
        if len(s)==0: return pd.Series(1/len(avail),index=avail)
        pick=s.nlargest(min(k,len(s))).index
        return pd.Series(1/len(pick),index=pick)
    return f
def bot_k(k):
    def f(sig,avail):
        s=sig.reindex(avail).dropna()
        if len(s)==0: return pd.Series(1/len(avail),index=avail)
        pick=s.nsmallest(min(k,len(s))).index
        return pd.Series(1/len(pick),index=pick)
    return f

print("\n=== within-long-book cross-sectional 6m-momentum (rebal monthly), 2023-26 ===")
bt(ew_all,"equal-weight all 17")
bt(top_k(6),"top-6 by trailing 6m")
bt(top_k(9),"top-9 by trailing 6m")
bt(bot_k(6),"bottom-6 (losers)")

# rank persistence
r6=me.pct_change(6)
ranks=r6.rank(axis=1)
ac=ranks.corrwith(ranks.shift(1),axis=1).dropna()
print(f"\nmonth-to-month rank autocorrelation: mean {ac.mean():.2f}  (6-month-ahead: {ranks.corrwith(ranks.shift(6),axis=1).dropna().mean():.2f})")
