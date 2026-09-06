"""Does the VA-filing work-type $ mix, mapped to the 9 names, beat equal-weight?
Point-in-time: at the start of calendar year Y use only projects FILED through Y-1."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/sandhyapersad/acadia")
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket import config

df = pd.read_csv("va_transmission_projects.csv")
df = df[df.disposition != "Canceled"].copy()
df["filed"] = pd.to_datetime(df["filed_date"])
df["cost"] = df["cost_usd_m"].astype(float)

NAMES = sorted(config.UNIVERSE)   # ETN FLNC GEV HUBB MYRG NVT PRIM PWR VRT
MAP = {
    "new_line": {"HUBB":1,"PWR":1,"MYRG":1,"PRIM":1},
    "rebuild": {"HUBB":1,"PWR":1,"MYRG":1,"PRIM":1},
    "reconductor": {"HUBB":1,"PWR":1,"MYRG":1},
    "corridor": {"PWR":1,"MYRG":1,"PRIM":1},
    "loop": {"HUBB":1,"PWR":1,"MYRG":1,"PRIM":1},
    "new_substation": {"ETN":1,"HUBB":1,"GEV":0.5},
    "substation_upgrade": {"ETN":1,"HUBB":1},
    "transformer": {"GEV":1},
    "conversion": {"ETN":1,"HUBB":1},
    "underground": {"PRIM":1,"MYRG":1},
    "gen_interconnect": {"GEV":1,"PWR":1},
}
def wt_dollars(frame):
    d={}
    for _,r in frame.iterrows():
        tags=str(r.work_type).split("|")
        for t in tags: d[t]=d.get(t,0.0)+r.cost/len(tags)
    return pd.Series(d)
def tilt(frame, blend):
    w=pd.Series(0.0,index=NAMES)
    ws=wt_dollars(frame)
    for t,v in ws.items():
        m=MAP.get(t)
        if not m: continue
        s=sum(m.values())
        for nm,x in m.items():
            if nm in w.index: w[nm]+=v*x/s
    if w.sum()==0: return pd.Series(1/len(NAMES),index=NAMES)
    w=w/w.sum()
    ew=pd.Series(1/len(NAMES),index=NAMES)
    return (blend*w + (1-blend)*ew)

px = fetch_prices(NAMES,"2019-06-01","2026-08-31")
px = px[[c for c in NAMES if c in px.columns]].sort_index().ffill()
rets = px.pct_change().dropna(how="all")

def run(blend, window, label):
    daily=[]
    wlog={}
    for Y in range(2021,2027):
        asof=pd.Timestamp(f"{Y-1}-12-31")
        sub=df[df.filed<=asof]
        if window: sub=sub[sub.filed>=asof-pd.DateOffset(years=window)]
        w=tilt(sub, blend)
        wlog[Y]=w
        mask=(rets.index>=f"{Y}-01-01")&(rets.index<=f"{Y}-12-31")
        r=rets.loc[mask,NAMES].fillna(0.0)
        daily.append(r.mul(w,axis=1).sum(axis=1).rename("tilt"))
        daily.append(pd.Series(0.0,index=r.index))  # placeholder
    tser=pd.concat([d for i,d in enumerate(daily) if i%2==0])
    ew=rets.loc[tser.index,NAMES].fillna(0.0).mean(axis=1)
    def stats(s):
        ann=252
        cagr=(1+s).prod()**(ann/len(s))-1
        sh=s.mean()/s.std()*np.sqrt(ann)
        dd=((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()
        return cagr,sh,dd
    ct,st,ddt=stats(tser); ce,se,dde=stats(ew)
    print(f"\n{label}  (blend={blend}, window={window or 'expanding'})")
    print(f"  {'':10}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}")
    print(f"  {'worktype':10}{ct*100:>8.1f}%{st:>9.2f}{ddt*100:>8.1f}%")
    print(f"  {'equalwt':10}{ce*100:>8.1f}%{se:>9.2f}{dde*100:>8.1f}%")
    print(f"  {'diff':10}{(ct-ce)*100:>+8.1f}%{st-se:>+9.2f}{(ddt-dde)*100:>+8.1f}%")
    W=pd.DataFrame(wlog).T
    print("  yearly tilt weights:"); print(W.round(3).to_string().replace("\n","\n  "))
    print(f"  avg turnover (sum|dw|)/2: {W.diff().abs().sum(axis=1).mean()/2:.3f}")

for bl in (1.0, 0.5):
    run(bl, None, "full-history mix")
run(1.0, 2, "trailing-2yr mix")
run(1.0, 3, "trailing-3yr mix")
