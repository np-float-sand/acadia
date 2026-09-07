import pandas as pd, numpy as np, glob, warnings
warnings.simplefilter("ignore")
A=252
P=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('/Users/sandhyapersad/acadia/grid_equipment_basket/data/cache/prices_*.parquet'))]+
            [pd.read_parquet('research/capex_cycle_pair_classifier/_cache/der_extra_prices.parquet')],axis=0)
P=P.groupby(level=0).last().sort_index().loc[:,lambda d:~d.columns.duplicated()].astype(float)
L=['ETN','HUBB','NVT','VRT','GEV','PWR','MYRG','PRIM','POWL','ATKR','WCC','ABBNY','SBGSY','PRYMY','HTHIY','NXT','FSLR']
S=['ENPH','SEDG','CHPT','RUN','BLNK','STEM','EVGO','WBX']
def ew(c,a,b):
    h=[x for x in c if x in P.columns and P[x].loc[a:b].notna().sum()>10]
    return P.loc[a:b,h].ffill().pct_change().clip(-.35,.35).mean(axis=1)

# PJM Base Residual Auction — RTO clearing price ($/MW-day), by (auction ~date, delivery year)
BRA=[("2007-04-01","2007/08",40.80),("2007-07-01","2008/09",111.92),("2007-10-01","2009/10",102.04),
     ("2008-01-01","2010/11",174.29),("2008-05-01","2011/12",110.00),("2009-05-01","2012/13",16.46),
     ("2010-05-01","2013/14",27.73),("2011-05-01","2014/15",125.99),("2012-05-01","2015/16",136.00),
     ("2013-05-01","2016/17",59.37),("2014-05-01","2017/18",120.00),("2015-08-01","2018/19",164.77),
     ("2016-05-01","2019/20",100.00),("2017-05-01","2020/21",76.53),("2018-05-01","2021/22",140.00),
     ("2021-06-01","2022/23",50.00),("2022-06-01","2023/24",34.13),("2022-12-01","2024/25",28.92),
     ("2024-07-01","2025/26",269.92),("2025-07-01","2026/27",329.17)]
bra=pd.DataFrame(BRA,columns=["auction_date","dy","price"]); bra["auction_date"]=pd.to_datetime(bra.auction_date)
bra["chg"]=bra.price.pct_change()
print(bra.to_string(index=False))

px_end=P.index[-1]
print("\n=== basket & long/short return AFTER each auction that falls in the price-data window ===")
for _,r in bra.iterrows():
    d0=r.auction_date
    if d0<pd.Timestamp("2019-07-01"): continue
    for horizon,lbl in [(6,"6m"),(12,"12m")]:
        d1=min(d0+pd.DateOffset(months=horizon),px_end)
        bk=(1+ew(L,d0,d1).fillna(0)).prod()-1
        ls=(1+(ew(L,d0,d1)-ew(S,d0,d1)).fillna(0)).prod()-1
        spy=(1+P['SPY'].loc[d0:d1].pct_change().fillna(0)).prod()-1
        print(f"  {d0.date()}  DY {r.dy}  price ${r.price:>6.2f} ({r.chg:+.0%})  ->  next {lbl}: basket {bk:+.0%} (vs SPY {spy:+.0%}), L/S {ls:+.0%}")

# where was the basket at the July-2024 10x auction, relative to its 2022-2026 run?
b=(1+ew(L,'2022-01-01','2026-08-31').fillna(0)).cumprod()
for d in ['2022-06-01','2022-12-01','2023-06-01','2024-01-01','2024-07-01','2025-01-01','2025-07-01','2026-08-31']:
    print(f"  basket cum since 2022-01 @ {d}: {b.asof(pd.Timestamp(d)):.2f}")
