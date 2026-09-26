"""THROWAWAY feasibility gate for the short-leg-as-insurance probe (2026-09-06).
Confirms: (a) long-book proxy reconstructs 2018->2026, (b) all hedge tickers have
usable history, (c) FRED DFII10 + BAMLH0A0HYM2 are fetchable."""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.basket import simulate_basket
from grid_equipment_basket.backtest import compute_metrics

START, END = "2017-06-01", "2026-08-31"

LONGBOOK = ["ETN","HUBB","GEV","VRT","PWR","MYRG","NVT","FLNC","PRIM"]
HEDGES = {
  "SPY": ["SPY"],
  "firm_gen": ["VST","NRG","CEG","TLN"],
  "dc_reits": ["DLR","EQIX"],
  "der_sleeve": ["ENPH","SEDG","CHPT","RUN","BLNK","STEM"],
  "crowding": ["SMCI","VRT","DELL","ANET"],
  "gld_dur": ["GLD","SHY","IEF"],
}
XCHECK = ["PAVE","VOLT","TAN","QQQ","XLU"]

all_tk = sorted(set(LONGBOOK + [t for v in HEDGES.values() for t in v] + XCHECK))
px = fetch_prices(all_tk, START, END)
print("fetched columns:", sorted(px.columns))
print("\nper-ticker first / last valid date, n obs:")
for t in all_tk:
    if t in px.columns:
        s = px[t].dropna()
        print(f"  {t:6s} {s.index.min().date()}  ->  {s.index.max().date()}   n={len(s)}")
    else:
        print(f"  {t:6s} MISSING")

# long-book proxy: equal-weight, staggered inception, 25% cap, quarterly, 42d lag
lb_cols = [t for t in LONGBOOK if t in px.columns]
br = simulate_basket(px[lb_cols], START, END, lag_days=42, cap=0.25)
m = compute_metrics(br.returns)
print(f"\nLONG-BOOK PROXY 2017-06..2026-08: CAGR {m['cagr']*100:.1f}%  vol {m['ann_vol']*100:.1f}%  "
      f"Sharpe {m['sharpe']}  MaxDD {m['max_dd']*100:.1f}%  n={m['n_obs']}")
print("first return date:", br.returns.index.min().date(), " last:", br.returns.index.max().date())

# cross-check vs PAVE / VOLT
for b in ["PAVE","VOLT"]:
    if b in px.columns:
        r = px[b].pct_change().dropna()
        common = br.returns.index.intersection(r.index)
        c = br.returns.reindex(common).corr(r.reindex(common))
        print(f"  corr(long-book, {b}) over {common.min().date()}..{common.max().date()}: {c:.3f}")

# FRED
print("\nFRED fetch test:")
for sid in ["DFII10","BAMLH0A0HYM2"]:
    try:
        f = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                        parse_dates=[0], index_col=0)
        f = pd.to_numeric(f.iloc[:,0], errors="coerce").dropna()
        print(f"  {sid}: {f.index.min().date()} -> {f.index.max().date()}  n={len(f)}  last={f.iloc[-1]}")
    except Exception as e:
        print(f"  {sid}: FAILED  {type(e).__name__}: {e}")
