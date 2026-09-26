"""THROWAWAY (2026-09-06) -- does a rate-conditional rotation into utilities (XLU)
on the `screen` supplier book help or hurt Sharpe? Rotate INTO XLU when the 6-month
change in the 10y real yield is negative (rates falling); funded from the supplier book."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket import hedges
from electrification_strategy import config, universe, backtest, hedge_overlay
from electrification_strategy.fred import fetch_series

START, END = "2017-06-01", "2026-08-31"; RF, AF = 0.04, 252
FAIR = ("2025-01-01", "2026-08-31")
SUB = {"2019-22": ("2019-01-01","2022-12-31"), "2023-26": ("2023-01-01","2026-08-31"),
       "COVID": ("2020-01-01","2020-06-30"), "fair(vs VOLT)": FAIR}

tk = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index)
            | {"SPY","XLU","GLD","IEF","VOLT"})
px = fetch_prices(tk, START, END)
spy = px["SPY"].pct_change()
cand = px[[c for c in tk if c in px.columns]]
screen_base = backtest.basket_returns(cand, "screen", START, END)      # raw EW supplier book
screen = backtest.compose(screen_base, spy, use_valuation=False)       # vol-targeted, no overlays
IDX = screen.index
xlu = px["XLU"].pct_change().reindex(IDX).fillna(0.0)
gld_ief = px[["GLD","IEF"]].pct_change().mean(axis=1).reindex(IDX).fillna(0.0)
dfii = fetch_series("DFII10","2003-01-01",END).reindex(pd.date_range(START,END)).ffill()

def mh(daily_bool):
    s = daily_bool.reindex(IDX).ffill().fillna(False)
    per = IDX.to_period("M"); me = s.groupby(per).last()
    return pd.Series(me.shift(1).reindex(per).to_numpy(), index=IDX).fillna(False).astype(bool)

d6 = (dfii - dfii.shift(126))
falling = mh((d6 < 0).shift(5).fillna(False))       # rates falling, 5d lag, month-held
rising  = mh((d6 > 0).shift(5).fillna(False))
print(f"rates-falling mask on {falling.mean()*100:.0f}% of days\n")

def rpt(label, r):
    def M(s):
        m = compute_metrics(s.dropna(), RF, AF); return m
    row = M(r)
    line = f"{label:<34} CAGR {row['cagr']*100:5.1f}%  Sh {row['sharpe']:.2f}  MaxDD {row['max_dd']*100:6.1f}%"
    for w,(a,b) in SUB.items():
        line += f"  | {w} Sh {M(r.loc[a:b])['sharpe']:+.2f}"
    print(line)

rpt("screen (no rotation)", screen)
# A) rotate INTO XLU when rates fall, funded from the supplier book
for w in (0.15, 0.30, 0.45):
    r = (1 - w*falling.astype(float))*screen + w*falling.astype(float)*xlu
    rpt(f"A) +{w:.2f} XLU when rates falling", r)
# B) graduated: w_xlu = clip(-k * 6m real-yield change, 0, 0.4)
kk = (-d6).reindex(IDX).ffill().clip(lower=0)
for k in (0.15, 0.30):
    w = (k*kk).clip(upper=0.40)
    w = pd.Series(mh(w > 0).astype(float).values, index=IDX) * w   # only when the sign persists a month
    w = w.reindex(IDX).fillna(0.0)
    r = (1-w)*screen + w*xlu
    rpt(f"B) graduated XLU (k={k})  avgW {w.mean():.2f}", r)
# C) for reference: the shipped GLD/IEF 15% sleeve (rate-agnostic), and XLU vs IEF swap
rpt("C) shipped: +15% GLD/IEF sleeve", 0.85*screen + 0.15*gld_ief)
rpt("C') +15% GLD/IEF + 0.30 XLU when falling", (0.85 - 0.30*falling.astype(float))*screen
    + 0.15*gld_ief + 0.30*falling.astype(float)*xlu)
# D) unconditional 30% XLU (what VOLT roughly does) -- the baseline to beat
rpt("D) unconditional 30% XLU", 0.70*screen + 0.30*xlu)

print("\nstandalone XLU:", end=" ")
rpt("XLU", xlu)
print("standalone screen supplier book vs SPY beta:",
      round(np.cov(pd.concat([screen,spy],axis=1).dropna().T)[0,1]/spy.reindex(IDX).var(),2))
