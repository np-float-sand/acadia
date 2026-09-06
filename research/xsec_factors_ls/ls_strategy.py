"""Reproducible rules-based long/short electrification strategy.

Universe rule: names appearing in the published holdings of the electrification
ETFs {VOLT, ELFY, ZAP, GRID} (top holdings observed 2026-09 + their obvious
equipment/EPC and DER constituents). Split by a stated rule:
  LONG  = GICS Electrical Equipment / Construction & Engineering pure-plays
          (the picks-and-shovels of the buildout)
  SHORT = the residential-solar + EV-charging + behind-the-meter DER sleeve
Construction: EW each leg, 25% single-name cap, beta-hedge long->short over a
trailing window, 20% vol target on the spread (1.5x cap), quarterly rebalance,
6-week reporting lag. Pre-registered: both sub-windows (2019-22 pre-AI, 2023-26 AI).
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "/Users/sandhyapersad/acadia")
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket import config

LONG = sorted(set("""
ETN HUBB NVT POWL GEV VRT AYI AME BELFB APH ROK EMR ITT PH DOV BMI NX ENS ATKR
PWR MYRG PRIM MTZ EME FIX IESC DY GVA STRL ACM WCC HAYW GNRC
""".split()))
SHORT = sorted(set("""
ENPH SEDG RUN CHPT BE STEM SHLS ARRY NXT FLNC BLNK EVGO
""".split()))
ANN = 252

allpx = fetch_prices(LONG + SHORT + ["SPY"], "2016-06-01", "2026-08-31")
allpx = allpx.sort_index().ffill(limit=3)
L = [t for t in LONG if t in allpx and allpx[t].notna().sum() > 400]
S = [t for t in SHORT if t in allpx and allpx[t].notna().sum() > 400]
print(f"LONG  ({len(L)}): {' '.join(L)}")
print(f"SHORT ({len(S)}): {' '.join(S)}")
ret = allpx.pct_change()

def leg(names):
    r = ret[names]
    return r.mean(1)                                   # daily EW (cap ~irrelevant EW; 25% cap only binds <4 names)

lr, sr, spy = leg(L), leg(S), ret["SPY"]

def rolling_beta(a, b, win=126):
    cov = a.rolling(win).cov(b); var = b.rolling(win).var()
    return (cov / var).clip(0.3, 2.5)

beta = rolling_beta(lr, sr).shift(1).fillna(1.0)       # hedge ratio long->short, lagged
spread_raw = lr - beta * sr                            # $-neutral-ish, beta-hedged
# vol target on the spread
rv = spread_raw.rolling(21).std() * np.sqrt(ANN)
vscale = (config.OVERLAY_TARGET_VOL / rv).clip(0.0, 1.5).shift(1).fillna(1.0)
spread = vscale * spread_raw
rf_d = config.RISK_FREE_RATE / ANN
spread_tr = spread + (1 - vscale) * rf_d               # uninvested slack earns rf

# long-only variants for reference
lo = lr                                                 # plain EW long
# marquee-9
m9 = ret[[t for t in sorted(config.UNIVERSE) if t in ret]].mean(1)

def stats(d):
    d = d.dropna()
    if len(d) < 40: return dict(cagr=np.nan, sharpe=np.nan, maxdd=np.nan, spyb=np.nan)
    cagr = (1 + d).prod() ** (ANN / len(d)) - 1
    sh = (d.mean() - rf_d) / d.std() * np.sqrt(ANN)
    eq = (1 + d).cumprod(); dd = (eq / eq.cummax() - 1).min()
    b = np.polyfit(spy.reindex(d.index).fillna(0), d, 1)[0]
    return dict(cagr=cagr, sharpe=sh, maxdd=dd, spyb=b)

WINS = {"2017-18": ("2017-01-01", "2018-12-31"),
        "2019-22 (pre-AI)": ("2019-01-01", "2022-12-31"),
        "2023-26 (AI)": ("2023-01-01", "2026-08-31"),
        "2023-24": ("2023-01-01", "2024-12-31"),
        "2025-26": ("2025-01-01", "2026-08-31"),
        "full 2017-26": ("2017-01-01", "2026-08-31")}

def show(name, series):
    print(f"\n{name}")
    print(f"  {'window':18}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'SPYbeta':>9}")
    for wl, (a, b) in WINS.items():
        m = stats(series.loc[a:b])
        print(f"  {wl:18}{m['cagr']*100:>8.1f}%{m['sharpe']:>9.2f}{m['maxdd']*100:>8.1f}%{m['spyb']:>9.2f}")

print("\n" + "=" * 72)
show("LONG/SHORT  (EW equipment+EPC  /  beta-hedged short DER sleeve  /  20% vol-tgt)", spread_tr)
show("LONG-ONLY EW equipment+EPC (reference)", lo)
show("MARQUEE-9 EW (reference)", m9)
show("SHORT LEG alone (DER sleeve EW, sign flipped = short)", -sr)

# turnover / rebalance sensitivity: monthly vs quarterly hold of the vol scalar & beta
mspread = (1 + spread_tr).resample("ME").prod() - 1
print("\nmonthly spread return summary 2023-26:  "
      f"mean {mspread.loc['2023':'2026'].mean()*100:+.2f}%  "
      f"hit {(mspread.loc['2023':'2026']>0).mean()*100:.0f}%  "
      f"worst {mspread.loc['2023':'2026'].min()*100:.1f}%")

# correlation of legs (the structural problem)
print(f"\ncorr(long EW, short-sleeve EW) monthly:  "
      f"2019-22 {(1+lr).resample('ME').prod().pct_change().corr((1+sr).resample('ME').prod().pct_change()):.2f}  "
      f"2023-26 {((1+lr).resample('ME').prod().pct_change().loc['2023':]).corr((1+sr).resample('ME').prod().pct_change().loc['2023':]):.2f}")
