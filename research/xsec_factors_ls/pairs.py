"""Cointegration pairs stat-arb on the electrification universe. Market-neutral,
no theme/timing signal. Train window selects pairs (Engle-Granger + half-life),
trade the hedge-ratio spread z-score OOS. Walk-forward, costs, both regimes."""
from __future__ import annotations
import sys, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "/Users/sandhyapersad/acadia")
from grid_equipment_basket.data.prices import fetch_prices

UNIV = sorted(set("""
ETN HUBB GEV VRT NVT POWL ATKR AYI ENS AZZ ROK EMR AME ITT PH DOV BMI
PWR MYRG PRIM MTZ EME IESC FIX DY GVA STRL ROAD ACM TPC
FLNC STEM SHLS ARRY NXT FSLR ENPH SEDG NX BDC HAYW GNRC WCC CNM
""".split()))
ANN = 252
px = fetch_prices(UNIV, "2015-06-01", "2026-08-31")
px = px[[c for c in UNIV if c in px.columns]].sort_index()
px = px[px.columns[px.notna().sum() > 1500]].ffill(limit=3).dropna()
lp = np.log(px)
print(f"{px.shape[1]} names, {px.index[0].date()}..{px.index[-1].date()}")

def adf_pval(x):
    """cheap ADF: regress dx on x_lag + const, t-stat of x_lag vs MacKinnon-ish crit."""
    x = np.asarray(x); dx = np.diff(x); xl = x[:-1]
    X = np.column_stack([np.ones(len(xl)), xl])
    b, *_ = np.linalg.lstsq(X, dx, rcond=None)
    res = dx - X @ b
    s2 = res @ res / (len(dx) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    tstat = b[1] / se
    return tstat                                   # more negative = more stationary; < -2.9 ~ 5%

def half_life(spread):
    s = np.asarray(spread); ds = np.diff(s); sl = s[:-1]
    b = np.polyfit(sl, ds, 1)[0]
    return -np.log(2) / b if b < 0 else np.inf

def select_pairs(train_lp, max_pairs=25):
    cols = train_lp.columns; out = []
    for a, c in itertools.combinations(cols, 2):
        ya, yc = train_lp[a].values, train_lp[c].values
        beta = np.polyfit(yc, ya, 1)[0]
        if beta <= 0: continue
        spr = ya - beta * yc
        t = adf_pval(spr); hl = half_life(spr)
        if t < -3.2 and 5 < hl < 60:
            out.append((a, c, beta, t, hl))
    out.sort(key=lambda r: r[3])
    return out[:max_pairs]

def bt_pairs(pairs, lp_oos, entry=2.0, exit_=0.3, stop=3.5, lb=60, cost_bps=10):
    rets = []
    for a, c, beta, *_ in pairs:
        ya, yc = lp_oos[a], lp_oos[c]
        spr = ya - beta * yc
        z = (spr - spr.rolling(lb).mean()) / spr.rolling(lb).std()
        pos = pd.Series(0.0, index=spr.index); cur = 0.0
        for i in range(len(z)):
            zz = z.iloc[i]
            if np.isnan(zz): continue
            if cur == 0.0:
                if zz > entry: cur = -1.0
                elif zz < -entry: cur = 1.0
            else:
                if abs(zz) < exit_ or abs(zz) > stop or np.sign(zz) == np.sign(-cur) * -1 and abs(zz) < exit_:
                    cur = 0.0
                elif abs(zz) > stop:
                    cur = 0.0
            pos.iloc[i] = cur
        dspr = spr.diff()                             # spread P&L per unit
        gross = (pos.shift(1) * dspr).fillna(0.0)
        trades = pos.diff().abs().fillna(0.0)
        net = gross - trades * (cost_bps / 1e4)
        rets.append(net)
    R = pd.concat(rets, axis=1).mean(1)               # equal-weight across active pairs
    return R

def st(r):
    r = r.dropna()
    if r.std() == 0 or len(r) < 20: return (np.nan, np.nan, np.nan)
    sh = r.mean() / r.std() * np.sqrt(ANN)
    eq = r.cumsum(); dd = (eq - eq.cummax()).min()
    return sh, r.mean() * ANN, dd

# walk-forward: 2yr train -> 1yr trade, roll
print("\nWALK-FORWARD cointegration pairs (2y train / 1y OOS trade, roll):")
starts = pd.date_range("2017-01-01", "2025-01-01", freq="12MS")
allret = []
for s in starts:
    tr0, tr1 = s - pd.DateOffset(years=2), s
    oos1 = s + pd.DateOffset(years=1)
    tl = lp.loc[tr0:tr1]
    if len(tl) < 400: continue
    pairs = select_pairs(tl)
    if not pairs:
        print(f"  {s.date()}  no pairs"); continue
    R = bt_pairs(pairs, lp.loc[tr1 - pd.DateOffset(days=90):oos1])
    R = R.loc[tr1:oos1]
    sh, ann, dd = st(R)
    allret.append(R)
    print(f"  trade {s.date()}..{oos1.date()}  {len(pairs):2} pairs  OOS Sharpe {sh:+.2f}  annret {ann*100:+.1f}%")

full = pd.concat(allret).sort_index()
full = full[~full.index.duplicated()]
for wl, (a, b) in {"2019-22": ("2019-01", "2022-12"), "2023-26": ("2023-01", "2026-08"),
                   "full OOS": ("2017-01", "2026-08")}.items():
    sh, ann, dd = st(full.loc[a:b])
    print(f"  [{wl}] stitched OOS Sharpe {sh:+.2f}  annret {ann*100:+.1f}%  maxDD {dd*100:.1f}%")
