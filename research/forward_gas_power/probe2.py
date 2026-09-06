"""HHub probe round 2: UNL (12-month gas strip ETF = deseasonalised ~1yr-forward
gas, free, 2010->), smoothing variants, and a water-equity ETF (PHO) check."""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sandhyapersad/acadia")
import yfinance as yf
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket import config

ANN = 12
def mc(t, start="2009-06-01"):
    s = yf.Ticker(t).history(period="max")["Close"].dropna()
    s.index = s.index.tz_localize(None)
    return s.resample("ME").last().loc[start:]

unl = mc("UNL"); ung = mc("UNG"); pho = mc("PHO"); ngf = mc("NG=F")

NAMES = sorted(config.UNIVERSE)
PX = fetch_prices(NAMES, "2016-06-01", "2026-08-31")
PX = PX[[c for c in NAMES if c in PX.columns]].sort_index().ffill()
bmon = (1 + PX.pct_change()).resample("ME").prod().mean(1) - 1.0
smh = mc("SMH").pct_change(); tnx = mc("^TNX").diff()

sig = pd.DataFrame(index=unl.index)
# UNL = 12-month strip -> already deseasonalised; use level changes + smoothed
sig["unl_chg12"] = np.log(unl).diff(12)
sig["unl_chg6"]  = np.log(unl).diff(6)
sig["unl_chg3"]  = np.log(unl).diff(3)
unl_sm3 = unl.rolling(3).mean()
sig["unl_sm3_chg6"]  = np.log(unl_sm3).diff(6)
sig["unl_sm3_chg12"] = np.log(unl_sm3).diff(12)
sig["unl_ema"] = np.log(unl).diff().ewm(span=6).mean() * 6          # smoothed 6m-ish momentum
# term structure: 12-mo strip vs front month (UNL/UNG proxy for curve slope)
sig["strip_vs_front"] = (unl / unl.iloc[0]) / (ung / ung.iloc[0]) - 1.0
sig["strip_vs_front_chg6"] = sig["strip_vs_front"].diff(6)
# water equity
sig["pho_ret3"] = pho.pct_change(3)
sig["pho_ret6"] = pho.pct_change(6)

M = pd.concat([sig, bmon.rename("bret"), smh.rename("smh"), tnx.rename("d10y")], axis=1)
M["bmsmh"] = M["bret"] - M["smh"]
WINS = {"2019-22": ("2019-01", "2022-12"), "2023-26": ("2023-01", "2026-08")}

def ic(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 8: return np.nan, np.nan, int(ok.sum())
    r = np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1]
    return r, r * np.sqrt(ok.sum() - 1), int(ok.sum())

print("=" * 80)
print("HHUB PROBE 2 -- UNL 12-mo strip + smoothing + water equity")
print(f"UNL span {unl.index.min().date()}..{unl.index.max().date()}  "
      f"latest UNL {unl.iloc[-1]:.2f}  strip/front slope {sig['strip_vs_front'].iloc[-1]:+.1%}")

print("\n-- Spearman IC: signal_t vs basket_ret_{t+1}   r / t  (need |t|>=2 BOTH windows) --")
for s in sig.columns:
    row = f"  {s:20}"
    for wl, (a, b) in WINS.items():
        r, t, n = ic(M.loc[a:b, s], M.loc[a:b, "bret"].shift(-1))
        row += f"  {wl}: {r:+.2f}/{t:+.2f}(n{n})"
    rp, tp, _ = ic(M[s], M["bret"].shift(-1))
    row += f"  | pooled t{tp:+.2f}"
    print(row)

print("\n-- IC vs (basket - SMH)_{t+1} --")
for s in ["unl_chg12", "unl_sm3_chg12", "unl_ema", "strip_vs_front", "pho_ret6"]:
    row = f"  {s:20}"
    for wl, (a, b) in WINS.items():
        r, t, n = ic(M.loc[a:b, s], M.loc[a:b, "bmsmh"].shift(-1))
        row += f"  {wl}: {r:+.2f}/{t:+.2f}"
    print(row)

print("\n-- control OLS: bret_{t+1} ~ signal + d10y + smh   (signal coef t, need |t|>=1.5 both) --")
for s in ["unl_chg12", "unl_sm3_chg12", "unl_ema", "strip_vs_front"]:
    for wl, (a, b) in WINS.items():
        sub = M.loc[a:b].copy(); sub["y"] = sub["bret"].shift(-1)
        sub = sub[[s, "d10y", "smh", "y"]].dropna()
        if len(sub) < 12: print(f"  {s:18} {wl}: n<12"); continue
        X = np.column_stack([np.ones(len(sub)), sub[s], sub["d10y"], sub["smh"]])
        bta, *_ = np.linalg.lstsq(X, sub["y"].values, rcond=None)
        res = sub["y"].values - X @ bta
        se = np.sqrt((res @ res) / (len(sub) - 4) * np.linalg.inv(X.T @ X)[1, 1])
        print(f"  {s:18} {wl}: coef {bta[1]:+.3f}  t={bta[1]/se:+.2f}  n={len(sub)}")

print("\n-- PHO (water equity) correlation with basket monthly return --")
r, t, n = ic(M["pho_ret6"], M["bret"])
print(f"  contemporaneous rank-corr(PHO 6m ret, basket 1m ret) = {r:+.2f}  (contamination check)")
cc = M[["pho_ret3", "bret"]].dropna()
print(f"  corr(PHO 3m ret, basket 3m ret) = {cc['pho_ret3'].corr(bmon.rolling(3).apply(lambda x:(1+x).prod()-1).reindex(cc.index)):+.2f}")

print("\n-- scaler: exposure = clip(1 + b*z36(signal), 0.3, 1.5), monthly --")
def st(r):
    r = r.dropna()
    cg = (1+r).prod()**(ANN/len(r))-1; sh = r.mean()/r.std()*np.sqrt(ANN)
    dd = ((1+r).cumprod()/(1+r).cumprod().cummax()-1).min()
    return cg, sh, cg/abs(dd) if dd else np.nan
for s in ["unl_chg12", "unl_sm3_chg12", "unl_ema", "strip_vs_front"]:
    z = (M[s] - M[s].rolling(36, min_periods=18).mean()) / M[s].rolling(36, min_periods=18).std()
    for bc in (0.15, 0.30):
        sc = (1 + bc*z).clip(0.3, 1.5).shift(1) * M["bret"]
        row = f"  {s:18} b={bc}"
        for wl, (a, b) in WINS.items():
            cg, sh, cal = st(sc.loc[a:b]); cg0, sh0, cal0 = st(M["bret"].loc[a:b])
            row += f"  {wl}: Sh {sh:+.2f}/{sh0:+.2f} Cal {cal:.2f}/{cal0:.2f}"
        print(row)
