"""Henry Hub forward-gas signal probe for the grid-equipment basket.
Throwaway. True long-dated power/gas forwards w/ 2017-26 history are paid
(yfinance drops expired dated contracts; EIA futures stop at contract 4;
STEO needs the vintage archive) -- so this tests the free near-curve proxies:
NG front-month trend + the RNGC4/RNGC1 term-structure slope."""
from __future__ import annotations
import io, os, sys, urllib.request, urllib.parse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sandhyapersad/acadia")
import yfinance as yf
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket import config

ANN = 12
KEY = open("/Users/sandhyapersad/acadia/.env").read().split("EIA_API_KEY=")[1].split()[0]

def eia_ng_futures(series, start="2005-01", end="2026-08"):
    url = ("https://api.eia.gov/v2/natural-gas/pri/fut/data/?frequency=monthly"
           "&data[0]=value&facets[series][]=" + series +
           f"&start={start}&end={end}&sort[0][column]=period&sort[0][direction]=asc"
           "&offset=0&length=5000&api_key=" + KEY)
    d = __import__("json").loads(urllib.request.urlopen(url, timeout=40).read())
    r = pd.DataFrame(d["response"]["data"])
    s = pd.Series(pd.to_numeric(r["value"]).to_numpy(),
                  index=pd.PeriodIndex(r["period"], freq="M").to_timestamp("M"), name=series)
    return s[~s.index.duplicated()]

# ── gas series (monthly) ────────────────────────────────────────────────────
ngf = yf.Ticker("NG=F").history(period="max")["Close"].dropna()
ngf.index = ngf.index.tz_localize(None)
ng_front = ngf.resample("ME").last().rename("ng_front")
rngc1 = eia_ng_futures("RNGC1"); rngc4 = eia_ng_futures("RNGC4")

G = pd.concat([ng_front, rngc1, rngc4], axis=1).dropna(how="all")
G["contango_4m"] = G["RNGC4"] / G["RNGC1"] - 1.0          # >0 = upward-sloping near curve
sig = pd.DataFrame(index=G.index)
sig["ng_chg12"] = np.log(G["ng_front"]).diff(12)          # structural level trend (deseasonalised)
sig["ng_chg6"]  = np.log(G["ng_front"]).diff(6)
sig["ng_chg3"]  = np.log(G["ng_front"]).diff(3)
sig["contango"] = G["contango_4m"]
sig["contango_chg3"] = G["contango_4m"].diff(3)

# ── basket / controls (monthly) ────────────────────────────────────────────
NAMES = sorted(config.UNIVERSE)
PX = fetch_prices(NAMES, "2016-06-01", "2026-08-31")
PX = PX[[c for c in NAMES if c in PX.columns]].sort_index().ffill()
bmon = (1 + PX.pct_change()).resample("ME").prod().mean(1) - 1.0
smh = yf.Ticker("SMH").history(period="max")["Close"].dropna(); smh.index = smh.index.tz_localize(None)
smh_m = smh.resample("ME").last().pct_change()
tnx = yf.Ticker("^TNX").history(period="max")["Close"].dropna(); tnx.index = tnx.index.tz_localize(None)
d10y = tnx.resample("ME").last().diff()

M = pd.concat([sig, bmon.rename("bret"), (bmon - smh_m).rename("bmsmh"),
               smh_m.rename("smh"), d10y.rename("d10y")], axis=1)
WINS = {"2019-22": ("2019-01", "2022-12"), "2023-26": ("2023-01", "2026-08")}

def sp_ic(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 8: return np.nan, np.nan, int(ok.sum())
    xr, yr = x[ok].rank(), y[ok].rank()
    r = np.corrcoef(xr, yr)[0, 1]
    return r, r * np.sqrt(ok.sum() - 1), int(ok.sum())

print("=" * 78)
print("HENRY HUB FORWARD-GAS PROBE (free near-curve proxies; true long end is paid)")
print(f"gas series span: {G.index.min().date()} -> {G.index.max().date()}")
print(f"latest: NG front {G['ng_front'].iloc[-1]:.2f}  RNGC1 {G['RNGC1'].iloc[-1]:.2f}  "
      f"RNGC4 {G['RNGC4'].iloc[-1]:.2f}  4m-contango {G['contango_4m'].iloc[-1]:+.1%}")

print("\n-- Spearman IC: signal_t vs basket_return_{t+1}  (r / t / n) --")
for s in ["ng_chg12", "ng_chg6", "ng_chg3", "contango", "contango_chg3"]:
    line = f"  {s:14}"
    for wl, (a, b) in WINS.items():
        sub = M.loc[a:b]
        r, t, n = sp_ic(sub[s], sub["bret"].shift(-1))
        line += f"  {wl}: {r:+.2f}/{t:+.2f}(n{n})"
    rp, tp, npd = sp_ic(M[s], M["bret"].shift(-1))
    line += f"  | pooled {rp:+.2f}/t{tp:+.2f}"
    print(line)

print("\n-- Spearman IC vs (basket - SMH)_{t+1} --")
for s in ["ng_chg12", "ng_chg6", "contango", "contango_chg3"]:
    line = f"  {s:14}"
    for wl, (a, b) in WINS.items():
        sub = M.loc[a:b]
        r, t, n = sp_ic(sub[s], sub["bmsmh"].shift(-1))
        line += f"  {wl}: {r:+.2f}/{t:+.2f}"
    print(line)

print("\n-- lead/lag (ng_chg12 vs bret shifted k; k<0 = gas leads) --")
for k in (-3, -2, -1, 0, 1, 2, 3):
    r, t, n = sp_ic(M["ng_chg12"], M["bret"].shift(-k))
    print(f"  k={k:+d}: r={r:+.2f} t={t:+.2f} n={n}")

print("\n-- control: OLS  bret_{t+1} ~ signal_t + d10y_t + smh_t  (signal coef t) --")
for s in ["ng_chg12", "ng_chg6", "contango", "contango_chg3"]:
    for wl, (a, b) in WINS.items():
        sub = M.loc[a:b].copy(); sub["y"] = sub["bret"].shift(-1)
        sub = sub[[s, "d10y", "smh", "y"]].dropna()
        if len(sub) < 12:
            print(f"  {s:14} {wl}: n<12"); continue
        X = np.column_stack([np.ones(len(sub)), sub[s], sub["d10y"], sub["smh"]])
        bta, *_ = np.linalg.lstsq(X, sub["y"].values, rcond=None)
        res = sub["y"].values - X @ bta
        se = np.sqrt((res @ res) / (len(sub) - 4) * np.linalg.inv(X.T @ X)[1, 1])
        print(f"  {s:14} {wl}: signal_coef={bta[1]:+.3f}  t={bta[1]/se:+.2f}  n={len(sub)}")

print("\n-- scaler backtest: exposure = clip(1 + b*z(signal), 0.3, 1.5), monthly --")
def stats(r):
    r = r.dropna()
    return ((1+r).prod()**(ANN/len(r))-1, r.mean()/r.std()*np.sqrt(ANN),
            ((1+r).cumprod()/(1+r).cumprod().cummax()-1).min())
for s in ["ng_chg12", "ng_chg6", "contango"]:
    z = (M[s] - M[s].expanding(18).mean()) / M[s].expanding(18).std()
    for bcoef in (0.15, 0.30):
        exp = (1 + bcoef * z).clip(0.3, 1.5).shift(1)
        scaled = exp * M["bret"]
        line = f"  {s:12} b={bcoef}"
        for wl, (a, b) in WINS.items():
            cg, sh, dd = stats(scaled.loc[a:b]); cg0, sh0, dd0 = stats(M["bret"].loc[a:b])
            cal = cg / abs(dd) if dd else np.nan; cal0 = cg0 / abs(dd0) if dd0 else np.nan
            line += f"  {wl}: Sh {sh:+.2f}/{sh0:+.2f} Cal {cal:.2f}/{cal0:.2f}"
        print(line)
