"""Cross-sectional PRICE factors on a WIDE electrification/grid-supply universe
(~45 names). All prior work tilted within the marquee 9 and failed on breadth.
This tests standard documented factors (12-1 momentum, 1m reversal, 6m low-vol,
dist-from-52w-high) L/S and long-tilt, monthly, both sub-windows."""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "/Users/sandhyapersad/acadia")
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket import config

# wide universe: equipment + contractors/EPC + DER/storage supply side, liquid US listings
UNIV = sorted(set("""
ETN HUBB GEV VRT NVT POWL ATKR AYI ENS AZZ ROK EMR AME ITT PH DOV NVEE BMI
PWR MYRG PRIM MTZ EME IESC FIX DY GVA STRL ROAD ACM TPC
FLNC STEM SHLS ARRY NXT FSLR ENPH SEDG
NX BDC HAYW GNRC WCC CNM
""".split()))
ANN = 12

px = fetch_prices(UNIV, "2016-06-01", "2026-08-31")
px = px[[c for c in UNIV if c in px.columns]].sort_index()
have = px.columns[px.notna().sum() > 400].tolist()          # need enough history
px = px[have].ffill(limit=5)
mpx = px.resample("ME").last()
mret = mpx.pct_change()
print(f"universe: {len(have)} names with usable history ({', '.join(have)})")

# ── factors (computed at month t, using info through t) ──────────────────────
def zc(df):                                                  # cross-sectional z per row
    return df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)

mom = np.log(mpx).diff(12) - np.log(mpx).diff(1)             # 12-1 momentum
rev = -np.log(mpx).diff(1)                                    # 1m reversal (contrarian)
lowvol = -mret.rolling(6).std()                               # prefer low vol
d52 = mpx / mpx.rolling(12).max() - 1.0                       # distance from 12m high (>0 pref = momentum-ish)
mom6 = np.log(mpx).diff(6) - np.log(mpx).diff(1)

FAC = {"mom12_1": mom, "mom6_1": mom6, "reversal_1m": rev, "lowvol_6m": lowvol, "dist52w": d52}
WINS = {"2019-22": ("2019-01", "2022-12"), "2023-26": ("2023-01", "2026-08")}

def stats(r):
    r = r.dropna()
    if len(r) < 6: return (np.nan, np.nan, np.nan)
    cg = (1 + r).prod() ** (ANN / len(r)) - 1
    sh = r.mean() / r.std() * np.sqrt(ANN)
    dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    return cg, sh, dd

def portfolios(score):
    """score at t -> hold t+1. Returns (LS tercile, long-tilt) monthly return series."""
    s = score.shift(0)                                        # decided at t
    fwd = mret.shift(-1)                                      # realised t->t+1
    ls, lt = [], []
    for dt in s.index:
        row = s.loc[dt].dropna()
        if len(row) < 12: ls.append(np.nan); lt.append(np.nan); continue
        n = len(row); k = max(3, n // 3)
        top = row.nlargest(k).index; bot = row.nsmallest(k).index
        f = fwd.loc[dt]
        ls.append(f[top].mean() - f[bot].mean())
        w = pd.Series(1.0 / n, index=row.index)               # long-tilt: +50% top, -50% bot weight
        w[top] *= 1.5; w[bot] *= 0.5; w /= w.sum()
        lt.append((w * f.reindex(w.index)).sum())
    return (pd.Series(ls, index=s.index), pd.Series(lt, index=s.index))

ew = mret.mean(1)                                             # equal-weight universe benchmark
# marquee-9 benchmark
m9 = mret[[c for c in sorted(config.UNIVERSE) if c in mret.columns]].mean(1)

print("\n" + "=" * 84)
print("EQUAL-WEIGHT UNIVERSE (benchmark):")
for wl, (a, b) in WINS.items():
    cg, sh, dd = stats(ew.loc[a:b]); print(f"  {wl}: CAGR {cg*100:+.1f}%  Sharpe {sh:+.2f}  MaxDD {dd*100:.0f}%")
print("MARQUEE-9 EW (reference):")
for wl, (a, b) in WINS.items():
    cg, sh, dd = stats(m9.loc[a:b]); print(f"  {wl}: CAGR {cg*100:+.1f}%  Sharpe {sh:+.2f}  MaxDD {dd*100:.0f}%")

print("\n" + "=" * 84)
print("PER-FACTOR: monthly rank-IC (score_t vs fwd ret) + L/S tercile + long-tilt Sharpe")
for fn, fac in FAC.items():
    # rank IC
    fwd = mret.shift(-1)
    ics = {wl: [] for wl in WINS}
    for dt in fac.index:
        for wl, (a, b) in WINS.items():
            if not (pd.Timestamp(a) <= dt <= pd.Timestamp(b)): continue
            x = fac.loc[dt].dropna(); y = fwd.loc[dt].reindex(x.index).dropna()
            x = x.reindex(y.index)
            if len(y) >= 12 and x.std() > 0:
                ics[wl].append(np.corrcoef(x.rank(), y.rank())[0, 1])
    ls, lt = portfolios(fac)
    print(f"\n  {fn}")
    for wl, (a, b) in WINS.items():
        ic = np.array(ics[wl]); t = ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) > 2 else np.nan
        _, sls, dls = stats(ls.loc[a:b]); clt, slt, dlt = stats(lt.loc[a:b])
        print(f"    {wl}: rank-IC {ic.mean():+.3f} (t={t:+.2f})  |  L/S Sharpe {sls:+.2f} DD {dls*100:.0f}%"
              f"  |  long-tilt Sharpe {slt:+.2f} (EW {stats(ew.loc[a:b])[1]:+.2f})")

# ── composite: average z of momentum + lowvol - reversal ────────────────────
comp = zc(mom).add(zc(lowvol), fill_value=0).add(zc(mom6), fill_value=0)
ls, lt = portfolios(comp)
print("\n" + "=" * 84)
print("COMPOSITE (z: mom12_1 + mom6_1 + lowvol_6m)")
for wl, (a, b) in WINS.items():
    _, sls, dls = stats(ls.loc[a:b]); clt, slt, dlt = stats(lt.loc[a:b])
    print(f"  {wl}: L/S Sharpe {sls:+.2f} DD {dls*100:.0f}%  |  long-tilt Sharpe {slt:+.2f} CAGR {clt*100:+.0f}%"
          f"  vs EW Sharpe {stats(ew.loc[a:b])[1]:+.2f}")
