"""Category-demand -> maker-rotation probe. Throwaway until it clears the bar.
docs/handoff_2026-09-03-category-demand-rotation.md."""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/sandhyapersad/acadia")

from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.data.category_demand import fetch_category_demand, momentum
from grid_equipment_basket import config

NAMES = sorted(config.UNIVERSE)              # ETN FLNC GEV HUBB MYRG NVT PRIM PWR VRT
ANN = 252

# ── category -> name exposure matrix (hand-built from 10-K segment mix, 2026) ──
# rows sum to 1 across {transformers, switchgear, energy_wire, broad_EE,
# TD_construction, unmapped}. "unmapped" = gas turbines/wind (GEV), DC cooling &
# services (VRT), storage (FLNC), solar EPC & pipeline (PRIM), C&I build (MYRG).
CAT = ["transformers", "switchgear", "energy_wire", "broad_EE", "TD_construction", "unmapped"]
EXPO = pd.DataFrame({
    "ETN":  [0.05, 0.40, 0.00, 0.45, 0.05, 0.05],
    "HUBB": [0.15, 0.15, 0.20, 0.35, 0.10, 0.05],
    "GEV":  [0.20, 0.12, 0.03, 0.15, 0.05, 0.45],
    "VRT":  [0.00, 0.20, 0.05, 0.20, 0.00, 0.55],
    "NVT":  [0.00, 0.10, 0.15, 0.70, 0.05, 0.00],
    "PWR":  [0.00, 0.00, 0.05, 0.03, 0.87, 0.05],
    "MYRG": [0.00, 0.00, 0.05, 0.05, 0.75, 0.15],
    "PRIM": [0.00, 0.00, 0.05, 0.05, 0.55, 0.35],
    "FLNC": [0.00, 0.00, 0.00, 0.25, 0.05, 0.70],
}, index=CAT)[NAMES].T                       # -> index=names, cols=categories
assert np.allclose(EXPO.sum(1), 1.0)

# ── monthly category demand-momentum panel ───────────────────────────────────
CD = fetch_category_demand()

def zmom(col, m):
    s = momentum(CD, col, m)
    return (s - s.expanding(24).mean()) / s.expanding(24).std()

def cat_z_panel(lb: int) -> pd.DataFrame:
    """monthly z-scored demand momentum per category (unmapped = 0)."""
    p = pd.DataFrame(index=CD.index)
    p["transformers"]    = zmom("ppi_transformers", lb)
    p["switchgear"]      = zmom("ppi_switchgear", lb)
    p["energy_wire"]     = zmom("ppi_energy_wire", lb)
    p["broad_EE"]        = zmom("ee_new_orders", lb)
    p["TD_construction"] = zmom("ee_unfilled_orders", lb)
    p["unmapped"]        = 0.0
    return p[CAT]

# ── prices / returns ────────────────────────────────────────────────────────
PX = fetch_prices(NAMES, "2018-06-01", "2026-08-31")
PX = PX[[c for c in NAMES if c in PX.columns]].sort_index().ffill()
RET = PX.pct_change().dropna(how="all")
MRET = (1 + RET).groupby(RET.index.to_period("M")).prod() - 1.0   # monthly name returns
MIDX = MRET.index

def name_scores(lb: int) -> pd.DataFrame:
    """monthly per-name demand score = EXPO @ category_z, lagged 2 months (M3 pub lag)."""
    cz = cat_z_panel(lb).reindex(pd.period_range(CD.index.min().to_period("M"),
                                                 MIDX.max(), freq="M").to_timestamp("M"))
    cz.index = cz.index.to_period("M")
    sc = cz.fillna(0.0) @ EXPO.T                 # index=month(P), cols=names
    sc = sc.shift(2)                             # data for month M usable from M+2
    return sc.reindex(MIDX).fillna(0.0)

def tilt_weights(sc_row: pd.Series, k: float, lock=None) -> pd.Series:
    """equal-weight + k * cross-sectionally-demeaned score, clipped >=0, renorm.
    `lock` = iterable of names pinned to 1/9 before renorm of the rest."""
    s = sc_row - sc_row.mean()
    w = pd.Series(1.0 / len(NAMES), index=NAMES) + k * s
    if lock:
        free = [n for n in NAMES if n not in lock]
        w[list(lock)] = 1.0 / len(NAMES)
        rest = max(1e-9, 1.0 - w[list(lock)].sum())
        wf = w[free].clip(lower=0)
        w[free] = wf / wf.sum() * rest
    else:
        w = w.clip(lower=0)
        w = w / w.sum()
    return w

def backtest(lb: int, k: float, lock=None):
    sc = name_scores(lb)
    W = pd.DataFrame({m: tilt_weights(sc.loc[m], k, lock) for m in MIDX}).T
    tilt_m = (W * MRET.reindex(W.index)).sum(1)
    ew_m = MRET.reindex(W.index).mean(1)
    turnover = W.diff().abs().sum(1).mean() * 12          # annualized 1-way
    return tilt_m, ew_m, W, turnover

def stats(m):                                             # monthly series -> annualised
    r = m.dropna()
    cagr = (1 + r).prod() ** (12 / len(r)) - 1
    sh = r.mean() / r.std() * np.sqrt(12)
    dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    return cagr, sh, dd

WINS = {"2019-22 (pre-AI)": ("2019-01", "2022-12"), "2023-26 (AI)": ("2023-01", "2026-08")}

print("=" * 74)
print("CATEGORY-DEMAND -> MAKER-ROTATION PROBE")
print("exposure matrix (rows=names):"); print(EXPO.round(2).to_string())
print(f"\nunmapped share of book: {EXPO['unmapped'].mean():.0%} avg, "
      f"GEV {EXPO.loc['GEV','unmapped']:.0%} VRT {EXPO.loc['VRT','unmapped']:.0%} "
      f"FLNC {EXPO.loc['FLNC','unmapped']:.0%}")

# ---- Signal A: full composite, plateau ----
print("\n" + "=" * 74)
print("SIGNAL A -- composite tilt vs equal-weight (Sharpe; plateau over lb x k)")
for lb in (3, 6, 9, 12):
    row = []
    for k in (0.03, 0.06):
        tm, em, W, tovr = backtest(lb, k)
        cells = []
        for wl, (a, b) in WINS.items():
            st_t = stats(tm.loc[a:b])[1]; st_e = stats(em.loc[a:b])[1]
            cells.append(f"{st_t:+.2f}/{st_e:+.2f}")
        row.append(f"k={k}: " + " | ".join(cells) + f"  tovr={tovr:.1f}x")
    print(f"  lb={lb:2}mo  " + "   ".join(row))
print("  (tilt Sharpe / EW Sharpe, per window)")

# ---- headline config lb=6,k=0.06 detail ----
lb, k = 6, 0.06
tm, em, W, tovr = backtest(lb, k)
print(f"\nheadline lb={lb} k={k}:  turnover {tovr:.2f}x/yr")
print(f"  {'window':18}{'CAGR t/e':>18}{'Sharpe t/e':>16}{'MaxDD t/e':>18}")
for wl, (a, b) in WINS.items():
    ct, sht, ddt = stats(tm.loc[a:b]); ce, she, dde = stats(em.loc[a:b])
    print(f"  {wl:18}{ct*100:>8.1f}/{ce*100:>6.1f}%{sht:>9.2f}/{she:>5.2f}{ddt*100:>10.1f}/{dde*100:>6.1f}%")
print("  avg tilt weights (headline):")
print("   " + W.mean().round(3).to_string().replace("\n", "\n   "))

# ---- pre-registered check 2: rank-IC of score vs fwd return ----
sc = name_scores(lb)
fwd = MRET.shift(-1).reindex(sc.index)
ics = []
for m in sc.index:
    a, b = sc.loc[m], fwd.loc[m]
    ok = a.notna() & b.notna()
    if ok.sum() >= 5 and a[ok].std() > 0:
        ics.append(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1])
ics = pd.Series(ics)
print(f"\nCHECK 2 rank-IC (score vs fwd-1m ret): mean {ics.mean():+.3f}  "
      f"t={ics.mean()/ics.std()*np.sqrt(len(ics)):+.2f}  n={len(ics)}")

# ---- pre-registered check 3: low-vol confound ----
vol = RET.rolling(63).std()
def lmh_daily():
    out = []
    for d in RET.index:
        v = vol.loc[:d].iloc[-1].dropna()
        if len(v) < 6:
            out.append(0.0); continue
        lo = v.nsmallest(3).index; hi = v.nlargest(3).index
        out.append(RET.loc[d, lo].mean() - RET.loc[d, hi].mean())
    return pd.Series(out, index=RET.index)
LMH = lmh_daily()
# rebuild tilt daily
scM = name_scores(lb); scM.index = scM.index.to_timestamp("M")
Wd = scM.reindex(RET.index, method="ffill").apply(lambda r: tilt_weights(r, k), axis=1)
tilt_d = (Wd * RET).sum(1)
ew_d = RET.mean(1)
exc = (tilt_d - ew_d).dropna()
X = pd.concat([LMH.reindex(exc.index), ew_d.reindex(exc.index)], axis=1).fillna(0.0)
X.columns = ["LMH", "MKT"]
import numpy.linalg as la
Xm = np.column_stack([np.ones(len(X)), X.values])
beta, *_ = la.lstsq(Xm, exc.values, rcond=None)
resid = exc.values - Xm @ beta
se = np.sqrt((resid @ resid) / (len(exc) - Xm.shape[1]) * la.inv(Xm.T @ Xm)[0, 0])
print(f"CHECK 3 low-vol confound: excess-ret alpha t={beta[0]/se:+.2f} "
      f"(LMH beta {beta[1]:+.2f})  [need |t|>=1.5 to survive]")

# ---- pre-registered check 4: force VRT/NVT/FLNC to EW ----
print("\nCHECK 4 -- force VRT/NVT/FLNC to 1/9, rotate only the other 6:")
tm4, em4, W4, tovr4 = backtest(lb, k, lock=("VRT", "NVT", "FLNC"))
for wl, (a, b) in WINS.items():
    print(f"  {wl:18} tilt Sharpe {stats(tm4.loc[a:b])[1]:+.2f}  vs EW {stats(em4.loc[a:b])[1]:+.2f}")
