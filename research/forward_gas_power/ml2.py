"""All session buildout-heat signals -> (1) GBM/RF walk-forward OOS, (2) inverted
composite overlay ("fade the guidance/heat"), 2023-26 + 2018-26. n~43/~99 monthly.
Honest eval only: TimeSeriesSplit OOF rank-IC + R2 + permutation null; overlay judged
vs static AND vol_target_only, with active_days recorded (point-4 discipline)."""
import sys, warnings, json, urllib.request
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "/Users/sandhyapersad/acadia")
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
from grid_equipment_basket.data.prices import fetch_prices
from grid_equipment_basket.data.category_demand import fetch_category_demand, momentum
from grid_equipment_basket import config

KEY = open("/Users/sandhyapersad/acadia/.env").read().split("EIA_API_KEY=")[1].split()[0]
def mc(t, s="2009-06-01"):
    x = yf.Ticker(t).history(period="max")["Close"].dropna(); x.index = x.index.tz_localize(None)
    return x.resample("ME").last().loc[s:]
def eia(sid):
    u = f"https://api.eia.gov/v2/steo/data/?frequency=monthly&data[0]=value&facets[seriesId][]={sid}&length=5000&api_key={KEY}"
    d = json.loads(urllib.request.urlopen(u, timeout=40).read())["response"]["data"]
    return pd.Series({pd.Period(r["period"], "M").to_timestamp("M"): float(r["value"]) for r in d}).sort_index()

ngf, unl, ung = mc("NG=F"), mc("UNL"), mc("UNG")
smh, d10y = mc("SMH").pct_change(), mc("^TNX").diff()
pjw, erc = eia("ELWHU_PJ"), eia("ELWHU_TX")
CD = fetch_category_demand()
def zc(c, m):
    s = momentum(CD, c, m); z = (s - s.expanding(24).mean()) / s.expanding(24).std()
    z.index = z.index.to_period("M").to_timestamp("M"); return z

NAMES = sorted(config.UNIVERSE)
PX = fetch_prices(NAMES, "2016-01-01", "2026-08-31"); PX = PX[[c for c in NAMES if c in PX.columns]].ffill()
bret = (1 + PX.pct_change()).resample("ME").prod().mean(1) - 1.0
rf_m = config.RISK_FREE_RATE / 12

idx = pd.date_range("2016-01-31", "2026-08-31", freq="ME")
X = pd.DataFrame(index=idx)
for m in (3, 6, 12):
    X[f"ng_chg{m}"] = np.log(ngf).reindex(idx).diff(m)
    X[f"unl_chg{m}"] = np.log(unl).reindex(idx).diff(m)
    X[f"pjw_chg{m}"] = np.log(pjw).reindex(idx).diff(m)
    X[f"erc_chg{m}"] = np.log(erc).reindex(idx).diff(m)
X["unl_ema6"] = np.log(unl).reindex(idx).diff().ewm(span=6).mean() * 6
X["strip_slope"] = (unl / ung).reindex(idx); X["strip_slope_chg6"] = X["strip_slope"].diff(6)
X["sparkPJ_chg6"] = np.log((pjw / ngf).reindex(idx)).diff(6)
X["sparkERC_chg6"] = np.log((erc / ngf).reindex(idx)).diff(6)
for c in ("ppi_switchgear", "ppi_transformers", "ee_new_orders", "ee_unfilled_orders"):
    X[f"cat_{c}"] = zc(c, 6).reindex(idx).shift(2)
X["catscore"] = 0.4 * X["cat_ppi_switchgear"] + 0.3 * X["cat_ppi_transformers"] + 0.3 * X["cat_ee_new_orders"]
X["smh"] = smh.reindex(idx); X["d10y"] = d10y.reindex(idx)
X["b_mom3"] = bret.reindex(idx).rolling(3).sum()
FEATS = list(X.columns)
y1 = bret.reindex(idx).shift(-1)
WINS = {"2023-26": ("2023-01", "2026-08"), "2018-26": ("2018-06", "2026-08")}

def ml_oos(a, b, nperm=60):
    D = X.join(y1.rename("y")).loc[a:b].dropna()
    Xm, ym = D[FEATS].values, D["y"].values
    tscv = TimeSeriesSplit(n_splits=5)
    out = {}
    for nm, mk in [("GBM", lambda: GradientBoostingRegressor(n_estimators=120, max_depth=2,
                    learning_rate=0.03, subsample=0.8, min_samples_leaf=4, random_state=0)),
                   ("RF", lambda: RandomForestRegressor(n_estimators=200, max_depth=3,
                    min_samples_leaf=4, max_features=0.5, random_state=0, n_jobs=-1))]:
        def oof_ic(yv):
            of = np.full(len(yv), np.nan)
            for tr, te in tscv.split(Xm):
                mm = mk(); mm.fit(Xm[tr], yv[tr]); of[te] = mm.predict(Xm[te])
            k = ~np.isnan(of)
            return spearmanr(of[k], yv[k])[0], of, k
        ic, of, k = oof_ic(ym)
        r2 = 1 - np.sum((ym[k] - of[k])**2) / np.sum((ym[k] - ym[k].mean())**2)
        hit = np.mean(np.sign(of[k]) == np.sign(ym[k]))
        rng = np.random.default_rng(1)
        nul = np.array([oof_ic(rng.permutation(ym))[0] for _ in range(nperm)])
        out[nm] = (len(D), ic, r2, hit, np.mean(nul >= ic), of, k, D.index)
    return out, D

def stats(r):
    r = r.dropna()
    cg = (1 + r).prod()**(12 / len(r)) - 1
    sh = (r.mean() - rf_m) / r.std() * np.sqrt(12)
    dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    return cg, sh, (cg / abs(dd) if dd else np.nan)

print("=" * 92)
print(f"[1] KITCHEN-SINK ML  ({len(FEATS)} feats)  walk-forward TimeSeriesSplit OOF, target = basket fwd 1m")
for a, b in WINS.values():
    o, D = ml_oos(a, b)
    for nm, (n, ic, r2, hit, p, *_ ) in o.items():
        print(f"  {a[:4]}-{b[:4]} [{nm}] n={n:3}  OOF rank-IC={ic:+.3f}  R2={r2:+.3f}  "
              f"dir-acc={hit:.2f}  perm p(IC>=obs)={p:.2f}")

print("\n[2] INVERTED COMPOSITE OVERLAY  ('fade the buildout-heat')")
HEAT = ["ng_chg12", "unl_ema6", "catscore", "pjw_chg6", "sparkPJ_chg6"]
Z = ((X[HEAT] - X[HEAT].expanding(24).mean()) / X[HEAT].expanding(24).std())
comp = Z.mean(1).rename("heat")                      # high = buildout narrative loud
for a, b in WINS.values():
    sub = pd.concat([comp, y1.rename("y")], axis=1).loc[a:b].dropna()
    ic_raw = spearmanr(sub["heat"], sub["y"])[0]
    print(f"  {a[:4]}-{b[:4]}  rank-IC(heat, fwd-ret) = {ic_raw:+.3f}   "
          f"rank-IC(-heat, fwd-ret) = {-ic_raw:+.3f}   n={len(sub)}   "
          f"(negative raw IC => fading it helps)")
# overlay: exposure = clip(1 - beta*z(heat)), monthly, vs static & vol-target-only
vol = bret.rolling(6).std() * np.sqrt(12)
vt = (config.OVERLAY_TARGET_VOL / vol.reindex(idx)).clip(0.3, 1.5)
for beta in (0.20, 0.40):
    exp = (1 - beta * comp).clip(0.3, 1.5).shift(1)
    for a, b in WINS.values():
        base = bret.loc[a:b]
        ov = (exp.reindex(base.index) * base + (1 - exp.reindex(base.index)) * rf_m)
        vto = (vt.shift(1).reindex(base.index) * base + (1 - vt.shift(1).reindex(base.index)) * rf_m)
        cg, sh, cal = stats(ov); cg0, sh0, cal0 = stats(base); cgv, shv, calv = stats(vto)
        act = int(((exp.reindex(base.index) - 1).abs() > 0.02).sum())
        print(f"  b={beta} {a[:4]}-{b[:4]}: Sharpe ov {sh:+.2f} / static {sh0:+.2f} / voltgt {shv:+.2f}"
              f"   Calmar {cal:.2f} / {cal0:.2f} / {calv:.2f}   active_months={act}/{len(base)}")

print("\n[3] GBM OOF prediction AS an overlay signal (2018-26, uses walk-forward preds only)")
o, D = ml_oos("2018-06", "2026-08")
_, ic, _, _, _, of, k, didx = o["GBM"]
pred = pd.Series(of[k], index=didx[k])
pz = (pred - pred.expanding(12).mean()) / pred.expanding(12).std()
for beta in (0.25, 0.5):
    exp = (1 + beta * pz).clip(0.3, 1.5).shift(1)
    base = bret.reindex(pred.index)
    ov = exp * base + (1 - exp) * rf_m
    cg, sh, cal = stats(ov); cg0, sh0, cal0 = stats(base)
    print(f"  b={beta}: Sharpe ov {sh:+.2f} / static {sh0:+.2f}   Calmar {cal:.2f} / {cal0:.2f}")
