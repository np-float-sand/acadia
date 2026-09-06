"""Split 2023-26 into IS/OOS halves (same AI regime) -- does the 2023-26 OOF
rank-IC survive a clean held-out slice, or was it overfitting the 43-mo draw?
Protocol A: train 2023-01..2024-12, test 2025-01..2026-08.
Protocol B: expanding walk-forward within 2023-26, min-train 18mo.
Models: GBM, RF, Ridge. Feature sets: FULL(25) and LEAN(6). Shuffle null on the holdout."""
import sys, warnings, json, urllib.request
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, "/Users/sandhyapersad/acadia")
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

FULL = list(X.columns)
LEAN = ["pjw_chg6", "sparkPJ_chg6", "unl_ema6", "catscore", "ng_chg12", "d10y"]
y1 = bret.reindex(idx).shift(-1)

def models():
    return {
        "GBM":   GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.03,
                    subsample=0.8, min_samples_leaf=4, random_state=0),
        "RF":    RandomForestRegressor(n_estimators=250, max_depth=3, min_samples_leaf=4,
                    max_features=0.5, random_state=0, n_jobs=-1),
        "Ridge": Ridge(alpha=3.0),
    }

def fit_pred(mdl, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    mdl.fit(sc.transform(Xtr), ytr)
    return mdl.predict(sc.transform(Xte))

def stats(r):
    r = r.dropna()
    cg = (1 + r).prod()**(12 / len(r)) - 1
    sh = (r.mean() - rf_m) / r.std() * np.sqrt(12)
    dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    return sh, (cg / abs(dd) if dd else np.nan)

def holdout(feats, tr_a, tr_b, te_a, te_b, label, nperm=150):
    D = X[feats].join(y1.rename("y")).dropna()
    tr = D.loc[tr_a:tr_b]; te = D.loc[te_a:te_b]
    print(f"\n  {label}  train {tr.index[0].date()}..{tr.index[-1].date()} (n={len(tr)})  "
          f"test {te.index[0].date()}..{te.index[-1].date()} (n={len(te)})")
    for nm, mdl in models().items():
        p = fit_pred(mdl, tr[feats].values, tr["y"].values, te[feats].values)
        ic = spearmanr(p, te["y"].values)[0]
        r2 = 1 - np.sum((te["y"].values - p)**2) / np.sum((te["y"].values - tr["y"].mean())**2)
        hit = np.mean(np.sign(p) == np.sign(te["y"].values))
        rng = np.random.default_rng(0)
        nul = np.array([spearmanr(fit_pred(models()[nm], tr[feats].values, rng.permutation(tr["y"].values),
                                            te[feats].values), te["y"].values)[0] for _ in range(nperm)])
        pz = pd.Series(p, index=te.index); pz = (pz - pz.mean()) / pz.std()
        for b in (0.3,):
            exp = (1 + b * pz).clip(0.3, 1.5)
            base = bret.reindex(te.index)
            ov = exp * base + (1 - exp) * rf_m
            sh, cal = stats(ov); sh0, cal0 = stats(base)
        print(f"    [{nm:5}] test rank-IC={ic:+.3f}  R2={r2:+.3f}  dir-acc={hit:.2f}  "
              f"perm p(IC>=obs)={np.mean(nul >= ic):.2f}   overlay Sh {sh:+.2f} vs static {sh0:+.2f}")

def walkfwd(feats, a, b, min_train=18, label=""):
    D = X[feats].join(y1.rename("y")).loc[a:b].dropna()
    preds = pd.Series(index=D.index, dtype=float)
    for i in range(min_train, len(D)):
        tr = D.iloc[:i]; row = D.iloc[[i]]
        preds.iloc[i] = fit_pred(models()["GBM"], tr[feats].values, tr["y"].values, row[feats].values)[0]
    k = preds.notna()
    ic = spearmanr(preds[k], D["y"][k])[0]
    hit = np.mean(np.sign(preds[k]) == np.sign(D["y"][k]))
    print(f"\n  {label} expanding walk-forward GBM (min_train={min_train})  OOS n={k.sum()}  "
          f"rank-IC={ic:+.3f}  dir-acc={hit:.2f}  span {D.index[k][0].date()}..{D.index[k][-1].date()}")

print("=" * 94)
print("2023-26 SPLIT INTO IS / OOS HALVES (same AI regime)")
print("\n[A] strict holdout: train 2023-24, test 2025-26")
holdout(FULL, "2023-01", "2024-12", "2025-01", "2026-08", "FULL 25 feats")
holdout(LEAN, "2023-01", "2024-12", "2025-01", "2026-08", "LEAN 6 feats")
print("\n[A-rev] robustness: train 2025-26, test 2023-24")
holdout(LEAN, "2025-01", "2026-08", "2023-01", "2024-12", "LEAN 6 feats")
print("\n[B] expanding walk-forward WITHIN 2023-26 (every test month trained only on prior 2023+ data)")
walkfwd(FULL, "2023-01", "2026-08", 18, "FULL")
walkfwd(LEAN, "2023-01", "2026-08", 18, "LEAN")
