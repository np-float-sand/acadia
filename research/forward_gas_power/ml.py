"""All session signals -> GBM / RandomForest, walk-forward OOS, 2023-26.
Honest eval: TimeSeriesSplit out-of-fold rank-IC + R2 + permutation null.
n~43 monthly obs -> expect noise; in-sample fit is not reported as evidence."""
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
smh, qqq = mc("SMH").pct_change(), mc("QQQ").pct_change()
d10y = mc("^TNX").diff()
pjw, erc, cal = eia("ELWHU_PJ"), eia("ELWHU_TX"), eia("ELWHU_CA")
CD = fetch_category_demand()
def zmom(c, m):
    s = momentum(CD, c, m); return (s - s.expanding(24).mean()) / s.expanding(24).std()

NAMES = sorted(config.UNIVERSE)
PX = fetch_prices(NAMES, "2016-01-01", "2026-08-31"); PX = PX[[c for c in NAMES if c in PX.columns]].ffill()
bret = (1 + PX.pct_change()).resample("ME").prod().mean(1) - 1.0

idx = pd.date_range("2016-01-31", "2026-08-31", freq="ME")
X = pd.DataFrame(index=idx)
for m in (3, 6, 12):
    X[f"ng_chg{m}"] = np.log(ngf).reindex(idx).diff(m)
    X[f"unl_chg{m}"] = np.log(unl).reindex(idx).diff(m)
    X[f"pjw_chg{m}"] = np.log(pjw).reindex(idx).diff(m)
    X[f"erc_chg{m}"] = np.log(erc).reindex(idx).diff(m)
X["unl_ema6"] = np.log(unl).reindex(idx).diff().ewm(span=6).mean() * 6
X["strip_slope"] = (unl / ung).reindex(idx)                      # 12mo strip / front (real ratio)
X["strip_slope_chg6"] = X["strip_slope"].diff(6)
X["sparkPJ"] = (pjw.reindex(idx) / ngf.reindex(idx))             # implied heat rate proxy
X["sparkPJ_chg6"] = np.log(X["sparkPJ"]).diff(6)
X["sparkERC"] = (erc.reindex(idx) / ngf.reindex(idx))
X["sparkERC_chg6"] = np.log(X["sparkERC"]).diff(6)
for c in ("ppi_switchgear", "ppi_transformers", "ee_new_orders", "ee_unfilled_orders"):
    z = zmom(c, 6); z.index = z.index.to_period("M").to_timestamp("M")
    X[f"cat_{c}"] = z.reindex(idx).shift(2)
X["catscore"] = (0.4 * X["cat_ppi_switchgear"] + 0.3 * X["cat_ppi_transformers"] + 0.3 * X["cat_ee_new_orders"])
X["smh"] = smh.reindex(idx); X["qqq"] = qqq.reindex(idx); X["d10y"] = d10y.reindex(idx)
X["b_mom3"] = bret.reindex(idx).rolling(3).sum()                 # basket own momentum (control feature)

y1 = bret.reindex(idx).shift(-1).rename("y1")
y3 = (bret.reindex(idx).shift(-1).rolling(3).sum().shift(-2)).rename("y3")

FEATS = [c for c in X.columns]

def run(win_start, win_end, target, label):
    D = X.join(target)
    D = D.loc[win_start:win_end].dropna()
    if len(D) < 25:
        print(f"  {label}: n={len(D)} too small"); return
    Xm, ym = D[FEATS].values, D.iloc[:, -1].values
    tscv = TimeSeriesSplit(n_splits=5)
    for name, mdl in [
        ("GBM", GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03,
                                          subsample=0.8, min_samples_leaf=4, random_state=0)),
        ("RF ", RandomForestRegressor(n_estimators=400, max_depth=3, min_samples_leaf=4,
                                      max_features=0.5, random_state=0)),
    ]:
        oof = np.full(len(ym), np.nan)
        for tr, te in tscv.split(Xm):
            mdl.fit(Xm[tr], ym[tr]); oof[te] = mdl.predict(Xm[te])
        ok = ~np.isnan(oof)
        r2 = 1 - np.sum((ym[ok] - oof[ok]) ** 2) / np.sum((ym[ok] - ym[ok].mean()) ** 2)
        ic, _ = spearmanr(oof[ok], ym[ok])
        hit = np.mean(np.sign(oof[ok]) == np.sign(ym[ok]))
        # permutation null on the OOF IC
        rng = np.random.default_rng(1); perm = []
        for _ in range(200):
            yp = rng.permutation(ym); of = np.full(len(ym), np.nan)
            for tr, te in tscv.split(Xm):
                mdl.fit(Xm[tr], yp[tr]); of[te] = mdl.predict(Xm[te])
            perm.append(spearmanr(of[~np.isnan(of)], yp[~np.isnan(of)])[0])
        perm = np.array(perm); pval = np.mean(perm >= ic)
        print(f"  {label} [{name}] n={ok.sum():3}  OOF R2={r2:+.3f}  OOF rank-IC={ic:+.3f}  "
              f"dir-acc={hit:.2f}  perm p(IC>=obs)={pval:.2f}")
    # full-sample importances (descriptive only)
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03,
                                   subsample=0.8, min_samples_leaf=4, random_state=0).fit(Xm, ym)
    imp = pd.Series(gb.feature_importances_, index=FEATS).sort_values(ascending=False)
    print(f"      top features (in-sample, descriptive): {', '.join(f'{k}={v:.2f}' for k,v in imp.head(6).items())}")

print("=" * 88)
print(f"ML on all session signals -- {len(FEATS)} features, walk-forward TimeSeriesSplit OOF")
print("target y1 = basket forward 1-month return")
run("2023-01", "2026-08", y1, "2023-26 y1")
run("2018-06", "2026-08", y1, "2018-26 y1 (more rows)")
print("\ntarget y3 = basket forward 3-month return")
run("2023-01", "2026-08", y3, "2023-26 y3")
run("2018-06", "2026-08", y3, "2018-26 y3")
