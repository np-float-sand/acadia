import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

OUT = Path("research/narrative_mispricing/approach_c")
panel = pd.read_parquet(OUT / "panel.parquet")

FULL_FEATS = ["narrative_z", "icr", "stress_beta", "gsi_factor", "own_mom6m",
              "capex_guidance_z", "category_demand_z", "ancillary_pjm_z",
              "real_yield_chg6m", "ng_mom6m"]
LEAN_FEATS = ["narrative_z", "icr", "gsi_factor", "capex_guidance_z"]

def hac_tstat(x, lags=3):
    """Newey-West t-stat for the mean of a series (matches this repo's
    HAC-OLS convention used throughout capex_guidance_signal.py etc.)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    mean = x.mean()
    resid = x - mean
    gamma0 = (resid @ resid) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1 - l / (lags + 1)
        cov = (resid[l:] @ resid[:-l]) / n
        var += 2 * w * cov
    se = np.sqrt(max(var, 1e-12) / n)
    return mean, mean / se if se > 0 else np.nan

def oof_rank_ic(panel_sub, feats, target, n_splits=5, model_kind="gbm", seed=0):
    """Walk-forward TimeSeriesSplit on the month axis. For each test fold,
    fit on all prior months, predict the held-out months, then compute
    cross-sectional Spearman IC per held-out month."""
    months_sorted = np.sort(panel_sub["month"].unique())
    if len(months_sorted) < n_splits + 5:
        n_splits = max(2, len(months_sorted) // 6)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ic_by_month = {}
    fold_importances = []
    for train_idx, test_idx in tscv.split(months_sorted):
        train_months = months_sorted[train_idx]
        test_months = months_sorted[test_idx]
        train = panel_sub[panel_sub["month"].isin(train_months)]
        test = panel_sub[panel_sub["month"].isin(test_months)]
        if train[target].notna().sum() < 20 or test[target].notna().sum() < 5:
            continue
        Xtr = train[feats].copy()
        ytr = train[target]
        med = Xtr.median().fillna(0.0)  # all-NaN-in-fold column (e.g. ancillary
                                         # post-2022) -> neutral 0 rather than NaN
        Xtr = Xtr.fillna(med).fillna(0.0)
        mask = ytr.notna()
        Xtr, ytr = Xtr[mask], ytr[mask]
        if model_kind == "gbm":
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                               learning_rate=0.05, subsample=0.8,
                                               random_state=seed)
        else:
            model = RandomForestRegressor(n_estimators=200, max_depth=4,
                                           min_samples_leaf=5, random_state=seed)
        model.fit(Xtr, ytr)
        fold_importances.append(pd.Series(model.feature_importances_, index=feats))
        for m in test_months:
            tm = test[test["month"] == m]
            Xte = tm[feats].fillna(med).fillna(0.0)
            yte = tm[target]
            valid = yte.notna()
            if valid.sum() < 5:
                continue
            pred = model.predict(Xte[valid])
            ic, _ = spearmanr(pred, yte[valid])
            ic_by_month[m] = ic
    ics = pd.Series(ic_by_month).sort_index()
    mean_ic, t = hac_tstat(ics.values)
    imp = pd.concat(fold_importances, axis=1).mean(axis=1) if fold_importances else pd.Series(dtype=float)
    return ics, mean_ic, t, imp

def permutation_pvalue(panel_sub, feats, target, obs_ic, n_splits, model_kind, n_perm=100, seed0=1000):
    """Shuffle the target WITHIN each month (preserves cross-sectional
    structure, breaks the feature->target link) and recompute OOF IC."""
    null_ics = []
    rng = np.random.default_rng(42)
    for i in range(n_perm):
        shuffled = panel_sub.copy()
        shuffled[target] = shuffled.groupby("month")[target].transform(
            lambda x: rng.permutation(x.values) if x.notna().sum() > 1 else x.values
        )
        _, mean_ic, _, _ = oof_rank_ic(shuffled, feats, target, n_splits, model_kind, seed=seed0 + i)
        if not np.isnan(mean_ic):
            null_ics.append(mean_ic)
    null_ics = np.array(null_ics)
    if len(null_ics) == 0 or np.isnan(obs_ic):
        return np.nan
    return float((np.abs(null_ics) >= abs(obs_ic)).mean())

WINDOWS = {
    "FULL_2018_2025": panel,
    "RECENT_2023_2025": panel[panel["month"] >= "2023-01-01"],
}

results = []
print(f"{'window':<18}{'model':<6}{'feats':<6}{'target':<10}{'mean_IC':>9}{'t':>7}{'perm_p':>8}")
for wname, wpanel in WINDOWS.items():
    for feat_name, feats in [("full", FULL_FEATS), ("lean", LEAN_FEATS)]:
        for target in ["fwd_ret_1m", "fwd_ret_3m"]:
            for model_kind in ["gbm", "rf"]:
                ics, mean_ic, t, imp = oof_rank_ic(wpanel, feats, target, n_splits=5, model_kind=model_kind)
                perm_p = permutation_pvalue(wpanel, feats, target, mean_ic, n_splits=5,
                                             model_kind=model_kind, n_perm=60)
                print(f"{wname:<18}{model_kind:<6}{feat_name:<6}{target:<10}{mean_ic:>9.3f}{t:>7.2f}{perm_p:>8.3f}")
                results.append(dict(window=wname, model=model_kind, feats=feat_name, target=target,
                                     mean_ic=mean_ic, t=t, perm_p=perm_p, n_months=len(ics)))
                if feat_name == "full" and model_kind == "gbm":
                    top_imp = imp.sort_values(ascending=False)
                    print("   top importances:", dict(top_imp.round(3)))

pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)
