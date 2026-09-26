"""
RF/GBM/classifier/quantile retest of the session's weak signals, combined, on the
2022-01 -> 2025-09 panel only (pre-2022 excluded per user instruction -- electrification
narrative wasn't really live before ChatGPT).

Pre-registered before looking at any result:
  - OOS split: train 2022-01->2023-12 (24mo), test 2024-01->2025-09 (21mo). Single clean
    time split, not random k-fold (autocorrelated monthly panel).
  - PASS bar: OOS rank-IC (regression) or OOS AUC (classification) must beat the in-sample
    fit by a plausible margin AND clear a level that would be interesting (rank-IC |t|>=2
    equivalent, AUC notably >0.55) -- given n_test=21 months this is a high bar almost by
    construction; a look-good-in-sample/collapse-out-of-sample pattern is the PRE-REGISTERED
    expectation given the small sample, not a surprise if it happens.
  - Sanity check: refit every model on a SHUFFLED target; if shuffled-target feature
    importances resemble the real ones, the "importance" is structural noise, not signal.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
np.random.seed(42)

panel = pd.read_parquet("research/narrative_mispricing/ml_panel_2022_2025.parquet")
FEATURES = ["icr_z", "edgar_z", "gdelt_z", "stress_beta_z", "xlu_xng_spread", "mom_z", "anc_composite"]

train_mask = panel["date"] < "2024-01-01"
test_mask = panel["date"] >= "2024-01-01"
print(f"train n={train_mask.sum()}, test n={test_mask.sum()}")


def prep(df, target):
    d = df.dropna(subset=FEATURES + [target]).copy()
    return d[FEATURES], d[target]


def rank_ic(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation


results = {}
for target in ["fwd1", "fwd3"]:
    print(f"\n{'='*20} TARGET: {target} {'='*20}")
    Xtr, ytr = prep(panel[train_mask], target)
    Xte, yte = prep(panel[test_mask], target)
    print(f"  train n={len(Xtr)}, test n={len(Xte)}")
    if len(Xte) < 20:
        print("  insufficient test data, skip")
        continue

    # --- RF regressor ---
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42)
    rf.fit(Xtr, ytr)
    ic_in = rank_ic(ytr, rf.predict(Xtr))
    ic_out = rank_ic(yte, rf.predict(Xte))
    print(f"  RF regressor:  in-sample IC={ic_in:+.3f}   OOS IC={ic_out:+.3f}")
    print(f"    feature importances: {dict(zip(FEATURES, rf.feature_importances_.round(3)))}")

    # --- GBM regressor ---
    gbm = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=42)
    gbm.fit(Xtr, ytr)
    ic_in_g = rank_ic(ytr, gbm.predict(Xtr))
    ic_out_g = rank_ic(yte, gbm.predict(Xte))
    print(f"  GBM regressor: in-sample IC={ic_in_g:+.3f}   OOS IC={ic_out_g:+.3f}")
    print(f"    feature importances: {dict(zip(FEATURES, gbm.feature_importances_.round(3)))}")

    # --- Quantile regression (GBM, q=0.25/0.5/0.75) ---
    print("  Quantile GBM (predict low/median/high quantile of fwd return):")
    for q in [0.25, 0.5, 0.75]:
        qgbm = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=150,
                                          max_depth=3, learning_rate=0.05, random_state=42)
        qgbm.fit(Xtr, ytr)
        # pinball loss OOS
        pred = qgbm.predict(Xte)
        err = yte.values - pred
        pinball = np.mean(np.maximum(q*err, (q-1)*err))
        ic_q = rank_ic(yte, pred)
        print(f"    q={q}: OOS pinball={pinball:.4f}  OOS rank-IC={ic_q:+.3f}")

    # --- Classification: top-tercile vs bottom-tercile forward return ---
    print("  Classification (top-tercile vs bottom-tercile fwd return, RF & GBM):")
    def tercile_labels(sub_df, target):
        d = sub_df.dropna(subset=FEATURES + [target]).copy()
        d["tercile"] = d.groupby("date")[target].transform(
            lambda x: pd.qcut(x, 3, labels=False, duplicates="drop"))
        d = d[d["tercile"].isin([0, 2])].copy()
        d["label"] = (d["tercile"] == 2).astype(int)
        return d[FEATURES], d["label"]

    Xtr_c, ytr_c = tercile_labels(panel[train_mask], target)
    Xte_c, yte_c = tercile_labels(panel[test_mask], target)
    if len(Xte_c) >= 20 and ytr_c.nunique() == 2 and yte_c.nunique() == 2:
        rfc = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42)
        rfc.fit(Xtr_c, ytr_c)
        auc_in = roc_auc_score(ytr_c, rfc.predict_proba(Xtr_c)[:, 1])
        auc_out = roc_auc_score(yte_c, rfc.predict_proba(Xte_c)[:, 1])
        print(f"    RF classifier:  in-sample AUC={auc_in:.3f}   OOS AUC={auc_out:.3f}")
        print(f"      importances: {dict(zip(FEATURES, rfc.feature_importances_.round(3)))}")

        gbc = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
        gbc.fit(Xtr_c, ytr_c)
        auc_in_g = roc_auc_score(ytr_c, gbc.predict_proba(Xtr_c)[:, 1])
        auc_out_g = roc_auc_score(yte_c, gbc.predict_proba(Xte_c)[:, 1])
        print(f"    GBM classifier: in-sample AUC={auc_in_g:.3f}   OOS AUC={auc_out_g:.3f}")
        print(f"      importances: {dict(zip(FEATURES, gbc.feature_importances_.round(3)))}")
    else:
        print("    insufficient tercile-labeled data")

    # --- Sanity check: shuffled-target feature importance comparison ---
    print("  Sanity check -- shuffled-target RF (should show similar 'importance' if noise):")
    yshuf = ytr.sample(frac=1.0, random_state=1).reset_index(drop=True)
    yshuf.index = ytr.index
    rf_shuf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42)
    rf_shuf.fit(Xtr, yshuf)
    print(f"    shuffled importances: {dict(zip(FEATURES, rf_shuf.feature_importances_.round(3)))}")
