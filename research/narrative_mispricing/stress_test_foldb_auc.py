"""Stress-test the eye-catching Fold B (train 2024-25, test 2026) GBM classifier
OOS AUC=0.758 on fwd3: (1) shuffled-label null distribution, (2) check whether it's
driven by one or two months/tickers, (3) same test with a different random_state to
check sensitivity to the specific model seed."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

panel = pd.read_parquet("research/narrative_mispricing/ml_panel_2022_2026.parquet")
FEATURES = ["icr_z", "edgar_z", "gdelt_z", "stress_beta_z", "xlu_xng_spread", "mom_z", "anc_composite"]
target = "fwd3"

tr_mask = (panel["date"] >= "2024-01-01") & (panel["date"] < "2026-01-01")
te_mask = (panel["date"] >= "2026-01-01") & (panel["date"] < "2026-09-01")

def tercile(df, target):
    d = df.dropna(subset=FEATURES + [target]).copy()
    d["tercile"] = d.groupby("date")[target].transform(lambda x: pd.qcut(x, 3, labels=False, duplicates="drop"))
    d = d[d["tercile"].isin([0, 2])]
    return d

d_tr = tercile(panel[tr_mask], target)
d_te = tercile(panel[te_mask], target)
Xtr, ytr = d_tr[FEATURES], (d_tr["tercile"] == 2).astype(int)
Xte, yte = d_te[FEATURES], (d_te["tercile"] == 2).astype(int)
print(f"test months: {sorted(d_te['date'].dt.strftime('%Y-%m').unique())}")
print(f"test n={len(Xte)}, positive class share={yte.mean():.2f}")

# Real model, multiple seeds
print("\n=== Real model, across 10 random seeds (model variance) ===")
real_aucs = []
for seed in range(10):
    m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    real_aucs.append(auc)
print(f"  AUCs: {[round(a,3) for a in real_aucs]}")
print(f"  mean={np.mean(real_aucs):.3f}  std={np.std(real_aucs):.3f}")

# Shuffled-label null distribution
print("\n=== Shuffled-label null distribution (50 shuffles) ===")
null_aucs = []
rng = np.random.RandomState(0)
for i in range(50):
    yshuf = ytr.sample(frac=1.0, random_state=rng.randint(100000)).values
    m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    m.fit(Xtr, yshuf)
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    null_aucs.append(auc)
null_aucs = np.array(null_aucs)
print(f"  null mean={null_aucs.mean():.3f}  std={null_aucs.std():.3f}  "
      f"[5th,95th pct]=[{np.percentile(null_aucs,5):.3f}, {np.percentile(null_aucs,95):.3f}]")
pctile = (null_aucs < 0.758).mean()
print(f"  real AUC=0.758 sits at the {pctile*100:.0f}th percentile of the null distribution")

# Per-month breakdown -- is this driven by one month?
print("\n=== Per-test-month prediction accuracy (real model, seed=42) ===")
m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
m.fit(Xtr, ytr)
d_te2 = d_te.copy()
d_te2["pred_proba"] = m.predict_proba(Xte)[:, 1]
d_te2["true"] = yte.values
for month, g in d_te2.groupby(d_te2["date"].dt.strftime("%Y-%m")):
    if g["true"].nunique() < 2:
        print(f"  {month}: n={len(g)}, only one class present, AUC undefined")
        continue
    print(f"  {month}: n={len(g)}, AUC={roc_auc_score(g['true'], g['pred_proba']):.3f}")

print("\n=== Classifier feature importances (the actual driver) ===")
print(f"  real: {dict(zip(FEATURES, m.feature_importances_.round(3)))}")

# Shuffled-label classifier importances for comparison
m_shuf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
yshuf_fixed = ytr.sample(frac=1.0, random_state=7).values
m_shuf.fit(Xtr, yshuf_fixed)
print(f"  shuffled: {dict(zip(FEATURES, m_shuf.feature_importances_.round(3)))}")

print("\n=== Drop-one-feature ablation (does removing gdelt_z, the stale one, change AUC?) ===")
for drop in FEATURES:
    feats = [f for f in FEATURES if f != drop]
    m2 = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
    m2.fit(Xtr[feats], ytr)
    auc2 = roc_auc_score(yte, m2.predict_proba(Xte[feats])[:, 1])
    print(f"  drop {drop:16s}: OOS AUC={auc2:.3f}  (full model = 0.758)")
