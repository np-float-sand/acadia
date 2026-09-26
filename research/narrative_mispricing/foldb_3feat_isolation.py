"""
Item 18 follow-up (b): isolate the stress_beta_z x icr_z x edgar_z interaction on
Fold B (train 2024-01->2025-12, test 2026-01->2026-08), dropping the other 4
features (mom_z, xlu_xng_spread, anc_composite -- noise/dead-weight per the
drop-one ablation -- and gdelt_z, known-stale in the test window).
Run on ALL_23 and EX_MERCHANT_19 (per item 18 + the merchant-exclusion addendum)
for a clean before/after comparison against the 7-feature numbers already logged.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

panel = pd.read_parquet("/Users/sandhyapersad/acadia/research/narrative_mispricing/ml_panel_2022_2026.parquet")
FULL7 = ["icr_z", "edgar_z", "gdelt_z", "stress_beta_z", "xlu_xng_spread", "mom_z", "anc_composite"]
LEAN3 = ["stress_beta_z", "icr_z", "edgar_z"]
target = "fwd3"
MERCHANT = ["VST", "CEG", "TLN", "NRG"]

tr_mask = (panel["date"] >= "2024-01-01") & (panel["date"] < "2026-01-01")
te_mask = (panel["date"] >= "2026-01-01") & (panel["date"] < "2026-09-01")

def tercile(df, feats, target):
    d = df.dropna(subset=feats + [target]).copy()
    d["tercile"] = d.groupby("date")[target].transform(
        lambda x: pd.qcut(x, 3, labels=False, duplicates="drop") if x.nunique() >= 3 else pd.Series(np.nan, index=x.index))
    return d[d["tercile"].isin([0, 2])]

CUTS = {"ALL_23": panel["ticker"].unique().tolist(),
        "EX_MERCHANT_19": [t for t in panel["ticker"].unique() if t not in MERCHANT]}

print(f"{'cut':<16}{'feats':<6}{'test_n':>8}{'mean_AUC(10 seeds)':>20}{'std':>7}{'null_pctile':>13}")
for cut_name, cut_tickers in CUTS.items():
    sub = panel[panel["ticker"].isin(cut_tickers)]
    for feat_name, feats in [("full7", FULL7), ("lean3", LEAN3)]:
        d_tr = tercile(sub[tr_mask], feats, target)
        d_te = tercile(sub[te_mask], feats, target)
        if len(d_tr) < 20 or len(d_te) < 10:
            print(f"{cut_name:<16}{feat_name:<6} insufficient data"); continue
        Xtr, ytr = d_tr[feats], (d_tr["tercile"] == 2).astype(int)
        Xte, yte = d_te[feats], (d_te["tercile"] == 2).astype(int)
        if ytr.nunique() < 2 or yte.nunique() < 2:
            print(f"{cut_name:<16}{feat_name:<6} one class only"); continue

        aucs = []
        for seed in range(10):
            m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                            subsample=0.8 if len(feats)>3 else 1.0, random_state=seed)
            m.fit(Xtr, ytr)
            aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)

        null_aucs = []
        rng = np.random.RandomState(0)
        for i in range(50):
            yshuf = ytr.sample(frac=1.0, random_state=rng.randint(100000)).values
            m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
            m.fit(Xtr, yshuf)
            null_aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        pctile = (np.array(null_aucs) < mean_auc).mean()
        print(f"{cut_name:<16}{feat_name:<6}{len(Xte):>8}{mean_auc:>20.3f}{std_auc:>7.3f}{pctile*100:>12.0f}%")

        if feat_name == "lean3":
            m42 = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42).fit(Xtr, ytr)
            print(f"    importances: {dict(zip(feats, m42.feature_importances_.round(3)))}")
            d_te2 = d_te.copy(); d_te2["p"] = m42.predict_proba(Xte)[:, 1]
            months = []
            for month, g in d_te2.groupby(d_te2["date"].dt.strftime("%Y-%m")):
                yb = (g["tercile"] == 2).astype(int)
                months.append(f"{month}:{roc_auc_score(yb, g['p']):.2f}" if yb.nunique()==2 else f"{month}:n/a")
            print(f"    per-month AUC: {months}")
