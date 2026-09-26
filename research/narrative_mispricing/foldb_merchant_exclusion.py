"""
Merchant-name confound check on the third session's Fold B finding
(train 2024-01->2025-12, test 2026-01->2026-08, GBM classifier tercile,
OOS AUC=0.758 documented in docs/ml-two-fold-walkforward-results.md).

Question: is that result broad across the regulated-utility universe, or is it
just VST/CEG/TLN/NRG (the four names already documented in
docs/grid-stress-signal-evidence.md S5 as the recurring confound behind every
other broken cross-sectional signal on this universe) re-rating for reasons
already known and unrelated to narrative/stress-beta combination?

Exact same methodology as stress_test_foldb_auc.py (same features, same GBM
hyperparams, same folds) on three ticker cuts:
  ALL_23      -- the original universe (reproduces 0.758 as a sanity check)
  EX_MERCHANT -- ALL_23 minus {VST, CEG, TLN, NRG} (19 regulated names)
  MERCHANT_4  -- only {VST, CEG, TLN, NRG}
Terciles are recomputed per-date WITHIN each cut (not inherited from the full
universe), since that's the fair way to ask "does this cut have its own
top/bottom-tercile signal," matching how a real long/short construction on a
sub-universe would work.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

panel = pd.read_parquet("/Users/sandhyapersad/acadia/research/narrative_mispricing/ml_panel_2022_2026.parquet")
FEATURES = ["icr_z", "edgar_z", "gdelt_z", "stress_beta_z", "xlu_xng_spread", "mom_z", "anc_composite"]
target = "fwd3"
MERCHANT = ["VST", "CEG", "TLN", "NRG"]

tr_mask = (panel["date"] >= "2024-01-01") & (panel["date"] < "2026-01-01")
te_mask = (panel["date"] >= "2026-01-01") & (panel["date"] < "2026-09-01")

def tercile(df, target):
    d = df.dropna(subset=FEATURES + [target]).copy()
    d["tercile"] = d.groupby("date")[target].transform(
        lambda x: pd.qcut(x, 3, labels=False, duplicates="drop") if x.nunique() >= 3 else pd.Series(np.nan, index=x.index))
    d = d[d["tercile"].isin([0, 2])]
    return d

CUTS = {
    "ALL_23": panel["ticker"].unique().tolist(),
    "EX_MERCHANT_19": [t for t in panel["ticker"].unique() if t not in MERCHANT],
    "MERCHANT_4": MERCHANT,
}

print(f"{'cut':<16}{'n_tickers':>10}{'test_n':>8}{'pos_share':>10}{'mean_AUC(10 seeds)':>20}{'std':>7}{'null_pctile':>13}")
for cut_name, cut_tickers in CUTS.items():
    sub = panel[panel["ticker"].isin(cut_tickers)]
    d_tr = tercile(sub[tr_mask], target)
    d_te = tercile(sub[te_mask], target)
    if len(d_tr) < 20 or len(d_te) < 10:
        print(f"{cut_name:<16} insufficient data (train n={len(d_tr)}, test n={len(d_te)})")
        continue
    Xtr, ytr = d_tr[FEATURES], (d_tr["tercile"] == 2).astype(int)
    Xte, yte = d_te[FEATURES], (d_te["tercile"] == 2).astype(int)
    if ytr.nunique() < 2 or yte.nunique() < 2:
        print(f"{cut_name:<16} only one class present, AUC undefined")
        continue

    real_aucs = []
    for seed in range(10):
        m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=seed)
        m.fit(Xtr, ytr)
        real_aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    mean_auc, std_auc = np.mean(real_aucs), np.std(real_aucs)

    null_aucs = []
    rng = np.random.RandomState(0)
    for i in range(50):
        yshuf = ytr.sample(frac=1.0, random_state=rng.randint(100000)).values
        m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(Xtr, yshuf)
        null_aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    null_aucs = np.array(null_aucs)
    pctile = (null_aucs < mean_auc).mean()

    print(f"{cut_name:<16}{len(cut_tickers):>10}{len(Xte):>8}{yte.mean():>10.2f}{mean_auc:>20.3f}{std_auc:>7.3f}{pctile*100:>12.0f}%")

    # feature importances at seed=42 for this cut
    m42 = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42).fit(Xtr, ytr)
    print(f"    importances: {dict(zip(FEATURES, m42.feature_importances_.round(3)))}")
    print(f"    per-month test AUC:")
    d_te2 = d_te.copy()
    d_te2["p"] = m42.predict_proba(Xte)[:, 1]
    for month, g in d_te2.groupby(d_te2["date"].dt.strftime("%Y-%m")):
        if g["tercile"].map({0: 0, 2: 1}).nunique() < 2:
            print(f"      {month}: n={len(g)}, one class only")
            continue
        yb = (g["tercile"] == 2).astype(int)
        print(f"      {month}: n={len(g)}, AUC={roc_auc_score(yb, g['p']):.3f}")
