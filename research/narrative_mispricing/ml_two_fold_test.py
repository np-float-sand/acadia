"""
Two-fold walk-forward retest per user request:
  Fold A: train 2022-01 -> 2023-12, test 2024-01 -> 2024-12
  Fold B: train 2024-01 -> 2025-12, test 2026-01 -> latest available
Each fold judged on its own terms -- an early fold's weak generalization is not labeled
"overfitting" if it reflects the trade/narrative still being early-regime, per user
correction. Same 7 features + shuffled-target sanity check as the original combined test.

KNOWN GAP: the GDELT 2026 extension never completed (stuck on repeated rate-limit backoffs
for ~25+ min); gdelt_z is forward-filled from its last real observation (~2025-09) through
2026, i.e. it is NOT informative for any 2026 test-period date -- treat gdelt_z's
contribution to Fold B as structurally void, not as evidence either way.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/sandhyapersad/acadia")
warnings.filterwarnings("ignore")
np.random.seed(42)

TODAY = pd.Timestamp("2026-09-08")
MONTHLY = pd.date_range("2022-01-01", "2026-08-01", freq="MS")

# ---------- Load signals (now extended through 2026) ----------
icr = pd.read_parquet("research/icr_edgar_retest/icr_edgar_panel.parquet")
edgar_dc = pd.read_parquet("research/narrative_mispricing/edgar_quarterly_mentions.parquet")
edgar_ll = pd.read_parquet("research/narrative_mispricing/edgar_quarterly_large_load.parquet")
gdelt = pd.read_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")
print(f"GDELT coverage ends: {gdelt.apply(lambda c: c.dropna().index.max() if c.notna().any() else None).max()}")

betas = pd.read_parquet("research/narrative_mispricing/rolling_betas_2018_2026.parquet")
xlu_xng = pd.read_parquet("research/narrative_mispricing/xlu_xng_spread_continuous.parquet")

anc21 = pd.read_parquet("research/merchant_beta_orthogonalization/pjm_ancillary_2018_2021.parquet")
anc25 = pd.read_parquet("research/narrative_mispricing/pjm_ancillary_2022_2025.parquet")
anc26 = pd.read_parquet("research/narrative_mispricing/pjm_ancillary_2026.parquet")
anc = pd.concat([anc21, anc25, anc26], ignore_index=True)

tickers = sorted(set(icr.columns) & set(edgar_dc.columns) & set(xlu_xng.columns))
print(f"{len(tickers)} tickers")

def to_monthly_ffill(df, dates=MONTHLY):
    return df.reindex(df.index.union(dates)).sort_index().ffill().reindex(dates)

def xsec_z(frame):
    return frame.apply(lambda row: (row - row.dropna().mean()) / row.dropna().std()
                        if row.dropna().shape[0] >= 5 else pd.Series(index=row.index, dtype=float),
                        axis=1)

icr_z = xsec_z(to_monthly_ffill(icr)[tickers])

edgar_combined = edgar_dc.add(edgar_ll, fill_value=0).sort_index().rolling(4, min_periods=1).sum()
edgar_z = xsec_z(to_monthly_ffill(edgar_combined)[tickers])

gdelt_self_mean = gdelt.rolling(756, min_periods=180).mean()
gdelt_self_std = gdelt.rolling(756, min_periods=180).std()
gdelt_self_z = ((gdelt - gdelt_self_mean) / gdelt_self_std.replace(0, np.nan)).clip(-4, 4)
gdelt_monthly = to_monthly_ffill(gdelt_self_z)
gdelt_cols = [t for t in tickers if t in gdelt_monthly.columns]

stress_beta_wide = betas["stress_beta"].unstack("ticker")
stress_beta_z = xsec_z(to_monthly_ffill(stress_beta_wide)[[t for t in tickers if t in stress_beta_wide.columns]])

xlu_xng_monthly = to_monthly_ffill(xlu_xng)[tickers]

px = yf.download(tickers, start="2020-01-01", end="2026-09-08", auto_adjust=True, progress=False)["Close"]
mom = px.pct_change(126).shift(21)
mom_z = xsec_z(to_monthly_ffill(mom)[tickers])

anc["Interval Start"] = pd.to_datetime(anc["Interval Start"])
anc["date"] = anc["Interval Start"].dt.date
rto = anc[anc["Ancillary Service"].str.startswith("PJM_RTO")]
anc_daily = rto.groupby(["date", "Ancillary Service"])["Market Clearing Price"].mean().unstack()
anc_daily.index = pd.to_datetime(anc_daily.index)
anc_z_daily = (anc_daily - anc_daily.rolling(252, min_periods=126).mean()) / anc_daily.rolling(252, min_periods=126).std()
anc_composite = anc_z_daily.mean(axis=1)
anc_monthly = to_monthly_ffill(anc_composite.to_frame("anc"))["anc"]
print(f"Ancillary coverage: {anc_monthly.notna().sum()}/{len(MONTHLY)} months, through {anc_monthly.dropna().index.max()}")

pjm_map = {"NRG":"ERCOT","VST":"ERCOT","CNP":"ERCOT","AEP":"PJM","EXC":"PJM","PPL":"PJM",
           "FE":"PJM","CEG":"PJM","TLN":"PJM","D":"PJM","PEG":"PJM"}

def sample_at(df, dates):
    return df.reindex(df.index.union(dates)).sort_index().bfill().reindex(dates)
fwd1 = sample_at(px.pct_change(21).shift(-21), MONTHLY)[tickers]
fwd3 = sample_at(px.pct_change(63).shift(-63), MONTHLY)[tickers]

rows = []
for d in MONTHLY:
    for t in tickers:
        rows.append({
            "date": d, "ticker": t,
            "icr_z": icr_z.loc[d, t] if d in icr_z.index and t in icr_z.columns else np.nan,
            "edgar_z": edgar_z.loc[d, t] if d in edgar_z.index and t in edgar_z.columns else np.nan,
            "gdelt_z": gdelt_monthly.loc[d, t] if (t in gdelt_cols and d in gdelt_monthly.index) else np.nan,
            "stress_beta_z": stress_beta_z.loc[d, t] if d in stress_beta_z.index and t in stress_beta_z.columns else np.nan,
            "xlu_xng_spread": xlu_xng_monthly.loc[d, t] if d in xlu_xng_monthly.index and t in xlu_xng_monthly.columns else np.nan,
            "mom_z": mom_z.loc[d, t] if d in mom_z.index and t in mom_z.columns else np.nan,
            "anc_composite": anc_monthly.loc[d] if d in anc_monthly.index and pjm_map.get(t) == "PJM" else 0.0,
            "fwd1": fwd1.loc[d, t] if d in fwd1.index and t in fwd1.columns else np.nan,
            "fwd3": fwd3.loc[d, t] if d in fwd3.index and t in fwd3.columns else np.nan,
        })
panel = pd.DataFrame(rows)
panel.to_parquet("research/narrative_mispricing/ml_panel_2022_2026.parquet")
print(f"Panel shape: {panel.shape}, date range {panel['date'].min()} -> {panel['date'].max()}")

FEATURES = ["icr_z", "edgar_z", "gdelt_z", "stress_beta_z", "xlu_xng_spread", "mom_z", "anc_composite"]

def rank_ic(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def run_fold(name, train_start, train_end, test_start, test_end, target):
    tr_mask = (panel["date"] >= train_start) & (panel["date"] < train_end)
    te_mask = (panel["date"] >= test_start) & (panel["date"] < test_end)
    d_tr = panel[tr_mask].dropna(subset=FEATURES + [target])
    d_te = panel[te_mask].dropna(subset=FEATURES + [target])
    Xtr, ytr = d_tr[FEATURES], d_tr[target]
    Xte, yte = d_te[FEATURES], d_te[target]
    n_test_months = d_te["date"].nunique()
    print(f"\n  [{name}] target={target}  train n={len(Xtr)} ({d_tr['date'].nunique()} months)"
          f"  test n={len(Xte)} ({n_test_months} months)")
    if len(Xte) < 15 or n_test_months < 2:
        print("    insufficient test data, skip"); return

    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42)
    rf.fit(Xtr, ytr)
    print(f"    RF:  in-sample IC={rank_ic(ytr, rf.predict(Xtr)):+.3f}   OOS IC={rank_ic(yte, rf.predict(Xte)):+.3f}")
    print(f"      importances: {dict(zip(FEATURES, rf.feature_importances_.round(3)))}")

    gbm = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    gbm.fit(Xtr, ytr)
    print(f"    GBM: in-sample IC={rank_ic(ytr, gbm.predict(Xtr)):+.3f}   OOS IC={rank_ic(yte, gbm.predict(Xte)):+.3f}")
    print(f"      importances: {dict(zip(FEATURES, gbm.feature_importances_.round(3)))}")

    # classification
    def tercile(df, target):
        d = df.dropna(subset=FEATURES + [target]).copy()
        d["tercile"] = d.groupby("date")[target].transform(lambda x: pd.qcut(x, 3, labels=False, duplicates="drop"))
        d = d[d["tercile"].isin([0, 2])]
        return d[FEATURES], (d["tercile"] == 2).astype(int)
    Xtr_c, ytr_c = tercile(panel[tr_mask], target)
    Xte_c, yte_c = tercile(panel[te_mask], target)
    if len(Xte_c) >= 15 and ytr_c.nunique() == 2 and yte_c.nunique() == 2:
        rfc = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42).fit(Xtr_c, ytr_c)
        gbc = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42).fit(Xtr_c, ytr_c)
        print(f"    RF classifier:  in-sample AUC={roc_auc_score(ytr_c, rfc.predict_proba(Xtr_c)[:,1]):.3f}"
              f"   OOS AUC={roc_auc_score(yte_c, rfc.predict_proba(Xte_c)[:,1]):.3f}")
        print(f"    GBM classifier: in-sample AUC={roc_auc_score(ytr_c, gbc.predict_proba(Xtr_c)[:,1]):.3f}"
              f"   OOS AUC={roc_auc_score(yte_c, gbc.predict_proba(Xte_c)[:,1]):.3f}")
    else:
        print(f"    classification: insufficient tercile data (test n={len(Xte_c)})")

    yshuf = ytr.sample(frac=1.0, random_state=1).values
    rf_shuf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=15, random_state=42).fit(Xtr, yshuf)
    print(f"    shuffled-target importances: {dict(zip(FEATURES, rf_shuf.feature_importances_.round(3)))}")


for target in ["fwd1", "fwd3"]:
    run_fold("FOLD A: train 2022-23, test 2024", "2022-01-01", "2024-01-01", "2024-01-01", "2025-01-01", target)
    run_fold("FOLD B: train 2024-25, test 2026", "2024-01-01", "2026-01-01", "2026-01-01", "2026-09-01", target)
