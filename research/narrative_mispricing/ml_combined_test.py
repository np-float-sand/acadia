"""
Multivariate retest: do the weak/failed signals from this whole session (ICR, narrative
EDGAR+GDELT, merchant stress-beta + XLU/XNG spread, PJM ancillary scarcity) show elevated
predictive power in combination (RF/GBM/classifier/quantile) that none showed standalone
(all |t| < 1.5 in every linear rank-IC test run this session)?

Window: 2022-01 -> 2025-09 only, per user instruction (electrification/AI-power narrative
wasn't really relevant before ChatGPT, Nov 2022) -- NOT the full 2018-2025 window used in
every other test this session. This shrinks the sample further (~40 months), which is
itself the single biggest risk for any ML result here (see the Henry Hub GBM overfit
precedent flagged in the Approach C handoff) -- treated as a first-class concern below,
not a footnote.

Throwaway spike. Findings -> docs/RESEARCH-LOG.md, docs/untried-tasks-log.md.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/Users/sandhyapersad/acadia")
warnings.filterwarnings("ignore")

START, END = "2022-01-01", "2025-09-30"
MONTHLY = pd.date_range(START, END, freq="MS")

# ---------- Load all signals built this session ----------
icr = pd.read_parquet("research/icr_edgar_retest/icr_edgar_panel.parquet")
edgar_dc = pd.read_parquet("research/narrative_mispricing/edgar_quarterly_mentions.parquet")
edgar_ll = pd.read_parquet("research/narrative_mispricing/edgar_quarterly_large_load.parquet")
gdelt = pd.read_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")
betas = pd.read_parquet("research/merchant_beta_orthogonalization/rolling_betas_baseline.parquet")
xlu_xng = pd.read_parquet("research/narrative_mispricing/xlu_xng_spread_continuous.parquet")

tickers = sorted(set(icr.columns) & set(edgar_dc.columns) & set(xlu_xng.columns))
print(f"{len(tickers)} tickers in the intersection universe")

# ---------- Build per-feature monthly panels, 2022-01 -> 2025-09 ----------
def to_monthly_ffill(df, dates=MONTHLY):
    return df.reindex(df.index.union(dates)).sort_index().ffill().reindex(dates)

def xsec_z(frame):
    return frame.apply(lambda row: (row - row.dropna().mean()) / row.dropna().std()
                        if row.dropna().shape[0] >= 5 else pd.Series(index=row.index, dtype=float),
                        axis=1)

icr_monthly = to_monthly_ffill(icr)[tickers]
icr_z = xsec_z(icr_monthly)

edgar_combined = edgar_dc.add(edgar_ll, fill_value=0).sort_index().rolling(4, min_periods=1).sum()
edgar_monthly = to_monthly_ffill(edgar_combined)[tickers]
edgar_z = xsec_z(edgar_monthly)

gdelt_self_mean = gdelt.rolling(756, min_periods=180).mean()
gdelt_self_std = gdelt.rolling(756, min_periods=180).std()
gdelt_self_z = (gdelt - gdelt_self_mean) / gdelt_self_std.replace(0, np.nan)
gdelt_monthly = to_monthly_ffill(gdelt_self_z.clip(-4, 4))
gdelt_cols = [t for t in tickers if t in gdelt_monthly.columns]

stress_beta_wide = betas["stress_beta"].unstack("ticker") if "ticker" in betas.index.names else betas.pivot_table(values="stress_beta", index=betas.index.get_level_values("date"), columns=betas.index.get_level_values("ticker"))
stress_beta_monthly = to_monthly_ffill(stress_beta_wide)[[t for t in tickers if t in stress_beta_wide.columns]]
stress_beta_z = xsec_z(stress_beta_monthly)

xlu_xng_monthly = to_monthly_ffill(xlu_xng)[tickers]  # already a meaningful raw spread, no need to z-score

# Momentum (cheap standard control)
px = yf.download(tickers, start="2020-01-01", end="2025-10-01", auto_adjust=True, progress=False)["Close"]
mom = px.pct_change(126).shift(21)  # 6mo momentum, 1mo lag to avoid overlap w/ fwd return
mom_monthly = to_monthly_ffill(mom)[tickers]
mom_z = xsec_z(mom_monthly)

# ---------- Ancillary composite (ISO-wide, PJM only) ----------
pjm_tickers_map = {"NRG":"ERCOT","VST":"ERCOT","CNP":"ERCOT","AEP":"PJM","EXC":"PJM","PPL":"PJM",
                    "FE":"PJM","CEG":"PJM","TLN":"PJM","D":"PJM","PEG":"PJM"}
try:
    anc21 = pd.read_parquet("research/merchant_beta_orthogonalization/pjm_ancillary_2018_2021.parquet")
    anc25 = pd.read_parquet("research/narrative_mispricing/pjm_ancillary_2022_2025.parquet")
    anc = pd.concat([anc21, anc25], ignore_index=True)
    anc["Interval Start"] = pd.to_datetime(anc["Interval Start"])
    anc["date"] = anc["Interval Start"].dt.date
    rto = anc[anc["Ancillary Service"].str.startswith("PJM_RTO")]
    anc_daily = rto.groupby(["date","Ancillary Service"])["Market Clearing Price"].mean().unstack()
    anc_daily.index = pd.to_datetime(anc_daily.index)
    anc_z_daily = (anc_daily - anc_daily.rolling(252, min_periods=126).mean()) / anc_daily.rolling(252, min_periods=126).std()
    anc_composite = anc_z_daily.mean(axis=1)
    anc_monthly = to_monthly_ffill(anc_composite.to_frame("anc"))["anc"]
    print(f"Ancillary composite: {anc_monthly.notna().sum()}/{len(MONTHLY)} months available (2022-2025 pull included)")
except FileNotFoundError:
    anc_monthly = pd.Series(np.nan, index=MONTHLY)
    print("Ancillary 2022-2025 not yet available -- using NaN placeholder")

# ---------- Forward returns / targets ----------
fwd1 = px.pct_change(21).shift(-21)
fwd3 = px.pct_change(63).shift(-63)
fwd1_monthly = to_monthly_ffill(fwd1)[tickers]  # note: ffill is wrong direction for fwd returns; recompute properly below
# Properly: sample fwd return AT each monthly date directly from daily series (nearest, not ffill)
def sample_at(df, dates):
    return df.reindex(df.index.union(dates)).sort_index().bfill().reindex(dates)
fwd1_monthly = sample_at(fwd1, MONTHLY)[tickers]
fwd3_monthly = sample_at(fwd3, MONTHLY)[tickers]

# ---------- Assemble long panel ----------
rows = []
for d in MONTHLY:
    for t in tickers:
        row = {
            "date": d, "ticker": t,
            "icr_z": icr_z.loc[d, t] if d in icr_z.index and t in icr_z.columns else np.nan,
            "edgar_z": edgar_z.loc[d, t] if d in edgar_z.index and t in edgar_z.columns else np.nan,
            "gdelt_z": gdelt_monthly.loc[d, t] if (t in gdelt_cols and d in gdelt_monthly.index) else np.nan,
            "stress_beta_z": stress_beta_z.loc[d, t] if d in stress_beta_z.index and t in stress_beta_z.columns else np.nan,
            "xlu_xng_spread": xlu_xng_monthly.loc[d, t] if d in xlu_xng_monthly.index and t in xlu_xng_monthly.columns else np.nan,
            "mom_z": mom_z.loc[d, t] if d in mom_z.index and t in mom_z.columns else np.nan,
            "anc_composite": anc_monthly.loc[d] if d in anc_monthly.index and pjm_tickers_map.get(t) == "PJM" else 0.0,
            "fwd1": fwd1_monthly.loc[d, t] if d in fwd1_monthly.index and t in fwd1_monthly.columns else np.nan,
            "fwd3": fwd3_monthly.loc[d, t] if d in fwd3_monthly.index and t in fwd3_monthly.columns else np.nan,
        }
        rows.append(row)
panel = pd.DataFrame(rows)
panel.to_parquet("research/narrative_mispricing/ml_panel_2022_2025.parquet")
print(f"\nPanel shape: {panel.shape}")
print(panel[["icr_z","edgar_z","gdelt_z","stress_beta_z","xlu_xng_spread","mom_z","anc_composite"]].describe())
