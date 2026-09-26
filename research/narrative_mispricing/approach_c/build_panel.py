"""
Item 9 Approach C (broadened): narrative composite + every reasonably-alignable
built feature from RESEARCH-LOG.md, as a multivariate GBM/RF panel on the
grid_resilience 25-name universe.

Pre-registered BEFORE looking at any IC number below:
  - Two windows: FULL (2018-06 -> 2025-09, n=88 months) and RECENT (2023-01 ->
    2025-09, n~=33 months), matching the hhub-work discipline that already
    caught one overfit ("2023-26 flicker" that vanished on the full sample).
  - Walk-forward only (sklearn TimeSeriesSplit(5) on the month axis) -- no
    random k-fold, given ~90 autocorrelated monthly cross-sections.
  - OOF rank-IC = mean cross-sectional Spearman(pred, fwd_ret) across the 25
    tickers each held-out month, HAC(Newey-West, 3 lags) t-stat across months.
  - PASS bar: |t| >= 2 on OOF rank-IC in BOTH windows (same sign), AND the
    observed IC beats a 200-draw within-month permutation null at p < 0.05 in
    BOTH windows. Anything short of that is FAIL, full stop -- a recent-window
    flicker alone is exactly the trap this project already hit once.
  - Report full-feature (~10) and lean (4) models, per the precedent that a
    smaller feature set is the more honest test on this little data.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("research/narrative_mispricing/approach_c")
REPO = Path(__file__).resolve().parents[2]

TICKERS = ['NRG','VST','CNP','AEP','EXC','PPL','FE','CEG','TLN','ETR','WEC','DTE',
           'CMS','PCG','EIX','XEL','ES','ED','D','PEG','AEE','EVRG','DUK','SO','NEE']

# ---- 1. narrative composite (per-ticker) ----
narr = pd.read_parquet(REPO / "research/narrative_mispricing/composite_monthly_refined.parquet")
narr.index = pd.to_datetime(narr.index).to_period("M").to_timestamp("M")
months = narr.index  # master month grid: 88 months, 2018-06 .. 2025-09

# ---- 2. ICR (per-ticker, quarterly -> ffill monthly) ----
icr = pd.read_parquet(REPO / "research/icr_edgar_retest/icr_edgar_panel.parquet")
icr.index = pd.to_datetime(icr.index)
icr_m = icr.reindex(pd.date_range(icr.index.min(), months.max(), freq="D")).ffill()
icr_m = icr_m.resample("ME").last().reindex(months)

# ---- 3. stress beta (per-ticker; multiple ISO rows possible -> mean) ----
sb = pd.read_parquet(REPO / "research/merchant_beta_orthogonalization/rolling_betas_baseline.parquet")
sb = sb.reset_index()
sb["date"] = pd.to_datetime(sb["date"]).dt.to_period("M").dt.to_timestamp("M")
sb_piv = sb.groupby(["date", "ticker"])["stress_beta"].mean().unstack("ticker")
sb_piv = sb_piv.reindex(months)

# ---- 4. GSI factor score (per-ticker, production pipeline, fresh run) ----
fs = pd.read_parquet(OUT / "factor_scores_fresh.parquet")
fs["date"] = pd.to_datetime(fs["date"]).dt.to_period("M").dt.to_timestamp("M")
fs_piv = fs.groupby(["date", "ticker"])["factor_score"].mean().unstack("ticker")
fs_piv = fs_piv.reindex(months)

# ---- 5. prices -> own 6mo momentum + fwd 1mo/3mo returns (per-ticker) ----
prices = pd.read_parquet(OUT / "prices_25.parquet")
prices.index = pd.to_datetime(prices.index)
monthly_px = prices.resample("ME").last().reindex(
    pd.date_range(prices.index.min(), months.max(), freq="ME")
)
mom6 = monthly_px.pct_change(6).reindex(months)
fwd1 = monthly_px.pct_change(1).shift(-1).reindex(months)
fwd3 = monthly_px.pct_change(3).shift(-3).reindex(months)

# ---- 6-10. market-wide macro features (broadcast to every ticker) ----
def _load_macro(fname, col):
    s = pd.read_parquet(OUT / fname)[col]
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    return s.reindex(months)

capex_z   = _load_macro("capex_guidance_monthly.parquet", "capex_guidance_z")
catdem_z  = _load_macro("category_demand_monthly.parquet", "category_demand_z")
ancil_z   = _load_macro("ancillary_monthly.parquet", "ancillary_pjm_z")
realyld   = _load_macro("real_yield_monthly.parquet", "real_yield_chg6m")
ng_mom    = _load_macro("ng_monthly.parquet", "ng_mom6m")

# ---- assemble long panel: one row per (month, ticker) ----
rows = []
for t in TICKERS:
    if t not in narr.columns:
        continue
    d = pd.DataFrame(index=months)
    d["ticker"] = t
    d["narrative_z"] = narr[t]
    d["icr"] = icr_m[t] if t in icr_m.columns else np.nan
    d["stress_beta"] = sb_piv[t] if t in sb_piv.columns else np.nan
    d["gsi_factor"] = fs_piv[t] if t in fs_piv.columns else np.nan
    d["own_mom6m"] = mom6[t] if t in mom6.columns else np.nan
    d["capex_guidance_z"] = capex_z
    d["category_demand_z"] = catdem_z
    d["ancillary_pjm_z"] = ancil_z
    d["real_yield_chg6m"] = realyld
    d["ng_mom6m"] = ng_mom
    d["fwd_ret_1m"] = fwd1[t] if t in fwd1.columns else np.nan
    d["fwd_ret_3m"] = fwd3[t] if t in fwd3.columns else np.nan
    rows.append(d.reset_index().rename(columns={"index": "month"}))

panel = pd.concat(rows, ignore_index=True)
panel.to_parquet(OUT / "panel.parquet")
print("panel shape:", panel.shape)
print("months:", panel['month'].nunique(), panel['month'].min(), panel['month'].max())
print("non-null counts:\n", panel.drop(columns=['ticker']).notna().sum())
