"""
Item 9, Approach A: narrative-extremity short -- the actual rank-IC test.

Composite per ticker per month = avg(z(trailing-4Q EDGAR "data center" mentions),
                                      z(trailing-63d GDELT news-volume))
Cross-sectional rank across the ~22-25 name universe each month.
Hypothesis (short thesis): high composite -> WORSE forward return (negative IC).

Throwaway spike. Findings -> docs/RESEARCH-LOG.md, docs/untried-tasks-log.md item 9.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/sandhyapersad/acadia")

edgar = pd.read_parquet("research/narrative_mispricing/edgar_quarterly_mentions.parquet")
gdelt = pd.read_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")

tickers = list(edgar.columns)
usable_gdelt = [t for t in tickers if gdelt[t].notna().sum() > 100]
print(f"{len(usable_gdelt)}/{len(tickers)} tickers have usable GDELT data: {usable_gdelt}")

# --- Build monthly composite ---
monthly_dates = pd.date_range("2018-06-01", "2025-09-01", freq="MS")

# EDGAR: trailing rolling-4-quarter sum, forward-filled to monthly
edgar_ff = edgar.sort_index().rolling(4, min_periods=1).sum()
edgar_monthly = edgar_ff.reindex(edgar_ff.index.union(monthly_dates)).sort_index().ffill().reindex(monthly_dates)

# GDELT: trailing 63-day mean, resampled to monthly
gdelt_roll = gdelt.rolling(63, min_periods=30).mean()
gdelt_monthly = gdelt_roll.reindex(gdelt_roll.index.union(monthly_dates)).sort_index().ffill().reindex(monthly_dates)

def xsec_z(row):
    valid = row.dropna()
    if len(valid) < 5:
        return pd.Series(index=row.index, dtype=float)
    return (row - valid.mean()) / valid.std()

edgar_z = edgar_monthly.apply(xsec_z, axis=1)
gdelt_z = gdelt_monthly[usable_gdelt].apply(xsec_z, axis=1)

composite = pd.DataFrame(index=monthly_dates, columns=tickers, dtype=float)
for d in monthly_dates:
    e = edgar_z.loc[d] if d in edgar_z.index else pd.Series(dtype=float)
    g = gdelt_z.loc[d] if d in gdelt_z.index else pd.Series(dtype=float)
    both = pd.concat([e.rename("e"), g.rename("g")], axis=1)
    composite.loc[d] = both.mean(axis=1, skipna=True)

composite.to_parquet("research/narrative_mispricing/composite_monthly.parquet")
print("\nComposite describe (all tickers, all months):")
print(composite.stack().describe())

# --- Fetch prices, compute forward returns ---
px = yf.download(tickers, start="2018-01-01", end="2025-10-01", auto_adjust=True, progress=False)["Close"]
monthly_px = px.reindex(px.index.union(monthly_dates)).sort_index().ffill().reindex(monthly_dates)

print("\n=== Cross-sectional rank-IC: composite_t vs fwd return ===")
for h in [1, 3, 6]:
    fwd = monthly_px.pct_change(h).shift(-h)
    ics = []
    for d in monthly_dates:
        sig = composite.loc[d].dropna()
        ret = fwd.loc[d].reindex(sig.index).dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < 8:
            continue
        rho = sig.loc[common].astype(float).corr(ret.loc[common], method="spearman")
        ics.append(rho)
    ics = pd.Series(ics).dropna()
    if len(ics) < 5:
        print(f"{h}mo: insufficient data"); continue
    mean_ic = ics.mean()
    t_stat = mean_ic / (ics.std() / np.sqrt(len(ics)))
    print(f"{h}mo fwd: n_months={len(ics)}  mean IC={mean_ic:+.4f}  t={t_stat:+.2f}  %neg={100*(ics<0).mean():.0f}%")

# --- VST/NRG narrative-flip-window check ---
print("\n=== VST/NRG composite score around documented narrative-flip windows ===")
flip_windows = ["2023-09-01", "2023-12-01", "2024-04-01", "2024-06-01",
                "2024-12-01", "2025-02-01", "2025-04-01", "2025-06-01"]
for t in ["VST", "NRG"]:
    vals = composite[t].reindex(pd.to_datetime(flip_windows), method="nearest")
    print(f"{t}: {dict(zip(flip_windows, vals.round(2).values))}")
