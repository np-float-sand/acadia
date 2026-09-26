"""
Approach A, refined: broader EDGAR keywords ("data center" + "large load" -- dropped
"hyperscaler", near-zero everywhere, adds nothing) + GDELT self-baseline normalization
(each ticker's own rolling z-score, not cross-sectional) instead of raw cross-sectional
GDELT volume, which penalized smaller/less-universally-covered names regardless of
narrative intensity.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

edgar_dc = pd.read_parquet("research/narrative_mispricing/edgar_quarterly_mentions.parquet")
edgar_ll = pd.read_parquet("research/narrative_mispricing/edgar_quarterly_large_load.parquet")
gdelt = pd.read_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")

edgar_combined = edgar_dc.add(edgar_ll, fill_value=0)
tickers = list(edgar_combined.columns)
usable_gdelt = [t for t in tickers if gdelt[t].notna().sum() > 100]
print(f"EDGAR combined ('data center' + 'large load') totals:\n{edgar_combined.sum().sort_values(ascending=False)}")

monthly_dates = pd.date_range("2018-06-01", "2025-09-01", freq="MS")

edgar_ff = edgar_combined.sort_index().rolling(4, min_periods=1).sum()
edgar_monthly = edgar_ff.reindex(edgar_ff.index.union(monthly_dates)).sort_index().ffill().reindex(monthly_dates)

# GDELT: SELF-baseline z-score (own trailing 756d mean/std), not cross-sectional
gdelt_own_mean = gdelt.rolling(756, min_periods=180).mean()
gdelt_own_std = gdelt.rolling(756, min_periods=180).std()
gdelt_self_z = (gdelt - gdelt_own_mean) / gdelt_own_std.replace(0, np.nan)
gdelt_self_monthly = gdelt_self_z.reindex(gdelt_self_z.index.union(monthly_dates)).sort_index().ffill().reindex(monthly_dates)

def xsec_z(row):
    valid = row.dropna()
    if len(valid) < 5:
        return pd.Series(index=row.index, dtype=float)
    return (row - valid.mean()) / valid.std()

edgar_z = edgar_monthly.apply(xsec_z, axis=1)  # EDGAR still cross-sectional (counts are inherently comparable-ish)
# GDELT is now already self-normalized (comparable across tickers by construction) -- use directly, just clip
gdelt_self_clipped = gdelt_self_monthly[usable_gdelt].clip(-4, 4)

composite = pd.DataFrame(index=monthly_dates, columns=tickers, dtype=float)
for d in monthly_dates:
    e = edgar_z.loc[d] if d in edgar_z.index else pd.Series(dtype=float)
    g = gdelt_self_clipped.loc[d] if d in gdelt_self_clipped.index else pd.Series(dtype=float)
    both = pd.concat([e.rename("e"), g.rename("g")], axis=1)
    composite.loc[d] = both.mean(axis=1, skipna=True)

composite.to_parquet("research/narrative_mispricing/composite_monthly_refined.parquet")

px = yf.download(tickers, start="2018-01-01", end="2025-10-01", auto_adjust=True, progress=False)["Close"]
monthly_px = px.reindex(px.index.union(monthly_dates)).sort_index().ffill().reindex(monthly_dates)

print("\n=== REFINED: Cross-sectional rank-IC: composite_t vs fwd return ===")
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

print("\n=== VST/NRG refined composite around narrative-flip windows ===")
flip_windows = ["2023-09-01", "2023-12-01", "2024-04-01", "2024-06-01",
                "2024-12-01", "2025-02-01", "2025-04-01", "2025-06-01"]
for t in ["VST", "NRG"]:
    vals = composite[t].reindex(pd.to_datetime(flip_windows), method="nearest")
    print(f"{t}: {dict(zip(flip_windows, vals.round(2).values))}")
