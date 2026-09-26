"""
Extend the Approach-C narrative composite through 2026-08 for an out-of-sample
2026-YTD holdout test (user request 2026-09-08). THROWAWAY.

Reproduces the exact `composite_monthly_refined.parquet` recipe (spike_a_refined_test.py):
  EDGAR combined = trailing-4Q ("data center" + "large load") mention counts, cross-sectional z
  GDELT          = each ticker's own trailing-756d self-baseline z, clipped +-4
  composite      = mean(edgar_z, gdelt_self_z) per (month, ticker), skipna
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import requests

HERE = "research/narrative_mispricing"
SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
CIK = {
    "NRG": "0001013871", "VST": "0001692819", "CNP": "0001130310", "AEP": "0000004904",
    "EXC": "0001109357", "PPL": "0000922224", "FE": "0001031296", "CEG": "0001868275",
    "TLN": "0001622536", "ETR": "0000065984", "WEC": "0000783325", "DTE": "0000936340",
    "CMS": "0000811156", "PCG": "0001004980", "EIX": "0000827052", "XEL": "0000072903",
    "ES": "0000072741", "ED": "0001047862", "D": "0000715957", "PEG": "0000788784",
    "AEE": "0001002910", "EVRG": "0001711269", "DUK": "0001326160", "SO": "0000092122",
    "NEE": "0000753308",
}
GDELT_NAME = {
    "NRG": "NRG Energy", "VST": "Vistra", "CNP": "CenterPoint Energy",
    "AEP": "American Electric Power", "EXC": "Exelon", "PPL": "PPL Corp",
    "FE": "FirstEnergy", "CEG": "Constellation Energy", "TLN": "Talen Energy",
    "ETR": "Entergy", "WEC": "WEC Energy", "DTE": "DTE Energy", "CMS": "CMS Energy",
    "PCG": "PG&E", "EIX": "Edison International", "XEL": "Xcel Energy",
    "ES": "Eversource Energy", "ED": "Consolidated Edison", "D": "Dominion Energy",
    "PEG": "Public Service Enterprise Group", "AEE": "Ameren", "EVRG": "Evergy",
    "DUK": "Duke Energy", "SO": "Southern Company", "NEE": "NextEra Energy",
}
NEW_Q = pd.period_range("2025Q4", "2026Q2", freq="Q")
END_MONTH = "2026-08-01"


def edgar_counts(cik: str, keyword: str, quarters) -> pd.Series:
    out = {}
    for q in quarters:
        try:
            r = requests.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": f'"{keyword}"', "forms": "8-K,10-Q,10-K", "dateRange": "custom",
                        "startdt": q.start_time.strftime("%Y-%m-%d"),
                        "enddt": q.end_time.strftime("%Y-%m-%d"), "ciks": cik},
                headers=SEC_HEADERS, timeout=20)
            r.raise_for_status()
            out[q.end_time.normalize()] = r.json()["hits"]["total"]["value"]
        except Exception as e:  # noqa: BLE001
            print(f"  [edgar] {cik} {keyword} {q}: {e}")
            out[q.end_time.normalize()] = np.nan
        time.sleep(0.28)
    return pd.Series(out)


def extend_edgar():
    for kw, fn in [("data center", "edgar_quarterly_mentions.parquet"),
                   ("large load", "edgar_quarterly_large_load.parquet")]:
        old = pd.read_parquet(f"{HERE}/{fn}")
        print(f"=== {kw}: extending {fn} (had -> {old.index.max().date()}) ===")
        add = {}
        for t in CIK:
            add[t] = edgar_counts(CIK[t], kw, NEW_Q)
            print(f"  {t}: +{add[t].fillna(0).astype(int).tolist()}")
        new = pd.DataFrame(add)
        merged = pd.concat([old, new[~new.index.isin(old.index)]]).sort_index()
        merged.to_parquet(f"{HERE}/{fn}")
        print(f"  -> now {merged.index.min().date()}..{merged.index.max().date()} ({len(merged)} q)")


def extend_gdelt():
    old = pd.read_parquet(f"{HERE}/gdelt_daily_volume.parquet")
    print(f"=== GDELT: re-fetching timeline through 2026-08 (had -> {old.index.max().date()}) ===")
    fetched: dict[str, pd.Series] = {}
    for t, name in GDELT_NAME.items():
        backoff = 20
        got = pd.Series(dtype=float)
        for _ in range(5):
            try:
                r = requests.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params={"query": name, "mode": "timelinevol", "format": "json",
                            "startdatetime": "20180101000000", "enddatetime": "20260831000000"},
                    timeout=30)
                if r.status_code == 429:
                    print(f"  {t}: 429, backoff {backoff}s"); time.sleep(backoff); backoff *= 1.6; continue
                r.raise_for_status()
                series = r.json()["timeline"][0]["data"]
                got = pd.Series({pd.Timestamp(d["date"][:8]): float(d["value"]) for d in series})
                break
            except Exception as e:  # noqa: BLE001
                print(f"  {t}: {e}"); time.sleep(backoff); backoff *= 1.6
        if len(got):
            fetched[t] = got.sort_index()
            print(f"  {t}: {got.index.min().date()}..{got.index.max().date()} ({len(got)}d), "
                  f"post-2025-09 pts={int((got.index > pd.Timestamp('2025-09-30')).sum())}")
        else:
            print(f"  {t}: FAILED to fetch (keeping old column)")
        time.sleep(12)

    all_idx = old.index
    for s in fetched.values():
        all_idx = all_idx.union(s.index)
    merged = old.reindex(all_idx).sort_index()
    for t, s in fetched.items():
        merged[t] = s.reindex(all_idx).combine_first(merged[t])
    merged.to_parquet(f"{HERE}/gdelt_daily_volume.parquet")
    print(f"  -> now {merged.index.min().date()}..{merged.index.max().date()} ({len(merged)} rows)")


def rebuild_composite():
    dc = pd.read_parquet(f"{HERE}/edgar_quarterly_mentions.parquet")
    ll = pd.read_parquet(f"{HERE}/edgar_quarterly_large_load.parquet")
    gd = pd.read_parquet(f"{HERE}/gdelt_daily_volume.parquet")

    edgar = dc.add(ll, fill_value=0)
    tickers = list(edgar.columns)
    usable_gdelt = [t for t in tickers if gd[t].notna().sum() > 100]

    months = pd.date_range("2018-06-01", END_MONTH, freq="MS")
    edgar_ff = edgar.sort_index().rolling(4, min_periods=1).sum()
    edgar_m = (edgar_ff.reindex(edgar_ff.index.union(months)).sort_index()
               .ffill().reindex(months))

    gmean = gd.rolling(756, min_periods=180).mean()
    gstd = gd.rolling(756, min_periods=180).std()
    gz = (gd - gmean) / gstd.replace(0, np.nan)
    gz_m = gz.reindex(gz.index.union(months)).sort_index().ffill().reindex(months)

    def xsec_z(row):
        v = row.dropna()
        if len(v) < 5:
            return pd.Series(index=row.index, dtype=float)
        return (row - v.mean()) / v.std()

    edgar_z = edgar_m.apply(xsec_z, axis=1)
    gz_clip = gz_m[usable_gdelt].clip(-4, 4)

    comp = pd.DataFrame(index=months, columns=tickers, dtype=float)
    for d in months:
        e = edgar_z.loc[d] if d in edgar_z.index else pd.Series(dtype=float)
        g = gz_clip.loc[d] if d in gz_clip.index else pd.Series(dtype=float)
        both = pd.concat([e.rename("e"), g.rename("g")], axis=1)
        comp.loc[d] = both.mean(axis=1, skipna=True)

    comp.to_parquet(f"{HERE}/composite_monthly_refined_2026.parquet")
    print(f"\ncomposite rebuilt: {comp.index.min().date()}..{comp.index.max().date()} "
          f"({len(comp)} months), 2026 rows:")
    print(comp.loc[comp.index >= "2026-01-01"].round(2).to_string())


if __name__ == "__main__":
    import sys
    steps = sys.argv[1:] or ["edgar", "gdelt", "composite"]
    if "edgar" in steps:
        extend_edgar()
    if "gdelt" in steps:
        extend_gdelt()
    if "composite" in steps:
        rebuild_composite()
