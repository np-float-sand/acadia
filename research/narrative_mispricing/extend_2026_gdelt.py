"""Extend GDELT daily news volume through today (2026-09-08)."""
import time
import pandas as pd
import requests

GDELT_NAME = {
    "NRG": "NRG Energy", "VST": "Vistra Corp", "CNP": "CenterPoint Energy",
    "AEP": "American Electric Power", "EXC": "Exelon", "PPL": "PPL Corp",
    "FE": "FirstEnergy", "CEG": "Constellation Energy", "TLN": "Talen Energy",
    "ETR": "Entergy", "WEC": "WEC Energy", "DTE": "DTE Energy", "CMS": "CMS Energy",
    "PCG": "PG&E", "EIX": "Edison International", "XEL": "Xcel Energy",
    "ES": "Eversource Energy", "ED": "Consolidated Edison", "D": "Dominion Energy",
    "PEG": "Public Service Enterprise Group", "AEE": "Ameren", "EVRG": "Evergy",
    "DUK": "Duke Energy", "SO": "Southern Company", "NEE": "NextEra Energy",
}

existing = pd.read_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")


def gdelt_extend(name, start, end, max_retries=5):
    backoff = 25
    for attempt in range(max_retries):
        try:
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": name, "mode": "timelinevol", "format": "json",
                        "startdatetime": start + "000000", "enddatetime": end + "000000"},
                timeout=30)
            if r.status_code == 429:
                print(f"    429, backing off {backoff}s")
                time.sleep(backoff); backoff *= 1.5; continue
            r.raise_for_status()
            series = r.json()["timeline"][0]["data"]
            return pd.Series({pd.Timestamp(d["date"][:8]): d["value"] for d in series})
        except Exception as e:
            print(f"    error: {e}")
            time.sleep(backoff); backoff *= 1.5
    return pd.Series(dtype=float)


results = {}
for t, name in GDELT_NAME.items():
    print(f"{t} ({name})...")
    s = gdelt_extend(name, "20251001", "20260908")
    print(f"  -> {len(s)} days")
    results[t] = s
    time.sleep(15)

merged = existing.copy()
for t, s in results.items():
    if len(s) > 0:
        combined = pd.concat([merged[t], s])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        merged[t] = combined
merged = merged.reindex(merged.index.union(pd.concat(list(results.values())).index if results else merged.index))
merged.to_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")
print("Extended, coverage:", {t: int(merged[t].notna().sum()) for t in merged.columns})
