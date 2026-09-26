"""Retry GDELT pulls for tickers that 429'd on the first pass, with real backoff."""
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, "/Users/sandhyapersad/acadia")

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
missing = [t for t in GDELT_NAME if existing[t].notna().sum() == 0] if set(GDELT_NAME) <= set(existing.columns) else list(GDELT_NAME)
print("missing tickers:", missing)


def gdelt_volume_retry(name: str, start: str, end: str, max_retries: int = 4) -> pd.Series:
    backoff = 25
    for attempt in range(max_retries):
        try:
            r = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={"query": name, "mode": "timelinevol", "format": "json",
                        "startdatetime": start + "000000", "enddatetime": end + "000000"},
                timeout=30,
            )
            if r.status_code == 429:
                print(f"    429, backing off {backoff}s (attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
                backoff *= 1.6
                continue
            r.raise_for_status()
            data = r.json()
            series = data["timeline"][0]["data"]
            return pd.Series({pd.Timestamp(d["date"][:8]): d["value"] for d in series})
        except Exception as e:
            print(f"    error: {e}")
            time.sleep(backoff)
            backoff *= 1.6
    return pd.Series(dtype=float)


results = {}
for t in missing:
    print(f"{t} ({GDELT_NAME[t]})...")
    results[t] = gdelt_volume_retry(GDELT_NAME[t], "20180101", "20250930")
    print(f"  -> {len(results[t])} days")
    time.sleep(12)  # generous spacing between tickers regardless of outcome

new_df = pd.DataFrame(results)
merged = existing.copy()
for t in missing:
    merged[t] = new_df[t]
merged.to_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")
print("\nFinal coverage:", {t: int(merged[t].notna().sum()) for t in GDELT_NAME})
