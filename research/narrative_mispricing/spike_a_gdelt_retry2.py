"""Final retry pass for the still-missing GDELT tickers, longer spacing throughout."""
import sys
import time

import pandas as pd
import requests

GDELT_NAME = {
    "NRG": "NRG Energy", "EXC": "Exelon", "CEG": "Constellation Energy",
    "DTE": "DTE Energy", "PCG": "PG&E", "XEL": "Xcel Energy",
    "ED": "Consolidated Edison", "D": "Dominion Energy", "EVRG": "Evergy",
    "DUK": "Duke Energy", "SO": "Southern Company",
}

existing = pd.read_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")


def gdelt_volume_retry(name: str, start: str, end: str, max_retries: int = 5) -> pd.Series:
    backoff = 30
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
                backoff *= 1.5
                continue
            r.raise_for_status()
            data = r.json()
            series = data["timeline"][0]["data"]
            return pd.Series({pd.Timestamp(d["date"][:8]): d["value"] for d in series})
        except Exception as e:
            print(f"    error: {e}")
            time.sleep(backoff)
            backoff *= 1.5
    return pd.Series(dtype=float)


results = {}
for t, name in GDELT_NAME.items():
    print(f"{t} ({name})...")
    results[t] = gdelt_volume_retry(name, "20180101", "20250930")
    print(f"  -> {len(results[t])} days")
    time.sleep(20)

merged = existing.copy()
for t in GDELT_NAME:
    if len(results[t]) > 0:
        merged[t] = results[t]
merged.to_parquet("research/narrative_mispricing/gdelt_daily_volume.parquet")
print("\nFinal coverage:", {t: int(merged[t].notna().sum()) for t in merged.columns})
