"""Extend EDGAR 'data center' + 'large load' quarterly counts through 2026Q3."""
import sys
import time

import numpy as np
import pandas as pd
import requests

CIK_BY_TICKER = {
    "NRG": "0001013871", "VST": "0001692819", "CNP": "0001130310", "AEP": "0000004904",
    "EXC": "0001109357", "PPL": "0000922224", "FE": "0001031296", "CEG": "0001868275",
    "TLN": "0001622536", "ETR": "0000065984", "WEC": "0000783325", "DTE": "0000936340",
    "CMS": "0000811156", "PCG": "0001004980", "EIX": "0000827052", "XEL": "0000072903",
    "ES": "0000072741", "ED": "0001047862", "D": "0000715957", "PEG": "0000788784",
    "AEE": "0001002910", "EVRG": "0001711269", "DUK": "0001326160", "SO": "0000092122",
    "NEE": "0000753308",
}
SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
NEW_QUARTERS = pd.period_range("2025Q4", "2026Q3", freq="Q")


def counts_for(cik, keyword, quarters):
    out = {}
    for q in quarters:
        start = q.start_time.strftime("%Y-%m-%d")
        end = q.end_time.strftime("%Y-%m-%d")
        for attempt in range(3):
            try:
                r = requests.get(
                    "https://efts.sec.gov/LATEST/search-index",
                    params={"q": f'"{keyword}"', "forms": "8-K,10-Q,10-K",
                            "dateRange": "custom", "startdt": start, "enddt": end, "ciks": cik},
                    headers=SEC_HEADERS, timeout=20)
                r.raise_for_status()
                out[q.end_time.normalize()] = r.json()["hits"]["total"]["value"]
                break
            except Exception as e:
                time.sleep(2)
        else:
            out[q.end_time.normalize()] = np.nan
        time.sleep(0.25)
    return pd.Series(out)


for kw, fname in [("data center", "edgar_quarterly_mentions.parquet"),
                   ("large load", "edgar_quarterly_large_load.parquet")]:
    path = f"research/narrative_mispricing/{fname}"
    existing = pd.read_parquet(path)
    new_data = {}
    for t, cik in CIK_BY_TICKER.items():
        new_data[t] = counts_for(cik, kw, NEW_QUARTERS)
        print(f"  {kw}/{t}: {dict(new_data[t])}")
    new_df = pd.DataFrame(new_data)
    combined = pd.concat([existing, new_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_parquet(path)
    print(f"Extended {path}: now through {combined.index.max()}")
