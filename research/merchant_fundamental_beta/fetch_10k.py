"""Fetch specific 10-K primary documents and convert to plain text for keyword search."""
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
OUT = Path("research/merchant_fundamental_beta/_filings")
OUT.mkdir(exist_ok=True, parents=True)

# (ticker, cik, accession, primary_document, report_date)
TARGETS = [
    ("NRG", "1013871", "0001013871-24-000005", "nrg-20231231.htm", "2023-12-31"),
    ("NRG", "1013871", "0001013871-25-000006", "nrg-20241231.htm", "2024-12-31"),
    ("NRG", "1013871", "0001013871-26-000004", "nrg-20251231.htm", "2025-12-31"),
    ("VST", "1692819", "0001692819-24-000012", "vistra-20231231.htm", "2023-12-31"),
    ("VST", "1692819", "0001692819-25-000013", "vistra-20241231.htm", "2024-12-31"),
    ("VST", "1692819", "0001692819-26-000006", "vistra-20251231.htm", "2025-12-31"),
    ("CEG", "1868275", "0001868275-24-000014", "ceg-20231231.htm", "2023-12-31"),
    ("CEG", "1868275", "0001868275-25-000023", "ceg-20241231.htm", "2024-12-31"),
    ("CEG", "1868275", "0001868275-26-000032", "ceg-20251231.htm", "2025-12-31"),
    ("TLN", "1622536", "0001628280-25-008786", "tln-20241231.htm", "2024-12-31"),
    ("TLN", "1622536", "0001622536-26-000017", "tln-20251231.htm", "2025-12-31"),
]

for ticker, cik, accession, doc, report_date in TARGETS:
    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
    out_html = OUT / f"{ticker}_{report_date}.htm"
    out_txt = OUT / f"{ticker}_{report_date}.txt"
    if out_txt.exists():
        print(f"skip (cached): {ticker} {report_date}")
        continue
    print(f"fetching {ticker} {report_date}: {url}")
    for attempt in range(5):
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    out_html.write_bytes(r.content)
    soup = BeautifulSoup(r.content, "lxml")
    text = soup.get_text("\n")
    # collapse excess blank lines
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    out_txt.write_text("\n".join(lines))
    print(f"  saved {len(lines)} lines, {out_txt.stat().st_size} bytes")
    time.sleep(0.4)

print("done")
