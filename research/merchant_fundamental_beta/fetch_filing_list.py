"""Fetch the list of 10-K filings for VST/NRG/CEG/TLN from EDGAR submissions API."""
import json
import time

import requests

HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}

CIKS = {
    "NRG": "0001013871",
    "VST": "0001692819",
    "CEG": "0001868275",
    "TLN": "0001622536",
}

for ticker, cik in CIKS.items():
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    recent = data["filings"]["recent"]
    rows = []
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            rows.append({
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": recent["filingDate"][i],
                "reportDate": recent["reportDate"][i],
                "primaryDocument": recent["primaryDocument"][i],
            })
    # older filings may be in additional files under filings.files
    for extra in data["filings"].get("files", []):
        eurl = f"https://data.sec.gov/submissions/{extra['name']}"
        er = requests.get(eurl, headers=HEADERS, timeout=30)
        if er.status_code == 200:
            edata = er.json()
            for i, form in enumerate(edata.get("form", [])):
                if form == "10-K":
                    rows.append({
                        "accessionNumber": edata["accessionNumber"][i],
                        "filingDate": edata["filingDate"][i],
                        "reportDate": edata["reportDate"][i],
                        "primaryDocument": edata["primaryDocument"][i],
                    })
        time.sleep(0.3)

    print(f"\n{ticker} ({cik}): {len(rows)} 10-Ks found")
    for row in sorted(rows, key=lambda r: r["filingDate"]):
        print(" ", row)
    with open(f"research/merchant_fundamental_beta/_cache/{ticker}_10k_list.json", "w") as f:
        json.dump(rows, f, indent=2)
    time.sleep(0.3)
