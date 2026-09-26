"""
SEC EDGAR filing-list + full-document-text fetcher, generalized from the
existing patterns in this project's research/ (per docs/backlog-pricing-power-
preregistration.md §5 -- not rebuilt from scratch): submissions-API pagination
+ 429 backoff mirrors research/merchant_fundamental_beta/fetch_filing_list.py;
primary-document HTML->text conversion mirrors research/merchant_fundamental_
beta/fetch_10k.py. Those two scripts hardcode 4 merchant-generator tickers;
this generalizes the same approach to all 27 names in the `screen` universe
with on-disk caching so repeat runs are instant.

Not production code. Throwaway research script.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SEC_HEADERS = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
_CACHE = Path(__file__).parent / "_cache"
_FILINGS_CACHE = _CACHE / "filings"
_TEXT_CACHE = _CACHE / "text"
for d in (_CACHE, _FILINGS_CACHE, _TEXT_CACHE):
    d.mkdir(parents=True, exist_ok=True)

FORMS = ("10-K", "10-Q")


def _get(url: str, timeout: int = 30) -> requests.Response:
    r = None
    for i in range(5):
        r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
        if r.status_code == 429:
            time.sleep(1.5 * (i + 1))
            continue
        return r
    return r


def list_filings(ticker: str, cik: str, forms: tuple[str, ...] = FORMS) -> pd.DataFrame:
    """
    All 10-K/10-Q filings for `ticker`, from SEC submissions API (paginated
    via filings.files[] for filers with long histories, same pattern as
    fetch_filing_list.py). Cached to disk keyed by ticker.

    Columns: form, accessionNumber, filingDate, reportDate, primaryDocument.
    """
    cache_path = _FILINGS_CACHE / f"{ticker}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    cik10 = f"{int(cik):010d}"
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = _get(url)
    r.raise_for_status()
    data = r.json()

    rows = []
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form in forms:
            rows.append({
                "form": form,
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": recent["filingDate"][i],
                "reportDate": recent["reportDate"][i],
                "primaryDocument": recent["primaryDocument"][i],
            })
    for extra in data["filings"].get("files", []):
        eurl = f"https://data.sec.gov/submissions/{extra['name']}"
        er = _get(eurl)
        if er is not None and er.status_code == 200:
            edata = er.json()
            for i, form in enumerate(edata.get("form", [])):
                if form in forms:
                    rows.append({
                        "form": form,
                        "accessionNumber": edata["accessionNumber"][i],
                        "filingDate": edata["filingDate"][i],
                        "reportDate": edata["reportDate"][i],
                        "primaryDocument": edata["primaryDocument"][i],
                    })
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        df["reportDate"] = pd.to_datetime(df["reportDate"])
        df = df.sort_values("filingDate").drop_duplicates("accessionNumber").reset_index(drop=True)
    df.to_parquet(cache_path)
    time.sleep(0.2)
    return df


def fetch_document_text(ticker: str, cik: str, accession: str, primary_doc: str) -> str | None:
    """Fetch one filing's primary document and return plain text (cached)."""
    acc_key = accession.replace("-", "")
    cache_path = _TEXT_CACHE / f"{ticker}_{acc_key}.txt"
    if cache_path.exists():
        return cache_path.read_text(errors="ignore")

    cik_int = int(cik)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_key}/{primary_doc}"
    r = _get(url, timeout=60)
    if r is None or r.status_code != 200:
        cache_path.write_text("")
        return None
    try:
        soup = BeautifulSoup(r.content, "lxml")
    except Exception:
        soup = BeautifulSoup(r.content, "html.parser")
    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    full_text = "\n".join(lines)
    cache_path.write_text(full_text)
    time.sleep(0.3)
    return full_text
