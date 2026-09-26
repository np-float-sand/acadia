"""EDGAR access: SIC populations, full-text discovery, and company facts.

Every function here is a thin, cached network wrapper. All *logic* lives in
`universe.py` / `fundamentals.py` so it can be tested without a network.

Point-in-time discipline:
  - SIC codes are assigned by the filer at filing time.
  - The full-text lexicon is queried per universe-year over filings *published
    before* that year, so a name joins only once it has said so in a filing.
  - Company facts carry a `filed` date, which is what `fundamentals` keys on.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests

from capex_intensity_strategy import config

_SIC_BROWSE = "https://www.sec.gov/cgi-bin/browse-edgar"
_FTS = "https://efts.sec.gov/LATEST/search-index"
_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_TICKERS = "https://www.sec.gov/files/company_tickers.json"

_COMPANY_INFO_RE = re.compile(r"<company-info[^>]*>(.*?)</company-info>", re.S)
_CIK_RE = re.compile(r"<cik>(\d+)</cik>")


def _headers(accept_json: bool = True) -> dict[str, str]:
    """SEC headers.

    `www.sec.gov/files/*` rejects an explicit `Accept: application/json` and
    returns a non-JSON body, while `efts.sec.gov` (full-text search) requires it.
    Callers hitting www.sec.gov pass accept_json=False.
    """
    headers = {"User-Agent": config.SEC_USER_AGENT}
    if accept_json:
        headers["Accept"] = "application/json"
    return headers


def _cached(name: str):
    return config.CACHE_DIR / name


def ticker_to_cik(refresh: bool = False) -> dict[str, str]:
    """Map upper-case ticker -> zero-padded CIK for every SEC-registered filer."""
    path = _cached("ticker_cik.json")
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    payload = requests.get(_TICKERS, headers=_headers(accept_json=False),
                           timeout=60).json()
    out: dict[str, str] = {}
    for row in payload.values():
        out.setdefault(row["ticker"].upper(), str(row["cik_str"]).zfill(10))
    path.write_text(json.dumps(out))
    return out


def sic_population(sic: int, max_pages: int = 9) -> set[str]:
    """Every 10-K filer CIK carrying this SIC code, including delisted ones."""
    found: set[str] = set()
    for start in range(0, max_pages * 100, 100):
        params = {
            "action": "getcompany", "SIC": str(sic), "type": "10-K", "dateb": "",
            "owner": "include", "count": "100", "start": str(start), "output": "atom",
        }
        resp = requests.get(_SIC_BROWSE, params=params,
                            headers=_headers(accept_json=False), timeout=60)
        if resp.status_code != 200:
            break
        blocks = _COMPANY_INFO_RE.findall(resp.text)
        if not blocks:
            break
        for block in blocks:
            match = _CIK_RE.search(block)
            if match:
                found.add(match.group(1).zfill(10))
        if len(blocks) < 100:
            break
        time.sleep(config.SEC_RATE_LIMIT_SLEEP)
    return found


def full_text_ciks(phrase: str, start: str, end: str, max_pages: int = 5) -> set[str]:
    """CIKs whose filings in [start, end] contain an exact phrase."""
    found: set[str] = set()
    for offset in range(0, max_pages * 10, 10):
        params = {
            "q": phrase, "forms": config.LEXICON_FORMS,
            "startdt": start, "enddt": end, "from": str(offset),
        }
        resp = requests.get(_FTS, params=params, headers=_headers(), timeout=60)
        if resp.status_code != 200:
            break
        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            for cik in hit.get("_source", {}).get("ciks", []):
                found.add(str(cik).zfill(10))
        if len(hits) < 10:
            break
        time.sleep(0.25)
    return found


# XBRL tags per item, in preference order. Every tag is parsed -- a filer that
# reports `Revenues` only as an annual total still has its ASC-606 quarters
# picked up -- and `fundamentals.ttm_panel` then keeps, per filer and per
# observation date, the single tag with the most visible quarters. Keeping one
# tag per series avoids mixing two differently-scoped measures inside one TTM sum.
#
# `PaymentsToAcquireProductiveAssets` is the only capex tag HUBB, GEV, CARR, JCI,
# EME and WCC use, and is where AEP reports its construction expenditure.
FACT_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    # Only used by the mechanical `screen_auto` construction, as the
    # point-in-time replacement for a hand-recorded "is it earnings-valued" flag.
    "net_income": ("NetIncomeLoss",),
}

_FACT_COLUMNS = ["item", "tag", "rank", "start", "end", "filed", "value"]


def parse_mention_count(payload: dict) -> int:
    """Number of matching filings in a full-text-search response."""
    return int(payload.get("hits", {}).get("total", {}).get("value", 0) or 0)


def parse_mention_dates(payload: dict) -> list[str]:
    """Filing dates of every hit in a full-text-search response."""
    return [h["_source"]["file_date"]
            for h in payload.get("hits", {}).get("hits", [])
            if h.get("_source", {}).get("file_date")]


def filer_mention_dates(phrase: str, cik: str, start: str, end: str,
                        forms: str = "10-K") -> list[str]:
    """Dates of ONE filer's filings in [start, end] containing an exact phrase.

    A filer publishes one 10-K a year, so a decade fits in a single page of hits.
    That makes the whole vocabulary screen one request per (filer, phrase) rather
    than one per (filer, phrase, year).
    """
    params = {"q": phrase, "forms": forms, "startdt": start, "enddt": end,
              "ciks": str(int(cik)).zfill(10)}
    resp = requests.get(_FTS, params=params, headers=_headers(), timeout=60)
    if resp.status_code != 200:
        return []
    return parse_mention_dates(resp.json())


def filer_mention_count(phrase: str, cik: str, start: str, end: str,
                        forms: str = "10-K") -> int:
    """How many of ONE filer's filings in [start, end] contain an exact phrase.

    The CIK-filtered form of `full_text_ciks`. Asking per filer costs one request
    instead of paging an unbounded result set, which is what makes a common phrase
    ("data center") usable as a screen at all.
    """
    params = {"q": phrase, "forms": forms, "startdt": start, "enddt": end,
              "ciks": str(int(cik)).zfill(10)}
    resp = requests.get(_FTS, params=params, headers=_headers(), timeout=60)
    if resp.status_code != 200:
        return 0
    return parse_mention_count(resp.json())


def parse_company_facts(gaap: dict) -> pd.DataFrame:
    """Pure parse of a companyfacts `us-gaap` block into quarterly USD flow facts.

    Every 10-Q/10-K USD flow fact up to a full year is kept -- discrete quarters
    and the year-to-date/annual ladder alike, because cash-flow items (capex) are
    reported cumulatively and only become quarters after differencing in
    `fundamentals.derive_quarters`. `rank` is the tag's position in `FACT_TAGS`,
    used downstream only to break ties.
    """
    rows = []
    for item, tags in FACT_TAGS.items():
        for rank, tag in enumerate(tags):
            if tag not in gaap:
                continue
            for unit, entries in gaap[tag].get("units", {}).items():
                if unit != "USD":
                    continue
                for entry in entries:
                    if entry.get("form") not in ("10-Q", "10-K"):
                        continue
                    if not all(entry.get(k) for k in ("start", "end", "filed")):
                        continue
                    span = (pd.Timestamp(entry["end"]) - pd.Timestamp(entry["start"])).days
                    if not config.QUARTER_MIN_DAYS < span <= config.FLOW_MAX_SPAN_DAYS:
                        continue
                    rows.append((item, tag, rank, entry["start"], entry["end"],
                                 entry["filed"], float(entry["val"])))
    return pd.DataFrame(rows, columns=_FACT_COLUMNS)


def company_facts(cik: str) -> pd.DataFrame:
    """Quarterly USD flow facts for one filer, with the date each was filed."""
    resp = requests.get(_FACTS.format(cik=cik), headers=_headers(), timeout=90)
    if resp.status_code != 200:
        return pd.DataFrame(columns=_FACT_COLUMNS)
    return parse_company_facts(resp.json().get("facts", {}).get("us-gaap", {}))


def fetch_facts(ciks: dict[str, str], refresh: bool = False,
                progress: bool = True) -> pd.DataFrame:
    """Company facts for many tickers, cached as one parquet."""
    path = _cached("edgar_facts.parquet")
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frames = []
    for i, (ticker, cik) in enumerate(sorted(ciks.items())):
        facts = company_facts(cik)
        if len(facts):
            facts.insert(0, "ticker", ticker)
            frames.append(facts)
        if progress and i % 25 == 0:
            print(f"  edgar facts {i}/{len(ciks)}", flush=True)
        time.sleep(config.SEC_RATE_LIMIT_SLEEP)
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=["ticker"] + _FACT_COLUMNS))
    out.to_parquet(path)
    return out
