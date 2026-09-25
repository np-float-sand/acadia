"""Fetch generator earnings-release 8-K exhibits (EX-99.*) from EDGAR and cache the text.

Uses EDGAR full-text search for 'hedged' within each CIK's 8-K filings, then downloads every
EX-99 hit to _cache/{ticker}/{file_date}_{filename}.txt (plain text, tags stripped).
"""
from __future__ import annotations

import html
import json
import pathlib
import re
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
CACHE = HERE / "_cache"
UA = {"User-Agent": "acadia research sand.gh1902@gmail.com"}
CIKS = {"VST": "0001692819", "CEG": "0001868275", "NRG": "0001013871", "TLN": "0001622536"}
START, END = "2022-01-01", "2026-09-24"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def search(cik: str) -> list[dict]:
    hits, frm = [], 0
    while True:
        url = (f"https://efts.sec.gov/LATEST/search-index?q=%22hedged%22&ciks={cik}&forms=8-K"
               f"&dateRange=custom&startdt={START}&enddt={END}&from={frm}")
        d = json.loads(_get(url))
        page = d["hits"]["hits"]
        hits += page
        frm += len(page)
        if not page or frm >= d["hits"]["total"]["value"]:
            return hits
        time.sleep(0.2)


def to_text(raw: bytes) -> str:
    t = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", "ignore"))
    return re.sub(r"\s+", " ", html.unescape(t))


def main() -> None:
    for tkr, cik in CIKS.items():
        out = CACHE / tkr
        out.mkdir(parents=True, exist_ok=True)
        for h in search(cik):
            s = h["_source"]
            if not str(s.get("file_type", "")).startswith("EX-99"):
                continue
            adsh, fname = h["_id"].split(":")
            dest = out / f"{s['file_date']}_{fname}.txt"
            if dest.exists():
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh.replace('-', '')}/{fname}"
            try:
                dest.write_text(to_text(_get(url)))
            except Exception as e:  # noqa: BLE001
                print("fail", tkr, url, e)
            time.sleep(0.15)
        print(tkr, len(list(out.glob("*.txt"))))


if __name__ == "__main__":
    main()
