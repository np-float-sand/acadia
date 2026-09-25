"""10-K business descriptions from EDGAR, for text-similar peer links (plan step 3a).

    python -m research.xsec_alpha_probe.tenk          # fetch everything missing

Hoberg-Phillips TNIC is keyed by Compustat gvkey (paid mapping) and stops at 2023,
so we rebuild the same kind of measure from free EDGAR text. For each firm and
fiscal year we keep only Item 1 (Business) as plain text, cached per filing under
`_cache/tenk/`. Point-in-time key = the filing date, not the fiscal year.
"""
from __future__ import annotations

import html as _html
import io
import json
import re
import time

import pandas as pd
import requests

from research.xsec_alpha_probe import data

UA = {"User-Agent": "acadia-research sand.gh1902@gmail.com"}
TENK = data.CACHE / "tenk_item1"
TENK.mkdir(exist_ok=True)
FIRST_FILED, LAST_FILED = "2017-06-01", "2026-09-30"
MAX_CHARS = 200_000
MIN_ITEM1 = 2_000
def _spaced(word: str) -> str:
    """Tolerate styling that splits a heading word ("B USINESS", "RIS K")."""
    return r"\s*".join(word)


_SEP = r"\s*[\.\:\-–—]?\s*"
_ITEM1 = re.compile(r"item\s*1" + _SEP + _spaced("business"), re.I)
_END = re.compile(r"item\s*1\s*a" + _SEP + _spaced("risk") + r"\s*" + _spaced("factors")
                  + r"|item\s*2" + _SEP + _spaced("propert"), re.I)
_XREF = re.compile(r"^[\s,\.”\"]*(of|in|and|to|within|under)\b", re.I)   # "Item 1A. Risk Factors of this report"


def _get(url: str) -> requests.Response:
    for attempt in range(6):
        try:
            r = requests.get(url, headers=UA, timeout=60)
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
            time.sleep(5 * (attempt + 1))       # SEC drops connections under load
            continue
        if r.status_code in (429, 503):
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(0.9)             # 8 shards in parallel stay under SEC's 10 requests/second
        return r
    raise RuntimeError(f"gave up after 6 attempts: {url}")


def sp500() -> pd.DataFrame:
    path = data.CACHE / "sp500.csv"
    if not path.exists():
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        t = pd.read_html(io.StringIO(r.text))[0][["Symbol", "CIK", "GICS Sector", "GICS Sub-Industry"]]
        t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
        t.to_csv(path, index=False)
    return pd.read_csv(path)


def firm_ciks() -> dict[str, str]:
    """Universe members (ever eligible) + S&P 500, ticker -> 10-digit CIK."""
    tc = json.loads((data._cis() / "ticker_cik.json").read_text())
    univ = data.universe()
    out = {t: tc[t] for t in univ.columns[univ.any()] if t in tc}
    for _, row in sp500().iterrows():
        out.setdefault(row["Symbol"], str(int(row["CIK"])).zfill(10))
    return out


def filings(cik: str) -> pd.DataFrame:
    """Every 10-K (not amendments) filed in the window: date, accession, document."""
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for extra in sub["filings"].get("files", []):
        frames.append(pd.DataFrame(_get(f"https://data.sec.gov/submissions/{extra['name']}").json()))
    f = pd.concat(frames)
    f = f[(f["form"] == "10-K") & (f["filingDate"] >= FIRST_FILED) & (f["filingDate"] <= LAST_FILED)]
    return f[["filingDate", "accessionNumber", "primaryDocument", "reportDate"]]


def item1(html: bytes) -> str:
    raw = html.decode("utf-8", errors="ignore")
    raw = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<ix:header>.*?</ix:header>", " ", raw)
    text = _html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"[\s\u00a0\u200b]+", " ", text)
    ends = [e for e in _END.finditer(text) if not _XREF.match(text[e.end(): e.end() + 20])]
    # The real Item 1 is the FIRST heading (after the table of contents) whose section
    # runs >= MIN_ITEM1 chars to an Item 1A/Item 2 heading. Taking the longest instead
    # lets a later cross-reference ("see Item 1. Business -- Backlog") inside Risk
    # Factors win, and the extract becomes Risk Factors.
    first, longest, open_ended = "", "", ""
    for m in _ITEM1.finditer(text):
        if _XREF.match(text[m.end(): m.end() + 20]):
            continue
        end = next((e for e in ends if e.start() > m.end()), None)
        if end is not None:
            seg = text[m.end(): end.start()]
            if len(seg) >= MIN_ITEM1 and not first:
                first = seg
            longest = seg if len(seg) > len(longest) else longest
        elif not open_ended:
            open_ended = text[m.end(): m.end() + MAX_CHARS]
    best = first or longest or open_ended
    if len(best) < MIN_ITEM1:           # non-standard layout (GE, combined utility filings): whole 10-K
        best = text
    return best[:MAX_CHARS]


def fetch_firm(ticker: str, cik: str) -> int:
    done = 0
    for _, f in filings(cik).iterrows():
        path = TENK / f"{ticker}_{f['filingDate']}.txt"
        if path.exists() and path.stat().st_size >= MIN_ITEM1:
            continue
        acc = f["accessionNumber"].replace("-", "")
        html = _get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{f['primaryDocument']}").content
        path.write_text(item1(html))
        done += 1
    return done


_COVER = re.compile(r"securities\s+and\s+exchange\s+commission", re.I)


def index() -> pd.DataFrame:
    """ticker, filed date, chars, and whether the file is the whole-10-K fallback
    (starts at the cover page) rather than Item 1, for every cached filing."""
    rows = []
    for p in TENK.glob("*.txt"):
        t, d = p.stem.rsplit("_", 1)
        with p.open() as fh:
            head = fh.read(1500)
        rows.append({"ticker": t, "filed": pd.Timestamp(d), "chars": p.stat().st_size, "path": p,
                     "full_text": bool(_COVER.search(head))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    firms = firm_ciks()
    args = sys.argv[1:]
    shard = next((a for a in args if a.startswith("--shard=")), None)
    only = [a for a in args if not a.startswith("--")]
    todo = {t: c for t, c in firms.items() if not only or t in only}
    if shard:                      # --shard=i/n: every n-th firm, for parallel runs
        i, n = map(int, shard.split("=")[1].split("/"))
        todo = dict(sorted(todo.items())[i::n])
    print(f"{len(todo)} firms")
    for i, (t, c) in enumerate(sorted(todo.items())):
        try:
            n = fetch_firm(t, c)
        except Exception as e:          # keep going; a firm with no 10-Ks (foreign filer) is fine
            print(f"  {t}: {type(e).__name__} {e}")
            continue
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} {t} +{n}", flush=True)
