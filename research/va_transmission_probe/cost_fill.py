"""Targeted cost re-pull for cases the first pass missed. Full-text parse of
he_report + final_order + staff_report + application_v1p1, canonical phrasings only."""
from __future__ import annotations
import json, re, pathlib
import requests, pdfplumber

META = pathlib.Path("meta"); PDFDIR = pathlib.Path("pdfs")
DOCS_BASE = "http://www.scc.virginia.gov/docketsearch/DOCS/"
S = requests.Session(); S.headers.update({"User-Agent": "research-probe/0.1"})

MISSING = ["PUR-2024-00225", "PUR-2024-00021", "PUR-2023-00049", "PUR-2023-00029",
           "PUR-2022-00198", "PUR-2022-00197", "PUR-2019-00215", "PUR-2019-00049",
           "PUR-2023-00203", "PUR-2023-00088", "PUR-2019-00078", "PUR-2024-00074",
           "PUR-2026-00009", "PUR-2023-00168", "PUR-2023-00110", "PUR-2023-00054"]

WS = re.compile(r"\s+")
# canonical: "estimated (conceptual) cost of the (Rebuild )Project ... approximately $X million"
PATS = [
    re.compile(r"estimated (?:conceptual )?cost of the (?:proposed )?(?:Rebuild )?Project[^.$]{0,90}?"
               r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|billion)", re.I),
    re.compile(r"Project[^.$]{0,40}?estimated to cost[^.$]{0,40}?"
               r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|billion)", re.I),
    re.compile(r"total (?:estimated )?(?:conceptual )?cost[^.$]{0,60}?"
               r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|billion)", re.I),
    re.compile(r"PROJECT COST[^$]{0,120}?\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|billion)", re.I),
]


def dl(fn):
    out = PDFDIR / fn.replace("/", "_")
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        r = S.get(DOCS_BASE + requests.utils.quote(fn), timeout=90)
        if r.status_code == 200 and len(r.content) < 90_000_000:
            out.write_bytes(r.content); return out
    except Exception as e:
        print("  dl err", e)
    return None


def fulltext(fn):
    p = dl(fn)
    if not p:
        return ""
    try:
        with pdfplumber.open(p) as pdf:
            return WS.sub(" ", "\n".join((pg.extract_text() or "") for pg in pdf.pages))
    except Exception:
        return ""


for case in MISSING:
    rec = json.load(open(META / f"{case}.json"))
    picks = rec["picks"]
    hits = []
    for kind in ("he_report", "final_order", "staff_report", "order_notice_hearing", "application_v1p1"):
        d = picks.get(kind)
        if not d:
            continue
        t = fulltext(d["FileName"])
        if not t:
            continue
        for pat in PATS:
            for m in pat.finditer(t):
                n = float(m.group(1).replace(",", "")); u = m.group(2).lower()
                v = n * (1e9 if u == "billion" else 1e6)
                if 5e5 <= v <= 5e9:
                    s = t[max(0, m.start() - 90):m.end() + 40]
                    hits.append((kind, v, s))
        if hits:
            break
    print("=" * 80)
    print(case, rec["detail"].get("Caption", "")[:80])
    if not hits:
        print("   STILL NONE")
    for k, v, s in hits[:5]:
        print(f"   [{k}] ${v/1e6:,.1f}M  ...{s.strip()[:200]}")
