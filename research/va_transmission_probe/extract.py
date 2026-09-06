"""Phase 2 (throwaway): download key PDFs per case and extract
cost / kV / in-service / need-text. Emits extracted.csv for human review."""
from __future__ import annotations
import json, re, csv, time, pathlib, urllib.parse
import requests
import pdfplumber

META = pathlib.Path("meta")
PDFDIR = pathlib.Path("pdfs"); PDFDIR.mkdir(exist_ok=True)
DOCS_BASE = "http://www.scc.virginia.gov/docketsearch/DOCS/"
S = requests.Session()
S.headers.update({"User-Agent": "research-probe/0.1"})

# doc kinds to pull, in priority order; stop once we have 2 with text
PRIORITY = ["he_report", "final_order", "staff_report", "order_notice_hearing", "application_v1p1"]
MAXBYTES = 60 * 1024 * 1024


def dl(filename: str) -> pathlib.Path | None:
    safe = filename.replace("/", "_")
    out = PDFDIR / safe
    if out.exists() and out.stat().st_size > 0:
        return out
    url = DOCS_BASE + urllib.parse.quote(filename)
    try:
        with S.get(url, timeout=90, stream=True, allow_redirects=True) as r:
            if r.status_code != 200:
                print(f"    HTTP {r.status_code} {filename}")
                return None
            total = 0
            with open(out, "wb") as f:
                for chunk in r.iter_content(65536):
                    total += len(chunk)
                    if total > MAXBYTES:
                        print(f"    too big (>{MAXBYTES//1024//1024}MB) {filename}")
                        f.close(); out.unlink(missing_ok=True); return None
                    f.write(chunk)
        return out
    except Exception as e:  # noqa
        print(f"    err {e} {filename}")
        return None


def text_of(path: pathlib.Path, max_pages: int = 40) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages[:max_pages])
    except Exception as e:  # noqa
        print(f"    pdf err {e} {path.name}")
        return ""


MONEY = re.compile(
    r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|billion)?", re.I)
KV = re.compile(r"\b(\d{2,3})\s*[- ]?kV\b", re.I)
INSVC = re.compile(
    r"in[- ]service[^.]{0,60}?(?:date[^.]{0,30}?)?(?:of|by|is|:)?\s*"
    r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+20\d{2}|\b20\d{2}\b)", re.I)

NEED_KW = ["needed to", "need for the project", "project is needed", "purpose of the project",
           "driven by", "in order to", "would address", "to address", "to maintain reliable",
           "data center", "technology park", "economic development", "load growth", "new load",
           "thermal overload", "overloaded", "end of life", "end-of-life", "aging infrastructure",
           "deteriorat", "nerc", "reliability criteria", "planning criteria", "undergroun",
           "interconnect", "generation", "solar", "customer request", "delivery point"]


def money_usd(text: str) -> tuple[float | None, list[str]]:
    """Largest dollar figure appearing on a line that talks about project cost."""
    cands = []
    for line in text.splitlines():
        low = line.lower()
        if not any(k in low for k in ("cost", "estimate", "invest", "$")):
            continue
        if not any(k in low for k in ("cost", "estimate", "invest", "project", "total")):
            continue
        for m in MONEY.finditer(line):
            num = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").lower()
            if unit == "billion":
                val = num * 1e9
            elif unit == "million":
                val = num * 1e6
            elif num < 1000:            # "$32.3" with 'million' on next token missed -> assume M
                val = num * 1e6
            else:
                val = num
            if 1e5 <= val <= 5e10:
                cands.append((val, line.strip()[:200]))
    if not cands:
        return None, []
    cands.sort(reverse=True)
    return cands[0][0], [c[1] for c in cands[:4]]


def kv_of(text: str, caption: str) -> int | None:
    vals = [int(x) for x in KV.findall(caption)] + [int(x) for x in KV.findall(text[:8000])]
    vals = [v for v in vals if v in (69, 115, 138, 161, 230, 500, 765)]
    return max(vals) if vals else None


def insvc_of(text: str) -> str | None:
    m = INSVC.search(text)
    return m.group(1) if m else None


def need_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in NEED_KW) and len(line.strip()) > 25:
            out.append(re.sub(r"\s+", " ", line.strip())[:240])
    # de-dup, keep order, cap
    seen, res = set(), []
    for l in out:
        if l not in seen:
            seen.add(l); res.append(l)
    return res[:12]


def main():
    rows = []
    lines = [l.strip() for l in open("cases.txt") if l.strip() and not l.startswith("#")]
    for i, line in enumerate(lines, 1):
        case = line.split("|")[0]
        rec = json.load(open(META / f"{case}.json"))
        detail, picks = rec["detail"], rec["picks"]
        caption = detail.get("Caption", "") or ""
        pulled, blob = [], ""
        for kind in PRIORITY:
            if len(pulled) >= 2 and kind == "application_v1p1":
                break
            d = picks.get(kind)
            if not d:
                continue
            p = dl(d["FileName"])
            if not p:
                continue
            t = text_of(p)
            if len(t) > 400:
                blob += f"\n\n##### {kind} #####\n" + t
                pulled.append(kind)
            if len(pulled) >= 2:
                break
        cost, cost_ev = money_usd(blob)
        kv = kv_of(blob, caption)
        insvc = insvc_of(blob)
        needs = need_lines(blob)
        row = {
            "case": case, "region": rec["region"],
            "status": detail.get("Status"), "disposition": detail.get("Disposition"),
            "established": detail.get("Case_Established_Date"),
            "final_order_date": detail.get("Final_Order_Date"),
            "caption": re.sub(r"\s+", " ", caption).strip()[:300],
            "kv": kv, "cost_usd": int(cost) if cost else None,
            "target_in_service": insvc,
            "docs_used": "|".join(pulled),
            "cost_evidence": " || ".join(cost_ev),
            "need_text": " || ".join(needs),
        }
        rows.append(row)
        print(f"[{i:2}/{len(lines)}] {case} kv={kv} cost={row['cost_usd']} "
              f"insvc={insvc} docs={pulled}")
        time.sleep(0.2)
    cols = ["case", "region", "status", "disposition", "established", "final_order_date",
            "kv", "cost_usd", "target_in_service", "docs_used", "caption",
            "cost_evidence", "need_text"]
    with open("extracted.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote extracted.csv")


if __name__ == "__main__":
    main()
