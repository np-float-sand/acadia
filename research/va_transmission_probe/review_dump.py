"""Phase 2b (throwaway): re-parse cached PDFs with sentence-level regexes and
dump a compact per-case review block for human verification of cost/driver/kV/ISD."""
from __future__ import annotations
import json, re, pathlib, textwrap
import pdfplumber

META = pathlib.Path("meta"); PDFDIR = pathlib.Path("pdfs")
PRIORITY = ["he_report", "final_order", "staff_report", "order_notice_hearing", "application_v1p1"]

WS = re.compile(r"\s+")
SENT = re.compile(r"[^.]*?\$[^.]*\.")            # any sentence containing a $
COSTSENT = re.compile(
    r"(?:[A-Z][^.]{0,240}?)?\b(?:cost|costs|investment|estimated?)\b[^.]{0,200}?"
    r"\$\s?[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:million|billion)?[^.]{0,160}\.", re.I)
MONEY = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|billion)?", re.I)
ISD = re.compile(
    r"in[- ]service[^.]{0,120}?"
    r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+20\d{2})", re.I)
NEEDWIN = re.compile(
    r"(?:needed to|need for the Project|Project is needed|purpose of the Project|"
    r"driven by|in order to|address(?:es)? (?:the|a|an)|to maintain|to serve|to provide"
    r"|end[- ]of[- ]life|thermal overload|reliability (?:criteria|need)|data center"
    r"|technology park|economic development|load growth|planning criteria)[^.]{0,320}\.", re.I)


def norm(t: str) -> str:
    return WS.sub(" ", t).strip()


def text_of(fn: str, pages=45) -> str:
    p = PDFDIR / fn.replace("/", "_")
    if not p.exists():
        return ""
    try:
        with pdfplumber.open(p) as pdf:
            return "\n".join((pg.extract_text() or "") for pg in pdf.pages[:pages])
    except Exception:
        return ""


def dollars(s: str) -> float | None:
    best = None
    for m in MONEY.finditer(s):
        n = float(m.group(1).replace(",", "")); u = (m.group(2) or "").lower()
        v = n * (1e9 if u == "billion" else 1e6 if u == "million" else (1e6 if n < 1000 else 1))
        if 1e5 <= v <= 5e10:
            best = v if best is None else max(best, v)
    return best


def main():
    lines = [l.strip() for l in open("cases.txt") if l.strip() and not l.startswith("#")]
    out = []
    for line in lines:
        case, region = line.split("|")
        rec = json.load(open(META / f"{case}.json"))
        det, picks = rec["detail"], rec["picks"]
        cap = norm(det.get("Caption", "") or "")
        blob = ""
        used = []
        for k in PRIORITY:
            d = picks.get(k)
            if not d:
                continue
            t = text_of(d["FileName"])
            if len(t) > 300:
                blob += "\n" + t
                used.append(k)
            if len([u for u in used if u in ("he_report", "final_order", "staff_report")]) >= 2:
                break
        nb = norm(blob)
        costs = []
        for m in COSTSENT.finditer(nb):
            s = m.group(0).strip()
            if "the Project" in s or "Project along" in s or "estimated" in s.lower():
                v = dollars(s)
                if v:
                    costs.append((v, s[:300]))
        # de-dup by sentence
        seen = set(); costs2 = []
        for v, s in costs:
            if s not in seen:
                seen.add(s); costs2.append((v, s))
        isd = ISD.search(nb)
        needs = []
        for m in NEEDWIN.finditer(nb):
            s = norm(m.group(0))
            if 30 < len(s) < 340 and s not in needs:
                needs.append(s)
        out.append({
            "case": case, "region": region,
            "disposition": det.get("Disposition"), "established": det.get("Case_Established_Date"),
            "final_order_date": det.get("Final_Order_Date"), "status": det.get("Status"),
            "caption": cap, "docs_used": used,
            "isd": isd.group(1) if isd else None,
            "cost_sentences": costs2[:6],
            "need_sentences": needs[:10],
        })

    json.dump(out, open("review.json", "w"), indent=1, default=str)
    with open("review.txt", "w") as f:
        for r in out:
            f.write("=" * 100 + "\n")
            f.write(f"{r['case']}  [{r['region']}]  {r['disposition']}  est={r['established']} "
                    f"FO={r['final_order_date']}  docs={r['docs_used']}\n")
            f.write(f"CAPTION: {r['caption']}\n")
            f.write(f"ISD(regex): {r['isd']}\n")
            f.write("COST sentences:\n")
            for v, s in r["cost_sentences"]:
                f.write(f"   [${v/1e6:,.1f}M]  {s}\n")
            if not r["cost_sentences"]:
                f.write("   (none matched)\n")
            f.write("NEED sentences:\n")
            for s in r["need_sentences"]:
                f.write(textwrap.fill(s, 130, initial_indent="   - ", subsequent_indent="     ") + "\n")
            f.write("\n")
    print("wrote review.txt / review.json for", len(out), "cases")


if __name__ == "__main__":
    main()
