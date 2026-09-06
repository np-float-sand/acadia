"""Phase 1 (throwaway): pull SCC case detail + document list for each Dominion VA
transmission case. Writes meta/<case>.json with {detail, docs, picks}."""
from __future__ import annotations
import json, sys, time, urllib.parse, pathlib
import requests

BREEZE = "https://www.scc.virginia.gov/DocketSearchAPI/breeze/CaseDetails"
OUT = pathlib.Path("meta"); OUT.mkdir(exist_ok=True)
S = requests.Session()
S.headers.update({"Accept": "application/json", "User-Agent": "research-probe/0.1"})


def q(path: str, flt: str, select: str | None = None) -> list:
    params = {"$filter": flt}
    if select:
        params["$select"] = select
    url = f"{BREEZE}/{path}?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    for attempt in range(4):
        try:
            r = S.get(url, timeout=45, allow_redirects=True)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} {url}")
        except Exception as e:  # noqa
            print(f"  err {e}")
        time.sleep(2 * (attempt + 1))
    return []


def pick_docs(docs: list) -> dict:
    """Choose the smallest set of documents that carry cost / need / in-service."""
    picks = {"final_order": None, "he_report": None, "order_notice_hearing": None,
             "staff_report": None, "application_v1p1": None}
    for d in docs:
        name = (d.get("Document_Name") or "").lower()
        fn = d.get("FileName")
        if not fn:
            continue
        if picks["final_order"] is None and "final order" in name:
            picks["final_order"] = d
        if picks["he_report"] is None and "hearing examiner" in name and "report of" in name:
            picks["he_report"] = d
        if picks["order_notice_hearing"] is None and ("order for notice and hearing" in name
                                                      or "order for notice" in name):
            picks["order_notice_hearing"] = d
        if picks["staff_report"] is None and "staff report" in name:
            picks["staff_report"] = d
        if picks["application_v1p1"] is None and "application" in name and (
                "vol. 1" in name or "volume 1" in name or "pt. 1 of" in name
                or "part 1 of" in name or name.strip().startswith(
                    "virginia electric and power company - application")):
            picks["application_v1p1"] = d
    return picks


def main():
    lines = [l.strip() for l in open("cases.txt") if l.strip() and not l.startswith("#")]
    summary = []
    for i, line in enumerate(lines, 1):
        case, region = line.split("|")
        cache = OUT / f"{case}.json"
        if cache.exists():
            rec = json.load(open(cache))
        else:
            det = q("GetDetail", f"Case_Number eq '{case}'")
            detail = det[0] if det else {}
            mno = detail.get("MATTER_NO")
            docs = []
            if mno:
                docs = q("GetDocuments", f"MATTER_NO eq {mno}",
                         "DocID,FileName,Document_Name,Date_Filed")
            rec = {"case": case, "region": region, "detail": detail,
                   "n_docs": len(docs), "docs": docs, "picks": pick_docs(docs)}
            json.dump(rec, open(cache, "w"), indent=1, default=str)
            time.sleep(0.4)
        p = rec["picks"]
        got = [k for k, v in p.items() if v]
        summary.append((case, region, rec["detail"].get("Status"),
                        rec["detail"].get("Disposition"),
                        rec["detail"].get("Case_Established_Date"),
                        rec["detail"].get("Final_Order_Date"),
                        rec["n_docs"], ",".join(got)))
        print(f"[{i:2}/{len(lines)}] {case} {region:8} "
              f"{str(rec['detail'].get('Disposition')):10} "
              f"est={rec['detail'].get('Case_Established_Date')} "
              f"fo={rec['detail'].get('Final_Order_Date')} docs={rec['n_docs']:3} -> {','.join(got)}")
    json.dump(summary, open("meta_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
