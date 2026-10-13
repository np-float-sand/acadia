"""Map dropout filers (CIK) to their old exchange ticker via the last 10-K's dei:TradingSymbol,
validated against Tiingo's supported-tickers list (ticker whose endDate is near the exit)."""
import json, re, sys, time, pathlib, requests, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from capex_intensity_strategy import config
D = pathlib.Path(__file__).parent; C = D/"_cache"; DOC = C/"tenk"; DOC.mkdir(parents=True, exist_ok=True)
SUBS = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else D/"_cache"/"subs"
H = {"User-Agent": config.SEC_USER_AGENT}
cand = pd.read_csv(D/"survivorship_candidates.csv", dtype={"cik": str})
cand["cik"] = cand.cik.str.zfill(10)
tg = pd.read_parquet(C/"tiingo_supported.parquet")
tg = tg[(tg.assetType == "Stock") & (tg.priceCurrency == "USD") & tg.endDate.notna()]
tg["endDate"] = pd.to_datetime(tg.endDate, errors="coerce"); tg["ticker"] = tg.ticker.str.upper()
SYM = re.compile(r'name="dei:TradingSymbol"[^>]*>\s*(?:<[^>]+>\s*)*([A-Za-z.\-]{1,6})', re.I)
TXT = re.compile(r'(?:NYSE|Nasdaq|NASDAQ|New York Stock Exchange)[^A-Za-z]{0,6}(?:American|Global|Stock Market|Select)?[^A-Za-z]{0,30}?(?:under the (?:ticker )?symbol|symbol)\s*["“]?([A-Z]{1,5})', re.S)
def get(url, path):
    if path.exists(): return path.read_text(errors="ignore")
    for k in range(4):
        try:
            r = requests.get(url, headers=H, timeout=60); time.sleep(0.12)
            if r.status_code == 200: path.write_text(r.text); return r.text
            return ""
        except requests.exceptions.RequestException: time.sleep(2*(k+1))
    return ""
rows = []
for _, c in cand.iterrows():
    sub = json.loads((SUBS/f"{c.cik}.json").read_text()); r = sub["filings"]["recent"]
    tenks = [(dt, acc, doc) for fm, dt, acc, doc in zip(r["form"], r["filingDate"], r["accessionNumber"], r["primaryDocument"]) if fm.startswith("10-K")]
    syms = []
    for dt, acc, doc in sorted(tenks, reverse=True)[:2]:
        txt = get(f"https://www.sec.gov/Archives/edgar/data/{int(c.cik)}/{acc.replace('-','')}/{doc}", DOC/f"{c.cik}_{acc}.htm")
        syms += [s.upper().strip(".") for s in SYM.findall(txt)] + TXT.findall(txt)
        if syms: break
    syms += [t.upper() for t in (sub.get("tickers") or []) if t]
    exit_dt = pd.Timestamp(c.event_date) if c.outcome != "STOPPED FILING" else pd.Timestamp(c.last_10k)
    best = None
    for s in dict.fromkeys(syms):
        hit = tg[tg.ticker == s]
        if len(hit):
            gap = (hit.endDate - exit_dt).abs().min().days
            if best is None or gap < best[1]: best = (s, gap)
    rows.append(dict(cik=c.cik, name=c["name"], outcome=c.outcome, symbols=",".join(dict.fromkeys(syms)),
                     ticker=best[0] if best else None, end_gap_days=best[1] if best else None))
out = pd.DataFrame(rows); out.to_csv(D/"ticker_map.csv", index=False)
ok = out[out.end_gap_days.notna() & (out.end_gap_days <= 400)]
print("candidates", len(out), "| symbol found", out.symbols.ne("").sum(), "| matched in Tiingo (end within 400d of exit)", len(ok))
