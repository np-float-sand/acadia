import json, re, time, sys, pathlib, requests, pandas as pd
sys.path.insert(0, "/Users/sandhyapersad/acadia")
from capex_intensity_strategy import edgar, universe, config
OUT = pathlib.Path(__file__).parent; SUB = OUT/"subs"; SUB.mkdir(exist_ok=True)
H = {"User-Agent": config.SEC_USER_AGENT}
sics = universe.load_sic_codes()
pop = {}
p = OUT/"sic_pop.json"
if p.exists(): pop = json.loads(p.read_text())
for s in sics.sic:
    if str(s) in pop: continue
    pop[str(s)] = sorted(edgar.sic_population(int(s))); print(s, len(pop[str(s)]), flush=True)
    p.write_text(json.dumps(pop))
t2c = edgar.ticker_to_cik(refresh=True); listed = set(t2c.values())
allc = {c: s for s, cs in pop.items() for c in cs}
gone = [c for c in allc if c not in listed]
print("filers", len(allc), "no ticker", len(gone), flush=True)
rows = []
for i, c in enumerate(gone):
    f = SUB/f"{c}.json"
    if not f.exists():
        r = None
        for k in range(5):
            try:
                r = requests.get(f"https://data.sec.gov/submissions/CIK{c}.json", headers=H, timeout=60); break
            except requests.exceptions.RequestException:
                time.sleep(2 * (k + 1))
        if r is None or r.status_code != 200: continue
        f.write_text(r.text); time.sleep(0.12)
    d = json.loads(f.read_text()); rec = d["filings"]["recent"]
    forms, dates = rec["form"], rec["filingDate"]
    tenk = [dt for fm, dt in zip(forms, dates) if fm in ("10-K", "10-K405", "10-KSB")]
    dl = [f"{fm}:{dt}" for fm, dt in zip(forms, dates) if fm in ("25-NSE", "25", "15-12B", "15-12G", "15-15D")]
    rows.append(dict(cik=c, name=d.get("name"), sic=allc[c], sicDesc=d.get("sicDescription"),
                     exchanges=",".join(x for x in (d.get("exchanges") or []) if x), tickers=",".join(x for x in (d.get("tickers") or []) if x), last_filing=max(dates) if dates else None,
                     last_10k=max(tenk) if tenk else None, delist_forms=";".join(dl[:3]),
                     former=";".join(x["name"] for x in d.get("formerNames", [])[:2])))
    if i % 200 == 0: print(i, flush=True)
pd.DataFrame(rows).to_csv(OUT/"no_ticker_filers.csv", index=False)
print("done")
