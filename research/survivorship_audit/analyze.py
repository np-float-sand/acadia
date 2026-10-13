import json, pathlib, time, requests, sys, pandas as pd
sys.path.insert(0, "/Users/sandhyapersad/acadia")
from capex_intensity_strategy import config
D = pathlib.Path(__file__).parent; SUB = D/"subs"; FL = D/"float"; FL.mkdir(exist_ok=True)
H = {"User-Agent": config.SEC_USER_AGENT}
pop = json.loads((D/"sic_pop.json").read_text())
sic_of = {c: s for s, cs in pop.items() for c in cs}
rows = []
for f in SUB.glob("*.json"):
    d = json.loads(f.read_text()); r = d["filings"]["recent"]
    forms, dates, items = r["form"], r["filingDate"], r.get("items", [""]*len(r["form"]))
    tenk = [dt for fm, dt in zip(forms, dates) if fm.startswith("10-K")]
    if not tenk or max(tenk) < "2016-01-01": continue
    if any(fm.startswith("10-K") for fm, dt in zip(forms, dates) if dt >= "2026-01-01"): continue  # still filing
    ev = [(dt, it) for fm, dt, it in zip(forms, dates, items) if fm.startswith("8-K") and it]
    bk = [dt for dt, it in ev if "1.03" in it]
    acq = [dt for dt, it in ev if "2.01" in it and "5.01" in it]  # completed deal + change in control
    dl = [dt for fm, dt in zip(forms, dates) if fm in ("25-NSE", "25")]
    dereg = [dt for fm, dt in zip(forms, dates) if fm.startswith("15-")]
    outcome = "BANKRUPTCY" if bk else "ACQUIRED" if acq else "DELISTED" if dl else "DEREGISTERED" if dereg else "STOPPED FILING"
    rows.append(dict(cik=f.stem, name=d["name"], sic=sic_of.get(f.stem), sicDesc=d.get("sicDescription"),
        last_10k=max(tenk), outcome=outcome, event_date=(bk or acq or dl or dereg or [max(dates)])[0],
        exch=",".join(x for x in (d.get("exchanges") or []) if x)))
df = pd.DataFrame(rows)
# public float (dei) for size ranking
def flt(c):
    f = FL/f"{c}.json"
    if not f.exists():
        try:
            r = requests.get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{c}/dei/EntityPublicFloat.json", headers=H, timeout=60)
            f.write_text(r.text if r.status_code == 200 else "{}")
        except requests.exceptions.RequestException:
            return None
        time.sleep(0.12)
    try: u = json.loads(f.read_text())["units"]["USD"]
    except Exception: return None
    u = [x for x in u if x.get("fy", 0) >= 2015]
    return max(x["val"] for x in u) if u else None
df["max_float_musd"] = [round((flt(c) or 0)/1e6) for c in df.cik]
df = df.sort_values("max_float_musd", ascending=False)
df.to_csv(D/"survivorship_candidates.csv", index=False)
print(len(df), df.outcome.value_counts().to_dict())
