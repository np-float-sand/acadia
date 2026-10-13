"""Survivorship shadows on the cleaned screen (data-center + gas T&D fixes, commit 4bb636f).

    python research/survivorship_audit/build_clean_shadow.py

Production now applies both fixes when it loads vocabulary hits, so a shadow copied from today's
caches carries them -- except for the dropouts' own "data center(s)" hits, which the production
exclusion list never saw. This builds
  _shadow_clean_base/  today's production caches, no dropouts (control; should reproduce 2.28)
  _shadow_clean/       the same + the 57 priced dropouts, with each dropout "data center(s)" hit
                       checked against that filing's Item 1 exactly as `build_vocab_filter` does
and writes clean_dc_dropout_audit.csv. Gas T&D dropouts (SIC 4922-4924) fall out of the short pool
on their own via config.RULE2_EXCLUDED_SICS. Then run run_compare.py clean_base / clean.
"""
import json, pathlib, subprocess, sys
import pandas as pd
D = pathlib.Path(__file__).resolve().parent; ROOT = D.parents[1]; sys.path.insert(0, str(ROOT))
from electrification_strategy import auto_screen, config as es_config
from research.xsec_alpha_probe import tenk

# tenk.filings() only lists 10-Ks filed from 2017-06; the dropouts' 2012-17 hits would otherwise be kept
# unchecked (e.g. Barnes's 2017 hit alone admits it for 2019). Production has the same limit for live names.
tenk.FIRST_FILED = "2012-01-01"

for args in (["--base", "--dir", "_shadow_clean_base"], ["--dir", "_shadow_clean"]):
    subprocess.run([sys.executable, str(D/"build_shadow.py"), *args], check=True)

S = D/"_shadow_clean"; ESD = S/"es_data"
drop = pd.read_csv(D/"dropout_universe_shadow.csv", dtype={"cik": str})
cik = dict(zip(drop.col, drop.cik.str.zfill(10)))
EXTRA = D/"_cache"/"item1_extra"; EXTRA.mkdir(parents=True, exist_ok=True)
cache = {}
for d in (tenk.TENK, EXTRA):
    for f in d.glob("*.txt"):
        t, dt = f.stem.rsplit("_", 1); cache.setdefault(t, []).append((pd.Timestamp(dt), f))

def item1(t, fd):
    near = [(abs((d - fd).days), f) for d, f in cache.get(t, []) if abs((d - fd).days) <= 10]
    if near:
        return min(near)[1]
    try:
        fl = tenk.filings(cik[t])
    except Exception:
        return None
    fl = fl[(pd.to_datetime(fl.filingDate) - fd).abs().dt.days <= 10]
    for _, r in fl.iterrows():
        acc = r.accessionNumber.replace("-", "")
        html = tenk._get(f"https://www.sec.gov/Archives/edgar/data/{int(cik[t])}/{acc}/{r.primaryDocument}").content
        p = EXTRA/f"{t}_{r.filingDate}.txt"; p.write_text(tenk.item1(html))
        cache.setdefault(t, []).append((pd.Timestamp(r.filingDate), p)); return p
    return None

hits = pd.read_csv(ESD/"vocabulary_hits.csv", comment="#")
dc = hits[hits.ticker.isin(cik) & hits.phrase.isin(es_config.AUTO_DC_TERMS)]
rows = []
for _, r in dc.iterrows():
    p = item1(r.ticker, pd.Timestamp(r.file_date))
    keep = True if p is None else auto_screen.is_market_mention(p.read_text(errors="ignore"))
    rows.append({"ticker": r.ticker, "phrase": r.phrase, "file_date": r.file_date,
                 "result": "no_item1_kept" if p is None else ("kept" if keep else "dropped")})
audit = pd.DataFrame(rows, columns=["ticker", "phrase", "file_date", "result"])
audit.to_csv(D/"clean_dc_dropout_audit.csv", index=False)
exf = ESD/auto_screen.DC_EXCLUDED_FILE
ex = pd.read_csv(exf, comment="#")
add = audit.loc[audit.result == "dropped", ["ticker", "phrase", "file_date"]]
with open(exf, "w") as fh:
    fh.write("# SHADOW (clean): production data-center exclusions + survivorship dropouts' own spurious hits\n")
    pd.concat([ex, add], ignore_index=True).to_csv(fh, index=False)
print("dropout data-center hits:", audit.result.value_counts().to_dict(), "| dropped tickers:", sorted(set(add.ticker)))
