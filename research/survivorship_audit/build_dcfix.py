"""Shadow cache with the 'data center(s)' vocabulary hits filtered to genuine market mentions.

A "data center"/"data centers" full-text hit is KEPT only if that filing's Item 1 (Business) mentions
data centers other than as the company's own IT ("our data centers", "third-party data centers",
"network and data center resiliency", ...). All other phrases are untouched. Item 1 text comes from
research/xsec_alpha_probe/_cache/tenk_item1 (read-only); filings missing there are fetched with the
same extractor into _cache/item1_extra/. A hit whose Item 1 still can't be found is KEPT and counted.
Writes research/survivorship_audit/_shadow_dcfix/ (a copy of _shadow_base with cleaned hit files).
"""
import json, pathlib, re, shutil, sys
from collections import defaultdict
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from research.xsec_alpha_probe import tenk
D = pathlib.Path(__file__).parent
SRC = ROOT/"research"/"xsec_alpha_probe"/"_cache"/"tenk_item1"; EXTRA = D/"_cache"/"item1_extra"; EXTRA.mkdir(parents=True, exist_ok=True)
DC = ('"data center"', '"data centers"')
ANY = re.compile(r"data cent(?:er|re)s?", re.I)
SELF = re.compile(r"(\b(our|its|the company's|third[- ]party|co-?located|cloud|own)\s+(?:\w+\s+)?data cent(?:er|re)s?"
                  r"|networks?\s+and\s+data cent(?:er|re)s?|data cent(?:er|re)s?\s+(?:resiliency|redundancy|and networks?))", re.I)
files = defaultdict(list)
for d in (SRC, EXTRA):
    for f in d.glob("*.txt"):
        t, dt = f.stem.rsplit("_", 1); files[t].append((pd.Timestamp(dt), f))
t2c = json.loads((ROOT/"capex_intensity_strategy/data/cache/ticker_cik.json").read_text())

def find(t, fd):
    c = [(abs((d - fd).days), f) for d, f in files.get(t, []) if abs((d - fd).days) <= 10]
    return min(c)[1] if c else None

def fetch_missing(t, fd):
    cik = t2c.get(t)
    if not cik: return None
    try:
        fl = tenk.filings(cik)
    except Exception:
        return None
    fl = fl[(pd.to_datetime(fl.filingDate) - fd).abs().dt.days <= 10]
    for _, r in fl.iterrows():
        acc = r.accessionNumber.replace("-", "")
        html = tenk._get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{r.primaryDocument}").content
        p = EXTRA/f"{t}_{r.filingDate}.txt"; p.write_text(tenk.item1(html))
        files[t].append((pd.Timestamp(r.filingDate), p)); return p
    return None

S = D/"_shadow_dcfix"
if S.exists(): shutil.rmtree(S)
shutil.copytree(D/"_shadow_base", S, symlinks=True)
audit = []
for fn in ("vocabulary_hits.csv", "vocabulary_hits_short_universe.csv"):
    h = pd.read_csv(S/"es_data"/fn, comment="#")
    keep = []
    for _, r in h.iterrows():
        if r.phrase not in DC: keep.append(True); continue
        fd = pd.Timestamp(r.file_date); f = find(r.ticker, fd) or fetch_missing(r.ticker, fd)
        if f is None:
            keep.append(True); audit.append((fn, r.ticker, r.file_date, r.phrase, "no_item1_kept", 0, 0)); continue
        txt = f.read_text(errors="ignore"); a, s = len(ANY.findall(txt)), len(SELF.findall(txt))
        ok = a - s > 0; keep.append(ok)
        audit.append((fn, r.ticker, r.file_date, r.phrase, "kept" if ok else "dropped", a, s))
    out = h[keep]
    with open(S/"es_data"/fn, "w") as fh:
        fh.write("# SHADOW (dcfix): data center(s) hits kept only with an Item-1 market mention\n")
        out.to_csv(fh, index=False)
a = pd.DataFrame(audit, columns=["file", "ticker", "file_date", "phrase", "result", "item1_mentions", "self_ref"])
a.to_csv(D/"dcfix_hit_audit.csv", index=False)
print(a.result.value_counts().to_dict())
