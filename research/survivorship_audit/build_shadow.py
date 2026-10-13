"""Build a shadow copy of the production caches with the priced dropouts added.

Production caches are never written. The shadow lives in research/survivorship_audit/_shadow/:
  capex_intensity_strategy/data/cache/  candidates, prices, dollar volume, EDGAR facts, ticker->CIK (+dropouts)
  es_data/                              electrification_strategy/data with vocabulary hits (+dropouts)
  xsec_cache/                           power_factors only; fundamental panel + power betas rebuild
  grid_resilience/data/cache            symlink to the real ISO cache (read-only)
Dropouts get their old ticker; where that ticker is now someone else's (e.g. B), the shadow column
is suffixed `_DL` so it can never collide with a live name.
"""
import json, pathlib, shutil, sys, time
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from capex_intensity_strategy import edgar, universe
from electrification_strategy import config as es_config
D = pathlib.Path(__file__).parent
BASE = "--base" in sys.argv          # identical rebuild with NO dropouts (the like-for-like control)
S = D/("_shadow_base" if BASE else "_shadow")
CIS, ESD, XC = S/"capex_intensity_strategy"/"data"/"cache", S/"es_data", S/"xsec_cache"
for p in (CIS, ESD, XC): p.parent.mkdir(parents=True, exist_ok=True)
if CIS.exists(): shutil.rmtree(CIS)
shutil.copytree(ROOT/"capex_intensity_strategy"/"data"/"cache", CIS)
if ESD.exists(): shutil.rmtree(ESD)
shutil.copytree(ROOT/"electrification_strategy"/"data", ESD)
XC.mkdir(exist_ok=True); shutil.copy(ROOT/"research"/"xsec_alpha_probe"/"_cache"/"power_factors.parquet", XC)
gr = S/"grid_resilience"/"data"; gr.mkdir(parents=True, exist_ok=True)
if not (gr/"cache").exists(): (gr/"cache").symlink_to(ROOT/"grid_resilience"/"data"/"cache")

if BASE:
    print("base shadow built (no dropouts)"); sys.exit(0)
drop = pd.read_csv(D/"dropout_universe.csv", dtype={"cik": str}); drop["cik"] = drop.cik.str.zfill(10)
# Known-bad price series, excluded rather than mispriced (logged as gaps in the results doc):
EXCLUDE = {"NOVA": "Tiingo serves the pink-sheet NOVA (2008-), not Sunnova's 2019-25 NYSE listing",
           "GTI": "Tiingo GTI history is a different 2014-15 issuer"}
drop = drop[~drop.ticker.isin(EXCLUDE)]
px_d = pd.read_parquet(D/"dropout_prices.parquet"); dv_d = pd.read_parquet(D/"dropout_dollar_volume.parquet")
px = pd.read_parquet(CIS/"prices.parquet"); dv = pd.read_parquet(CIS/"dollar_volume.parquet")
cand = pd.read_parquet(CIS/"candidates.parquet"); t2c = json.loads((CIS/"ticker_cik.json").read_text())
sic_group = universe.load_sic_codes().set_index("sic")["group"].to_dict()
drop["col"] = [t if t not in px.columns and t not in t2c else f"{t}_DL" for t in drop.ticker]
ren = dict(zip(drop.ticker, drop.col))
px_d, dv_d = px_d.rename(columns=ren), dv_d.rename(columns=ren)
px = px.join(px_d.reindex(px.index), how="left"); dv = dv.join(dv_d.reindex(dv.index), how="left")
px.to_parquet(CIS/"prices.parquet"); dv.to_parquet(CIS/"dollar_volume.parquet")
cand = pd.concat([cand, pd.DataFrame({"ticker": drop.col, "cik": drop.cik, "sic": drop.sic.astype(int),
                                      "group": [sic_group.get(int(s)) for s in drop.sic], "source": "dropout"})],
                 ignore_index=True)
cand.to_parquet(CIS/"candidates.parquet")
t2c.update(dict(zip(drop.col, drop.cik))); (CIS/"ticker_cik.json").write_text(json.dumps(t2c))

# EDGAR facts (cached per CIK so a rerun is free)
FC = D/"_cache"/"facts"; FC.mkdir(parents=True, exist_ok=True)
frames = []
for _, r in drop.iterrows():
    f = FC/f"{r.cik}.parquet"
    if not f.exists():
        edgar.company_facts(r.cik).to_parquet(f); time.sleep(0.12)
    x = pd.read_parquet(f)
    if len(x): x.insert(0, "ticker", r.col); frames.append(x)
facts = pd.read_parquet(CIS/"edgar_facts.parquet")
pd.concat([facts] + frames, ignore_index=True).to_parquet(CIS/"edgar_facts.parquet")

# Vocabulary hits: same phrases, forms and span as build_auto_screen's fetch
VC = D/"_cache"/"vocab"; VC.mkdir(parents=True, exist_ok=True)
terms = tuple(es_config.AUTO_CUSTOMER_TERMS) + tuple(es_config.AUTO_POLICY_TERMS)
rows = []
for _, r in drop.iterrows():
    f = VC/f"{r.cik}.json"
    if not f.exists():
        hits = {}
        for ph in terms:
            hits[ph] = edgar.filer_mention_dates(ph, r.cik, "2012-01-01", "2026-08-31", forms=es_config.AUTO_FORMS)
            time.sleep(0.15)
        f.write_text(json.dumps(hits))
    for ph, dates in json.loads(f.read_text()).items():
        rows += [{"ticker": r.col, "phrase": ph, "file_date": d} for d in dates]
vh = pd.read_csv(ESD/"vocabulary_hits.csv", comment="#")
with open(ESD/"vocabulary_hits.csv", "w") as fh:
    fh.write("# SHADOW copy: production hits + survivorship-audit dropouts\n")
    pd.concat([vh, pd.DataFrame(rows)], ignore_index=True).to_csv(fh, index=False)
drop.to_csv(D/"dropout_universe_shadow.csv", index=False)
print("shadow built:", len(drop), "dropouts |", sum(len(x) for x in frames), "facts |", len(rows), "vocab hits")
