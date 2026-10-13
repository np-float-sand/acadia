"""Shadow variants with gas transmission / distribution names removed from the rule-2 short pool.

Gas pipeline and distribution filers (SIC 4922 Natural Gas Transmission, 4923 Gas Transmission &
Distribution, 4924 Natural Gas Distribution) have every vocabulary hit removed, so they fail rule 2
and can't be shorted. The long book already excludes them (POWER group). Electric-and-gas combination
utilities (4931) are untouched.
    python build_gasfix.py _shadow_base  _shadow_gas    # gas fix only
    python build_gasfix.py _shadow_dcfix _shadow_both   # data-center fix + gas fix
"""
import pathlib, shutil, sys
import pandas as pd
D = pathlib.Path(__file__).parent; src, dst = D/sys.argv[1], D/sys.argv[2]
GAS = {4922, 4923, 4924}
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst, symlinks=True)
cand = pd.read_parquet(dst/"capex_intensity_strategy/data/cache/candidates.parquet")
gas = set(cand.loc[cand.sic.isin(GAS), "ticker"])
for fn in ("vocabulary_hits.csv", "vocabulary_hits_short_universe.csv"):
    h = pd.read_csv(dst/"es_data"/fn, comment="#"); n = len(h)
    h = h[~h.ticker.isin(gas)]
    with open(dst/"es_data"/fn, "w") as fh:
        fh.write(f"# SHADOW ({dst.name}): gas T&D (SIC 4922/4923/4924) hits removed\n"); h.to_csv(fh, index=False)
    print(fn, "removed", n - len(h), "hits")
print(len(gas), "gas T&D tickers:", sorted(gas))
