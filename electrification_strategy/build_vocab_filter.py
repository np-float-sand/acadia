"""Build `data/vocabulary_hits_dc_excluded.csv`: the "data center(s)" full-text hits that don't count.

    python -m electrification_strategy.build_vocab_filter

EDGAR full-text search matches "data center(s)" anywhere in a 10-K, including the filer's own IT
("information stored in our data centers", cybersecurity boilerplate). For every such hit in the
vocabulary-hit files, this reads that filing's Item 1 (Business) and keeps the hit only if
`auto_screen.is_market_mention` finds a market use. Item 1 text comes from the 10-K cache in
research/xsec_alpha_probe/_cache/tenk_item1 (filings missing there are fetched and added); a hit
whose Item 1 can't be found is kept. The output is a build artifact, reproducible from filings.
See docs/survivorship-audit-results.md (vocabulary clean-up, 2026-10-12).
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from capex_intensity_strategy import config as cis_config
from electrification_strategy import auto_screen, config

FILES = ("vocabulary_hits.csv", "vocabulary_hits_short_universe.csv")
WINDOW_DAYS = 10             # full-text file_date vs the submissions filingDate


def main() -> None:
    from research.xsec_alpha_probe import tenk
    cache = defaultdict(list)
    for f in tenk.TENK.glob("*.txt"):
        t, dt = f.stem.rsplit("_", 1)
        cache[t].append((pd.Timestamp(dt), f))
    t2c = json.loads((cis_config.CACHE_DIR / "ticker_cik.json").read_text())

    def item1_path(ticker: str, fd: pd.Timestamp):
        near = [(abs((d - fd).days), f) for d, f in cache.get(ticker, []) if abs((d - fd).days) <= WINDOW_DAYS]
        if near:
            return min(near)[1]
        cik = t2c.get(ticker)
        if not cik:
            return None
        try:
            fl = tenk.filings(cik)
        except Exception:
            return None
        fl = fl[(pd.to_datetime(fl["filingDate"]) - fd).abs().dt.days <= WINDOW_DAYS]
        for _, r in fl.iterrows():
            acc = r["accessionNumber"].replace("-", "")
            html = tenk._get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{r['primaryDocument']}").content
            p = tenk.TENK / f"{ticker}_{r['filingDate']}.txt"
            p.write_text(tenk.item1(html))
            cache[ticker].append((pd.Timestamp(r["filingDate"]), p))
            return p
        return None

    hits = auto_screen.load_vocabulary_hits(FILES, clean=False)
    dc = hits[hits["phrase"].isin(config.AUTO_DC_TERMS)]
    out, missing = [], 0
    for _, r in dc.iterrows():
        p = item1_path(r["ticker"], pd.Timestamp(r["file_date"]))
        if p is None:
            missing += 1
            continue
        if not auto_screen.is_market_mention(p.read_text(errors="ignore")):
            out.append(r[["ticker", "phrase", "file_date"]])
    ex = pd.DataFrame(out, columns=["ticker", "phrase", "file_date"]).sort_values(["ticker", "file_date"])
    path = config.DATA_DIR / auto_screen.DC_EXCLUDED_FILE
    with open(path, "w") as fh:
        fh.write("# BUILD ARTIFACT -- \"data center(s)\" full-text hits whose 10-K Item 1 has no market use\n")
        fh.write("# (own-IT / cybersecurity boilerplate). Regenerate: python -m electrification_strategy.build_vocab_filter\n")
        ex.to_csv(fh, index=False)
    print(f"wrote {path} -- {len(ex)} of {len(dc)} data-center hits excluded; {missing} kept for lack of Item 1 text")


if __name__ == "__main__":
    main()
