"""EDGAR full-text hits for 10-Ks filed 2012-01-01..2016-12-31 (rule 2 customer terms + rule 4 policy
terms), for every name that can be in the long or short universe in 2016-18. Feeds the 2016-18 slack-
grid holdout (`holdout_2016_18.py`). Resumable; writes electrification_strategy/data/vocabulary_hits_2012_2016.csv.

    python research/electrification_strategy_probes/holdout_fetch_hits.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from capex_intensity_strategy import edgar, pipeline  # noqa: E402
from electrification_strategy import auto_screen, config  # noqa: E402
from research.xsec_alpha_probe import data  # noqa: E402

OUT = os.path.join(ROOT, "electrification_strategy", "data", "vocabulary_hits_2012_2016.csv")
START, END = "2012-01-01", "2016-12-31"


def tickers() -> list[str]:
    u = data.universe()
    short_side = set(u.columns[u.loc["2015-01-01":"2018-12-31"].any()])
    built = pipeline.build(start="2015-01-01", end="2018-12-31")
    light = auto_screen.light_capex_mask(built["score"], built["mask"], quantile=0.60)
    long_side = set(light.columns[light.loc[:"2018-12-31"].any()])
    return sorted(short_side | long_side)


def main():
    names = tickers()
    cik = edgar.ticker_to_cik()
    terms = tuple(config.AUTO_CUSTOMER_TERMS) + tuple(config.AUTO_POLICY_TERMS)
    done = set()
    if os.path.exists(OUT):
        done = {r["ticker"] for r in csv.DictReader(line for line in open(OUT) if not line.startswith("#"))
                if r["phrase"] == "__done__"}
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        if new:
            fh.write(f"# EDGAR full-text hits, 10-K, {START}..{END}, customer + policy terms (2016-18 holdout).\n")
            fh.write("# Rows with phrase __done__ mark a completed ticker (resume markers).\n")
        w = csv.writer(fh)
        if new:
            w.writerow(["ticker", "phrase", "file_date"])
        todo = [x for x in names if x in cik and x not in done]
        print(f"{len(names)} names, {len(todo)} to fetch", flush=True)
        for i, tk in enumerate(todo):
            for phrase in terms:
                for attempt in range(4):
                    try:
                        dates = edgar.filer_mention_dates(phrase, cik[tk], START, END, forms=config.AUTO_FORMS)
                        break
                    except Exception as e:  # noqa: BLE001 -- network: retry, then stop loudly
                        print("retry", tk, phrase, e, flush=True)
                        time.sleep(5 * (attempt + 1))
                else:
                    raise SystemExit(f"failed {tk} {phrase}")
                for d in dates:
                    w.writerow([tk, phrase, d])
                time.sleep(0.15)
            w.writerow([tk, "__done__", ""])
            fh.flush()
            if i % 20 == 0:
                print(f"{i}/{len(todo)} {tk}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
