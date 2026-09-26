"""Regenerate the `screen_auto` build artifacts from SEC filings.

    python -m electrification_strategy.build_auto_screen            # from caches
    python -m electrification_strategy.build_auto_screen --refresh  # re-query EDGAR

Writes `data/vocabulary_hits.csv` (raw full-text hits) and
`data/screen_auto_members.csv` (the resulting membership). Both are build
artifacts: reproducible from filings, committed only so a backtest can run
without a 30-minute EDGAR round trip.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from capex_intensity_strategy import edgar, fundamentals, pipeline

from electrification_strategy import auto_screen, config


def fetch_vocabulary_hits(ciks: dict[str, str], years) -> pd.DataFrame:
    """One EDGAR full-text query per (filer, phrase) across the whole span."""
    years = list(years)
    start = f"{min(years) - config.AUTO_LOOKBACK_YEARS}-01-01"
    end = f"{max(years) - 1}-12-31"
    terms = tuple(config.AUTO_CUSTOMER_TERMS) + tuple(config.AUTO_POLICY_TERMS)
    rows = []
    for i, (ticker, cik) in enumerate(sorted(ciks.items())):
        for phrase in terms:
            for date in edgar.filer_mention_dates(phrase, cik, start, end,
                                                  forms=config.AUTO_FORMS):
                rows.append({"ticker": ticker, "phrase": phrase, "file_date": date})
            time.sleep(0.15)
        if i % 20 == 0:
            print(f"  vocabulary {i}/{len(ciks)}", flush=True)
    return pd.DataFrame(rows, columns=["ticker", "phrase", "file_date"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="build_auto_screen")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-08-31")
    ap.add_argument("--refresh", action="store_true", help="re-query EDGAR full-text search")
    args = ap.parse_args(argv)

    built = pipeline.build(start=args.start, end=args.end)
    score, mask, candidates = built["score"], built["mask"], built["candidates"]
    panel = fundamentals.ttm_panel(edgar.fetch_facts({}), 
                                   pd.date_range(args.start, args.end, freq="ME"))
    power = set(candidates.loc[candidates["group"] == "POWER", "ticker"])

    if args.refresh:
        light = auto_screen.light_capex_mask(score, mask)
        cand = sorted(light.columns[light.any()])
        cik = edgar.ticker_to_cik()
        hits = fetch_vocabulary_hits({t: cik[t] for t in cand if t in cik},
                                     range(int(args.start[:4]), int(args.end[:4]) + 1))
        path = config.DATA_DIR / auto_screen.VOCABULARY_FILE
        with open(path, "w") as fh:
            fh.write("# EDGAR full-text hits behind `screen_auto`'s vocabulary filters.\n")
            fh.write(f"# Regenerate: python -m electrification_strategy.build_auto_screen --refresh\n")
            hits.to_csv(fh, index=False)
    else:
        hits = auto_screen.load_vocabulary_hits()

    members = auto_screen.build_membership(score, mask, panel, hits, power_tickers=power)
    long = (members.stack().rename("v").reset_index()
            .rename(columns={"level_0": "date", "level_1": "ticker"}))
    long = long[long["v"]][["date", "ticker"]]
    path = config.DATA_DIR / auto_screen.MEMBERSHIP_FILE
    with open(path, "w") as fh:
        fh.write("# BUILD ARTIFACT -- not a hand-recorded input. Every row is derived from\n")
        fh.write("# SEC SIC codes, XBRL filings and full-text search by\n")
        fh.write("# `python -m electrification_strategy.build_auto_screen`.\n")
        long.to_csv(fh, index=False)
    print(f"wrote {path} -- {len(long)} rows, "
          f"{members.sum(axis=1).replace(0, float('nan')).mean():.1f} names/month")


if __name__ == "__main__":
    main()
