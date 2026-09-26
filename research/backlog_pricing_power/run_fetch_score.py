"""
Fetch + score every 10-K/10-Q for the 27-name `screen` universe, 2017-01 through
2026-08 (enough lead-in before the 2019-01 panel start, per docs/backlog-
pricing-power-preregistration.md §6). Writes research/backlog_pricing_power/
_cache/filing_scores.parquet. Resumable: skips tickers already fully written
to the running output, and the underlying edgar_filings fetch/text cache means
a restart never re-downloads.

Not production code. Throwaway research script. Run:
    python3 research/backlog_pricing_power/run_fetch_score.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from cik_map import CIK_BY_TICKER, SCREEN_27
from edgar_filings import fetch_document_text, list_filings
from score import extract_mdna, score_text

OUT = Path(__file__).parent / "_cache" / "filing_scores.parquet"
START, END = "2017-01-01", "2026-08-31"


def score_ticker(ticker: str, cik: str) -> pd.DataFrame:
    flist = list_filings(ticker, cik)
    flist = flist[(flist.filingDate >= START) & (flist.filingDate <= END)]
    rows = []
    for _, r in flist.iterrows():
        text = fetch_document_text(ticker, cik, r["accessionNumber"], r["primaryDocument"])
        if not text:
            continue
        mdna, ok = extract_mdna(text, r["form"])
        sc = score_text(mdna)
        row = {"ticker": ticker, "form": r["form"], "accessionNumber": r["accessionNumber"],
               "filingDate": r["filingDate"], "reportDate": r["reportDate"],
               "mdna_extracted": ok}
        row.update(sc)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    done_tickers: set[str] = set()
    frames = []
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        frames.append(existing)
        done_tickers = set(existing["ticker"].unique())
        print(f"resuming: {len(done_tickers)} tickers already scored")

    for t in SCREEN_27:
        if t in done_tickers:
            continue
        print(f"scoring {t} ...", flush=True)
        df = score_ticker(t, CIK_BY_TICKER[t])
        print(f"  {t}: {len(df)} filings scored, "
              f"{df['mdna_extracted'].sum() if len(df) else 0} extracted OK", flush=True)
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_parquet(OUT)

    final = pd.concat(frames, ignore_index=True)
    print(f"\nDONE. {len(final)} filings total across {final['ticker'].nunique()} tickers.")
    print(f"MD&A extraction success rate: {final['mdna_extracted'].mean():.1%}")
    print(final.groupby("ticker")["mdna_extracted"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
