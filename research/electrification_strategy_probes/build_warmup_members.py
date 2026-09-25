"""2018 `screen_auto` membership for the hedge-beta WARM-UP only (2026-09-25).

    python research/electrification_strategy_probes/build_warmup_members.py

The published membership (`output_html/img/a1a4/_build_cache.pkl`) starts 2019-01, so the
"trailing year" behind every hedge ratio in the selection proposal was an empty 2018 book and
the hedge ramped up from ~0 through 2019. This runs the IDENTICAL rule (same quantile, lookback,
power exclusion) from 2018-01, and keeps only the months before 2019 -- it never touches the
published 2019+ membership. Data limit: EDGAR vocabulary hits start 2017-01, so 2018 sees one year
of 10-K wording (the published 2019 sees two). Writes output_html/img/tophalf/_warmup_members.pkl.
"""
from __future__ import annotations

import os
import pickle
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_a1a4_proposal as a1  # noqa: E402
from capex_intensity_strategy import edgar, fundamentals, pipeline  # noqa: E402
from electrification_strategy import auto_screen  # noqa: E402

WARM_START = "2018-01-01"
OUT = os.path.join(ROOT, "output_html", "img", "tophalf", "_warmup_members.pkl")


def main():
    built = pipeline.build(start=WARM_START, end=a1.END)
    score, mask, cands = built["score"], built["mask"], built["candidates"]
    power = set(cands.loc[cands["group"] == "POWER", "ticker"])
    panel = fundamentals.ttm_panel(edgar.fetch_facts({}), pd.date_range(WARM_START, a1.END, freq="ME"))
    hits = auto_screen.load_vocabulary_hits()
    members = auto_screen.build_membership(score, mask, panel, hits, power_tickers=power,
                                           quantile=a1.QUANTILE, lookback=a1.LOOKBACK)
    pub = pickle.load(open(os.path.join(ROOT, "output_html", "img", "a1a4", "_build_cache.pkl"), "rb"))["members"]
    # consistency check: the rebuilt rule must reproduce the published 2019+ membership
    both = members.loc["2019-01-01":].reindex(columns=pub.columns.union(members.columns), fill_value=False)
    ref = pub.loc["2019-01-01":].reindex(index=both.index, columns=both.columns, fill_value=False)
    agree = float((both.astype(bool) == ref.astype(bool)).all(axis=1).mean())
    warm = members.loc[:"2018-12-31"]
    pickle.dump(warm, open(OUT, "wb"))
    print(f"2018 names/month: {warm.sum(axis=1).to_dict()}")
    print(f"rebuilt 2019+ membership matches the published one in {agree:.0%} of months")


if __name__ == "__main__":
    main()
