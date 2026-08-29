from __future__ import annotations

"""Value-chain reframe (spec 2026-08-29): frozen maker/contractor buckets, a
gross-margin + backlog-coverage signal, and the two weight builders. Plain
arithmetic on config constants — no z-scoring, no build_factor."""

import numpy as np
import pandas as pd

from grid_equipment_basket import config

_BUCKET = {t: "maker" for t in config.BUCKET_MAKERS}
_BUCKET.update({t: "contractor" for t in config.BUCKET_CONTRACTORS})


def bucket_of(ticker: str) -> str:
    try:
        return _BUCKET[ticker]
    except KeyError:
        raise KeyError(f"{ticker} is not in a frozen value-chain bucket") from None


def composite_rank(signal_a: pd.Series, signal_b: pd.Series, names: list[str]) -> pd.Series:
    ra = signal_a.reindex(names).rank(method="dense", ascending=True)
    rb = signal_b.reindex(names).rank(method="dense", ascending=True)
    both = pd.concat([ra, rb], axis=1)
    return both.mean(axis=1, skipna=True).reindex(names)
