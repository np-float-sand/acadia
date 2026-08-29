from __future__ import annotations

"""Value-chain reframe (spec 2026-08-29): frozen maker/contractor buckets, a
gross-margin + backlog-coverage signal, and the two weight builders. Plain
arithmetic on config constants — no z-scoring, no build_factor."""

import numpy as np
import pandas as pd

from grid_equipment_basket import config, margin_data

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


def coverage_ratio(backlog_df: pd.DataFrame, fund_df: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    ttm_rev = margin_data.ttm_revenue(fund_df, asof)
    vis = backlog_df[(backlog_df["availability_date"] <= asof)
                     & (backlog_df["disclosure_type"] != "book_to_bill_only")]
    out: dict[str, float] = {}
    for tkr, g in vis.groupby("ticker"):
        g = g.sort_values("quarter_end")
        latest_qe = g["quarter_end"].iloc[-1]
        if (asof - latest_qe).days > config.VC_STALENESS_MAX_DAYS:
            out[tkr] = float("nan")
            continue
        rev = ttm_rev.get(tkr, float("nan"))
        latest_backlog = float(g["metric_value"].iloc[-1])
        out[tkr] = latest_backlog / rev if rev and rev == rev else float("nan")
    return pd.Series(out, dtype=float)


def coverage_change_signal(backlog_df: pd.DataFrame, fund_df: pd.DataFrame,
                           asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    now = coverage_ratio(backlog_df, fund_df, asof)
    prior = coverage_ratio(backlog_df, fund_df, asof - pd.Timedelta(days=365))
    names = sorted(set(now.index) | set(prior.index))
    return (now.reindex(names) - prior.reindex(names))
