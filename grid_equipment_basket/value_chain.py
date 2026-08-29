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
        latest_row = g.iloc[-1]
        latest_backlog = float(latest_row["metric_value"])
        metric_unit = latest_row["metric_unit"]
        if metric_unit == "USD":
            scale = 1.0
        elif metric_unit == "USD_million":
            scale = 1e6
        else:
            raise ValueError(f"unhandled metric_unit {metric_unit!r} for {tkr}")
        latest_backlog_usd = latest_backlog * scale
        out[tkr] = latest_backlog_usd / rev if rev and rev == rev else float("nan")
    return pd.Series(out, dtype=float)


def coverage_change_signal(backlog_df: pd.DataFrame, fund_df: pd.DataFrame,
                           asof: pd.Timestamp) -> pd.Series:
    asof = pd.Timestamp(asof)
    now = coverage_ratio(backlog_df, fund_df, asof)
    prior = coverage_ratio(backlog_df, fund_df, asof - pd.Timedelta(days=365))
    names = sorted(set(now.index) | set(prior.index))
    return (now.reindex(names) - prior.reindex(names))


from grid_equipment_basket.basket import apply_cap


def _signals(available, asof, fund_df, backlog_df):
    margin_sig = margin_data.ttm_gross_margin_signal(fund_df, asof).reindex(available)
    cover_sig = coverage_change_signal(backlog_df, fund_df, asof).reindex(available)
    return margin_sig, cover_sig


def _within_bucket_factor(comp: pd.Series, within_top: float, within_bottom: float) -> pd.Series:
    factor = pd.Series(1.0, index=comp.index)
    ranked = comp.dropna().sort_values()
    k = len(ranked)
    if k >= 2:
        half = k // 2
        factor.loc[ranked.index[:half]] = within_bottom
        factor.loc[ranked.index[k - half:]] = within_top
    return factor


def value_chain_tilt_targets(available, asof, fund_df, backlog_df, *, cap: float = 0.25,
                             base_maker: float = 1.25, base_contractor: float = 0.75,
                             within_top: float = 1.10, within_bottom: float = 0.90) -> pd.Series:
    names = sorted(available)
    if not names:
        return pd.Series(dtype=float)
    base = pd.Series(1.0 / len(names), index=names)
    margin_sig, cover_sig = _signals(names, asof, fund_df, backlog_df)

    factor = pd.Series(1.0, index=names)
    for bkt, base_mult in (("maker", base_maker), ("contractor", base_contractor)):
        bkt_names = [t for t in names if bucket_of(t) == bkt]
        if not bkt_names:
            continue
        comp = composite_rank(margin_sig, cover_sig, bkt_names)
        within = _within_bucket_factor(comp, within_top, within_bottom)
        factor.loc[bkt_names] = base_mult * within.reindex(bkt_names).fillna(1.0)

    tilted = base * factor
    tilted = tilted / tilted.sum()
    return apply_cap(tilted, cap)
