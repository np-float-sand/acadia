from __future__ import annotations

"""Long-only thematic-basket construction and hold-shares/periodic-rebalance
simulation. Simple-return convention throughout.
"""

import warnings
from dataclasses import dataclass

import pandas as pd


def apply_cap(weights: pd.Series, cap: float) -> pd.Series:
    w = weights.astype(float).copy()
    n = len(w)
    if n == 0:
        return w
    if n * cap < 1.0 - 1e-9:
        warnings.warn(
            f"single-name cap {cap} infeasible for {n} names; using equal weight",
            RuntimeWarning, stacklevel=2,
        )
        return pd.Series(1.0 / n, index=w.index)
    w = w / w.sum()
    for _ in range(n):
        over = w[w > cap + 1e-12]
        if over.empty:
            break
        excess = float((over - cap).sum())
        w.loc[over.index] = cap
        under = w[w < cap - 1e-12]
        if under.empty:
            break
        w.loc[under.index] += excess * under / under.sum()
    return w


def equal_weight_targets(available, cap: float = 0.25) -> pd.Series:
    names = sorted(available)
    if not names:
        return pd.Series(dtype=float)
    raw = pd.Series(1.0 / len(names), index=names)
    return apply_cap(raw, cap)


def rebalance_dates(start: str, end: str, lag_days: int = 42) -> list:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for year in range(s.year - 1, e.year + 2):
        for m in (3, 6, 9, 12):
            qend = pd.Timestamp(year=year, month=m, day=1) + pd.offsets.MonthEnd(0)
            rd = qend + pd.Timedelta(days=lag_days)
            if s <= rd <= e:
                out.append(rd)
    return sorted(out)


@dataclass
class BasketResult:
    returns: pd.Series
    weights: pd.DataFrame
    rebalances: list


def simulate_basket(
    prices: pd.DataFrame, start: str, end: str,
    lag_days: int = 42, cap: float = 0.25, target_fn=None,
) -> BasketResult:
    px = prices.loc[start:end].sort_index().ffill()
    if px.empty:
        return BasketResult(pd.Series(dtype=float), pd.DataFrame(), [])

    if target_fn is None:
        def target_fn(available, asof):
            return equal_weight_targets(available, cap)

    raw_dates = [d for d in rebalance_dates(start, end, lag_days) if d <= px.index[-1]]
    snapped = []
    for d in raw_dates:
        pos = px.index.get_indexer([d], method="bfill")[0]
        if pos != -1:
            snapped.append(px.index[pos])
    form_dates = sorted(set([px.index[0], *snapped]))

    value = pd.Series(index=px.index, dtype=float)
    weights_log: dict = {}
    shares = pd.Series(dtype=float)
    cur_val = 1.0

    for i, day in enumerate(px.index):
        if not shares.empty:
            held = px.loc[day, shares.index].fillna(0.0)
            cur_val = float((shares * held).sum())
        value.iloc[i] = cur_val
        if day in form_dates:
            row = px.loc[day].dropna()
            available = [t for t in row.index if row[t] > 0]
            tgt = target_fn(available, day)
            if tgt is not None and not tgt.empty:
                tgt = tgt / tgt.sum()
                shares = (cur_val * tgt) / row.reindex(tgt.index)
                weights_log[day] = tgt

    ret = value.pct_change().dropna()
    wdf = pd.DataFrame(weights_log).T.sort_index()
    wdf = wdf.reindex(columns=sorted(wdf.columns)).fillna(0.0)
    return BasketResult(ret, wdf, [d for d in form_dates if d != px.index[0]])
