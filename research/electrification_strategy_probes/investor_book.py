"""The investor proposal's construction, with every trade costed.

    long  : `screen_auto` A1+A4 members, top half by the capex-light + revenue-acceleration composite,
            equal weight, quarterly rebalance; a held name stays while it remains in the top LONG_HOLD
            of the list (buffer), new names enter from the top half.
    size  : 20% vol target x stretch cut, traded only when the target moves more than BAND from the
            drifted exposure (no-trade band); 15% GLD/IEF sleeve reset monthly.
    short : bottom quintile of the 14-signal composite over rule-2 names not held long, re-formed QUARTERLY,
            sized $0.25 per $1 of book and reset MONTHLY (drifts in between).
    rule 11: invested only while the 10-year real yield is above zero AND the long-eligible suppliers'
            median revenue growth after inflation is above zero; otherwise everything in 3-month T-bills.

Every trade is charged `bp` (default 10bp): long name changes, long resizing, sleeve resets, short
re-formation, short resizing, rule-11 switches (~1.25x capital each). Borrow is charged daily on the
short notional. Held exposure and the short/sleeve weights drift with prices between trades.
"""
from __future__ import annotations

import pandas as pd

import build_tophalf_proposal as t
from electrification_strategy import backtest, config, hedge_overlay, valuation_overlay
from grid_equipment_basket.basket import apply_cap, simulate_basket
from research.xsec_alpha_probe import data, short_compare

BAND = 0.20            # no-trade band on long-side exposure, in exposure units
LONG_HOLD = 0.65       # a held long name stays while in the top 65% of the eligible list
SHORT_FREQ = "Q"       # short basket re-formed at quarter-ends
SWITCH_TRADE = 1.25    # capital traded when rule 11 switches (long 0.85 + sleeve 0.15 + short 0.25)
_CACHE: dict = {}      # expensive pieces (long book, short basket, 14-signal score), keyed by their settings


def long_book(members, comp, px, hold=LONG_HOLD, rank=True):
    """Top half by composite with a hold buffer (rank=False: every name passing rules 1-5, equal
    weight). Returns (daily returns, daily weights)."""
    key = ("long", id(members), hold, rank)
    if key in _CACHE:
        return _CACHE[key]
    state = {"held": set()}

    def fn(available, asof):
        asof = pd.Timestamp(asof)
        upto = members.loc[:asof]
        if upto.empty:
            return pd.Series(dtype=float)
        row = upto.iloc[-1]
        counts = px.loc[:asof].notna().sum()
        listed = set(counts[counts >= config.LISTING_MIN_DAYS].index)
        elig = sorted(set(row.index[row.astype(bool)]) & set(available) & listed)
        if not elig:
            return pd.Series(dtype=float)
        if not rank:
            raw = pd.Series(1.0, index=elig)
            return apply_cap(raw / raw.sum(), config.MAX_SINGLE_NAME_WEIGHT)
        up = comp.loc[:asof]
        zz = (up.iloc[-1] if not up.empty else pd.Series(dtype=float)).reindex(elig).fillna(0.0)
        order = list(zz.sort_values(ascending=False).index)
        n = max(1, len(order) // 2)
        keep = [x for x in order[: max(1, int(round(len(order) * hold)))] if x in state["held"]] if hold else []
        pick = keep + [x for x in order if x not in keep][: max(0, n - len(keep))]
        state["held"] = set(pick)
        raw = pd.Series(1.0, index=sorted(pick))
        return apply_cap(raw / raw.sum(), config.MAX_SINGLE_NAME_WEIGHT)

    cand = px[[c for c in members.columns if c in px.columns]]
    br = simulate_basket(cand, t.START, t.END, config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, fn)
    _CACHE[key] = (br.returns, br.weights)
    return _CACHE[key]


def banded_exposure(raw, spy, band=BAND):
    """Vol-target x stretch-cut exposure, traded only outside the band. Returns (held, |trade|)."""
    rf = config.RF / config.ANN
    vscal = backtest._vol_scalar(raw)
    vmult = valuation_overlay.extension_multiplier((1 + raw).cumprod(), spy.reindex(raw.index), scale=1.0)
    target = (vmult.reindex(raw.index).fillna(1.0) * vscal.reindex(raw.index)).clip(
        upper=config.VOL_MAX_LEVERAGE).fillna(1.0)
    held, trade = [], []
    h, rp = float(target.iloc[0]), 0.0
    for d, e in target.items():
        hd = h * (1 + rp) / (1 + h * rp + (1 - h) * rf)
        hn = e if abs(e - hd) > band else hd
        trade.append(abs(hn - hd))
        held.append(hn)
        h, rp = hn, float(raw.loc[d]) if pd.notna(raw.loc[d]) else 0.0
    return pd.Series(held, index=raw.index), pd.Series(trade, index=raw.index)


def short_basket(px, days, freq=SHORT_FREQ, long_weights=None, q=None, key=None):
    """Bottom quintile of rule-2 names by the 14-signal composite, re-formed at month/quarter ends,
    skipping any name the long book holds on the formation date (no name is both long and short).
    Returns (daily return, re-formation trade per $1 of basket, names held, {formation date: names})."""
    ck = ("short", freq, q, key)
    if key is not None and ck in _CACHE:
        return _CACHE[ck]
    q = t.NAME_Q if q is None else q
    if "comp14" not in _CACHE:
        _CACHE["comp14"] = short_compare.all14_short()[1]
    comp14 = _CACHE["comp14"]
    univ = data.universe()
    rets = px.pct_change(fill_method=None)
    r2 = t.rule2_filter()
    ends = [d for d in comp14.index if pd.Timestamp(t.START) - pd.offsets.MonthEnd(3) <= d <= pd.Timestamp(t.END)]
    if freq == "Q":
        ends = [d for d in ends if d.month in (3, 6, 9, 12)]
    elif freq == "2M":
        ends = [d for d in ends if d.month % 2 == 0]
    elif freq == "S":
        ends = [d for d in ends if d.month in (6, 12)]
    out, trade = pd.Series(0.0, index=days), pd.Series(0.0, index=days)
    held_n, log = pd.Series(0, index=days), {}
    w_prev = pd.Series(dtype=float)
    for i, me in enumerate(ends):
        seg = days[(days > me) & (days <= (ends[i + 1] if i + 1 < len(ends) else pd.Timestamp(t.END)))]
        if seg.empty or me not in univ.index:
            continue
        priced = px.loc[:me].iloc[-1].notna()
        sc = comp14.loc[me].where(univ.loc[me].reindex(comp14.columns, fill_value=False)
                                  & priced.reindex(comp14.columns, fill_value=False)).dropna()
        sc = sc.drop([x for x in r2(me, sc.index) if x in sc.index])
        if long_weights is not None:
            lw = long_weights.loc[:me]
            if not lw.empty:
                sc = sc.drop([x for x in lw.columns[lw.iloc[-1] > 0] if x in sc.index])
        if len(sc) < 10:
            continue
        names = list(sc[sc <= sc.quantile(q)].index)
        log[me] = sorted(names)
        w = pd.Series(1.0 / len(names), index=names)
        trade.loc[seg[0]] = w.sub(w_prev, fill_value=0).abs().sum()
        val = w.copy()
        for d in seg:
            nv = val * (1 + rets.loc[d, names].fillna(0.0))
            out.loc[d] = nv.sum() / val.sum() - 1
            val = nv
            held_n.loc[d] = len(names)
        w_prev = val / val.sum()
    if key is not None:
        _CACHE[ck] = (out, trade, held_n, log)
    return out, trade, held_n, log


def build(members, comp, px, spy, dfii10, ry, cash, short="names", bp=10.0, fee=0.0, growth=None,
          band=BAND, long_hold=LONG_HOLD, rank=True, overlay=True, short_k=None, short_q=None,
          short_freq=SHORT_FREQ, ry_thr=0.0, g_thr=0.0):
    """The strategy's daily returns net of `bp` trading costs, borrow and `fee` (a year), gated by
    rule 11. short="names" (the proposal), "PAVE" (the simple alternative) or None (long only).
    Rule 11 uses the real yield (> ry_thr) and, if `growth` is given, real supplier growth (> g_thr);
    pass ry clipped high to switch it off. overlay=False drops the vol target, stretch cut and sleeve."""
    cost = bp / 1e4
    rf = config.RF / config.ANN
    k = t.FIXED_K if short_k is None else short_k
    raw, W = long_book(members, comp, px, hold=long_hold, rank=rank)
    days = raw.loc[t.START:t.END].index
    raw, W = raw.reindex(days).fillna(0.0), W.reindex(days).ffill().fillna(0.0)
    if overlay:
        hx, rtrade = banded_exposure(raw, spy.reindex(days), band=band)
    else:
        hx, rtrade = pd.Series(1.0, index=days), pd.Series(0.0, index=days)
    core = hx * raw + (1 - hx) * rf
    sl = hedge_overlay.sleeve_returns(px).reindex(days).fillna(0.0)
    ws = config.HEDGE_SLEEVE_WEIGHT if overlay else 0.0
    month_end = set(pd.Series(days, index=days).resample("ME").last().values)
    if short == "names":
        b, btrade, bn, blog = short_basket(px, days, freq=short_freq, long_weights=W, q=short_q,
                                           key=(long_hold, rank))
        borrow = t.NAME_BORROW
    elif short is None:
        k = 0.0
        b, btrade, bn, blog, borrow = (pd.Series(0.0, index=days), pd.Series(0.0, index=days),
                                       pd.Series(0, index=days), {}, 0.0)
    else:
        b = px[short].pct_change().reindex(days).fillna(0.0)
        btrade, bn, blog, borrow = pd.Series(0.0, index=days), pd.Series(0, index=days), {}, t.HEDGE_BORROW
    book, sleeve_tr, wsl = [], [], ws
    ls, short_tr, s = [], [], k
    for d in days:
        rb = (1 - wsl) * core[d] + wsl * sl[d]
        wd = wsl * (1 + sl[d]) / (1 + rb)
        if d in month_end:
            sleeve_tr.append(abs(wd - ws) * 2)
            wsl = ws
        else:
            sleeve_tr.append(0.0)
            wsl = wd
        r = rb - s * b[d] - s * borrow / config.ANN
        sd = s * (1 + b[d]) / (1 + r)
        if d in month_end:
            short_tr.append(abs(sd - k))
            s = k
        else:
            short_tr.append(0.0)
            s = sd
        book.append(rb)
        ls.append(r)
    ls = pd.Series(ls, index=days)
    trades = pd.DataFrame({"long names": W.diff().abs().sum(axis=1).fillna(0.0) * hx * (1 - ws),
                           "long resize": rtrade * (1 - ws), "sleeve": pd.Series(sleeve_tr, index=days),
                           "short basket": btrade * k, "short size": pd.Series(short_tr, index=days)})
    net = ls - trades.sum(axis=1) * cost - fee / config.ANN
    on = ry.reindex(days, method="ffill").shift(1) > ry_thr
    if growth is not None:                         # both conditions, each known the day before
        on &= growth.reindex(days, method="ffill").shift(1).fillna(1.0) > g_thr
    w = on.astype(float)
    w.iloc[0] = 1.0
    tb = cash.reindex(days, method="ffill").fillna(0.0) / 100 / config.ANN
    switch = w.diff().abs().fillna(0.0) * SWITCH_TRADE
    gated = w * net + (1 - w) * tb - switch * cost
    years = (days[-1] - days[0]).days / 365.25
    turnover = (trades.mul(w, axis=0).sum() / years).to_dict()
    turnover["rule-11 switches"] = float(switch.sum() / years)
    invested = float(trades[w > 0].sum().sum() / (years * float(w.mean())))
    return {"returns": gated, "invested": w, "turnover": turnover, "turnover_invested": invested,
            "long_held": (W > 0).sum(axis=1), "short_held": bn, "short_log": blog, "short_returns": b,
            "long_weights": W, "exposure": hx}
