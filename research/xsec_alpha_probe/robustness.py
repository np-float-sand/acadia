"""Robustness of the selection book (plan checks 5 + 6), with both hedge sleeves.

    python -m research.xsec_alpha_probe.robustness

Fixed 2026-09-25, before results. Everything runs through the published book's
path (build_tophalf_proposal): its membership, daily simulate_basket, quarterly
rebalance (quarter-end + 42 days), 20% vol target, valuation de-lever.

  long side    top 40% / 50% / 60% of the book's names by the frozen composite
               (equal weight), and top 50% weighted by rank within the top half
  sleeve       15% GLD/IEF (shipped)  vs  25% GLD/BTAL 50/50 (the write-up's
               "What we would change" sleeve)
  short side   none; PAVE (the doc's default ETF hedge); the doc's single-name
               short sleeve (bottom quintile of the universe by the 14-signal
               score, `build_tophalf_proposal.name_short_basket`); and the bottom
               half of the book's own names. Each at the trailing-252-day beta
               (the doc's `hedge()`: beta ending the prior day, estimated on the
               same rule run over 2018 via `_warmup_members.pkl` -- no back-fill)
               and, for PAVE and the 14-signal short, at a fixed $0.25 per $1.
               Borrow: ETF 0.5%/yr, single names 1%/yr; 10bp on every change in
               hedge size and on each basket's own re-formation.

Judged on Sharpe and Calmar from 2023 (the grid-tight regime), 2022 on, and full.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from electrification_strategy import backtest, config, hedge_overlay
from electrification_strategy.fred import fetch_series
from grid_equipment_basket.basket import apply_cap, simulate_basket
from research.xsec_alpha_probe import data

sys.path.insert(0, os.path.join(str(data.data_root()), "research", "electrification_strategy_probes"))
import build_tophalf_proposal as tp  # noqa: E402

START, END, WARM = tp.START, tp.END, tp.WARMUP_START
WINDOWS = {"full": (START, END), "2022+": ("2022-01-01", END), "2023+": ("2023-01-01", END)}


def target_fn(kind: str, q: float, panel: pd.DataFrame, members: pd.DataFrame, px: pd.DataFrame):
    """kind: 'top' (top q, EW), 'rank' (top q, rank-weighted), 'bottom' (bottom q, EW)."""
    def _fn(available, asof):
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
        up = panel.loc[:asof]
        z = (up.iloc[-1] if not up.empty else pd.Series(dtype=float)).reindex(elig).fillna(0.0)
        n = max(1, int(round(len(z) * q)))
        if kind == "bottom":
            pick = z.sort_values().index[:n]
            raw = pd.Series(1.0, index=pick)
        else:
            top = z.sort_values(ascending=False).index[:n]
            raw = (pd.Series(np.arange(n, 0, -1, dtype=float), index=top) if kind == "rank"
                   else pd.Series(1.0, index=top))
        return apply_cap(raw / raw.sum(), config.MAX_SINGLE_NAME_WEIGHT)
    return _fn


def basket(fn, cand: pd.DataFrame, start: str):
    br = simulate_basket(cand, start, END, config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, fn)
    return br.returns, (br.weights > 0).sum(axis=1), br.weights


def overlay(raw: pd.Series, spy: pd.Series, dfii10, px, sleeve: str) -> pd.Series:
    core = backtest.compose(raw, spy.reindex(raw.index), use_valuation=True)
    if sleeve == "15% GLD/IEF":
        rising = hedge_overlay.real_yield_rising_mask(dfii10, core.index)
        return hedge_overlay.apply_hedge(core, hedge_overlay.sleeve_returns(px),
                                         hedge_overlay.short_returns(px), rising, True, False)
    hp = pd.read_parquet(os.path.join(str(data.data_root()), "output_html", "img", "a1a4", "_hedge_px.parquet"))
    s = hp[["GLD", "BTAL"]].reindex(core.index).pct_change().fillna(0.0).mean(axis=1)
    return 0.75 * core + 0.25 * s


def metrics(r: pd.Series) -> dict:
    out = {}
    for w, (a, b) in WINDOWS.items():
        m = backtest.compute_metrics(r.loc[a:b].dropna(), config.RF, config.ANN)
        out[f"SR {w}"] = m["sharpe"]
        out[f"Calmar {w}"] = m["cagr"] / abs(m["max_dd"]) if m["max_dd"] < 0 else np.nan
        out[f"MaxDD {w}"] = m["max_dd"]
        out[f"CAGR {w}"] = m["cagr"]
    return out


def main() -> pd.DataFrame:
    pub, members, px, comp = tp.load()
    dfii10 = fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", END)
    spy = px["SPY"].pct_change()
    warm = pd.read_pickle(os.path.join(tp.OUT, "_warmup_members.pkl"))
    members_w = pd.concat([warm.loc[:"2018-12-31"], members.loc["2019-01-01":]]).fillna(False).astype(bool)
    cand = px[[c for c in members_w.columns.union(members.columns) if c in px.columns]]

    pave = px["PAVE"].pct_change().loc[WARM:END]
    names, names_cost, names_held = tp.name_short_basket(px, WARM)
    bot, bot_held, bot_w = basket(target_fn("bottom", 0.5, comp, members_w, px), cand, WARM)
    bot_cost = (bot_w.diff().abs().sum(axis=1) * tp.HEDGE_COST_BP / 1e4).reindex(bot.index).fillna(0)
    shorts = {"short PAVE (beta)": dict(h=pave),
              "short PAVE ($0.25)": dict(h=pave, fixed=tp.FIXED_K),
              "short 14-signal names (beta)": dict(h=names, borrow=tp.NAME_BORROW, unit_cost=names_cost),
              "short 14-signal names ($0.25)": dict(h=names, borrow=tp.NAME_BORROW, unit_cost=names_cost,
                                                    fixed=tp.FIXED_K),
              "short bottom half of book (beta)": dict(h=bot, borrow=tp.NAME_BORROW, unit_cost=bot_cost)}
    short_names = {"short PAVE (beta)": 1.0, "short PAVE ($0.25)": 1.0,
                   "short 14-signal names (beta)": float(names_held.loc["2023-01-01":END].mean()),
                   "short 14-signal names ($0.25)": float(names_held.loc["2023-01-01":END].mean()),
                   "short bottom half of book (beta)": float(bot_held.loc["2023-01-01":END].mean())}

    longs = {"top 40%": ("top", 0.4), "top 50%": ("top", 0.5), "top 60%": ("top", 0.6),
             "top 50% rank-weighted": ("rank", 0.5), "all names (published)": ("top", 1.0)}
    rows = []
    for lname, (kind, q) in longs.items():
        fn_live = target_fn(kind, q, comp, members, px)
        fn_warm = target_fn(kind, q, comp, members_w, px)
        raw, held, _ = basket(fn_live, cand, START)
        raw_w, _, _ = basket(fn_warm, cand, WARM)
        for sleeve in ("15% GLD/IEF", "25% GLD/BTAL"):
            book = overlay(raw, spy, dfii10, px, sleeve)
            book_w = overlay(raw_w, spy, dfii10, px, sleeve)
            base = {"long": lname, "sleeve": sleeve,
                    "long names 2023+": float(held.loc["2023-01-01":END].mean())}
            rows.append({**base, "short": "none", "short names 2023+": 0.0, "short $ per $1": 0.0,
                         **metrics(book)})
            for sname, kw in shorts.items():
                net, beta = tp.hedge(book, beta_src=book_w, **kw)
                rows.append({**base, "short": sname, "short names 2023+": short_names[sname],
                             "short $ per $1": float(beta.loc[START:END].mean()), **metrics(net)})
            print(f"  {lname} | {sleeve}: done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(data.CACHE / "results_robustness.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 260, "display.max_rows", 200)
    res = main()
    cols = ["long", "sleeve", "short", "SR 2023+", "Calmar 2023+", "MaxDD 2023+", "CAGR 2023+",
            "SR 2022+", "Calmar 2022+", "SR full", "Calmar full", "MaxDD full",
            "long names 2023+", "short names 2023+", "short $ per $1"]
    print(res[cols].round(2).to_string(index=False))
