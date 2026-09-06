"""Backtest comparison harness for the electrification strategy (spec 6).

Builds a {marquee, frozen, thematic} x {plain, +val, +val+sleeve, +val+sleeve+short}
grid, scores each cell on the pre-registered metrics, and names a pitch winner by
the spec 6.3 rule. Simple-return convention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grid_equipment_basket.backtest import compute_metrics
from grid_equipment_basket.basket import simulate_basket
from grid_equipment_basket.overlay import vol_target_scalar
from grid_equipment_basket import hedges as _geh

from electrification_strategy import config, hedge_overlay, universe, valuation_overlay

STACKS = [
    ("plain", False, False, False),
    ("+val", True, False, False),
    ("+val+sleeve", True, True, False),
    ("+val+sleeve+short", True, True, True),
]


def basket_returns(cand_prices, construction, start, end):
    """Raw equal-weight basket daily return for one universe construction (before
    the vol-target and valuation layers)."""
    br = simulate_basket(cand_prices, start, end, config.REBALANCE_LAG_DAYS,
                         config.MAX_SINGLE_NAME_WEIGHT, universe.target_fn(construction, cand_prices))
    return br.returns


def compose(base_r, spy_ret, use_valuation, val_scale=1.0):
    """base_r -> vol-target -> optional valuation multiplier, slack at rf."""
    if base_r.empty:
        return base_r
    vscal = vol_target_scalar(base_r, config.VOL_LOOKBACK, config.VOL_TARGET,
                              config.VOL_MAX_LEVERAGE).reindex(base_r.index).fillna(1.0)
    if use_valuation:
        level = (1.0 + base_r).cumprod()
        vmult = valuation_overlay.extension_multiplier(
            level, spy_ret.reindex(base_r.index), scale=val_scale).reindex(base_r.index).fillna(1.0)
    else:
        vmult = pd.Series(1.0, index=base_r.index)
    exposure = (vmult * vscal).clip(upper=config.VOL_MAX_LEVERAGE)
    return exposure * base_r.fillna(0.0) + (1.0 - exposure) * config.RF / config.ANN


def core_return(cand_prices, construction, start, end, spy_ret, use_valuation, val_scale=1.0):
    return compose(basket_returns(cand_prices, construction, start, end),
                   spy_ret, use_valuation, val_scale)


def episode_drawdowns(returns, episodes):
    out = {}
    for name, (peak_win, trough_end) in episodes.items():
        try:
            peak, trough = _geh.find_drawdown_episode(returns, peak_win, trough_end)
            out[name] = float(_geh.episode_drawdown(returns, peak, trough))
        except (ValueError, KeyError, IndexError):
            out[name] = float("nan")
    return out


def feared_pnl(cell_r, plain_r, tan_ret):
    cm = (1.0 + cell_r).resample("ME").prod() - 1.0
    pm = (1.0 + plain_r).resample("ME").prod() - 1.0
    if tan_ret is None or len(tan_ret) == 0:
        return float("nan")
    tm = (1.0 + tan_ret.reindex(cell_r.index).fillna(0.0)).resample("ME").prod() - 1.0
    bad = (pm < 0) & (tm.reindex(pm.index) > 0)
    return float((cm.reindex(pm.index)[bad] - pm[bad]).sum())


def calm_drag(cell_r, plain_r, windows):
    out = {}
    for name, (a, b) in windows.items():
        pc = compute_metrics(plain_r.loc[a:b], config.RF, config.ANN)["cagr"]
        cc = compute_metrics(cell_r.loc[a:b], config.RF, config.ANN)["cagr"]
        out[name] = float(pc - cc) if (pc == pc and cc == cc) else float("nan")
    return out


def _beta(r, mkt):
    d = pd.concat([r.rename("r"), mkt.rename("m")], axis=1, sort=False).dropna()
    if len(d) < 2 or d["m"].var() == 0:
        return float("nan")
    return float(d["r"].cov(d["m"]) / d["m"].var())


def _corr(r, other):
    d = pd.concat([r.rename("r"), other.rename("o")], axis=1, sort=False).dropna()
    return float(d["r"].corr(d["o"])) if len(d) > 2 else float("nan")


def _effective_n(weights_row):
    w = weights_row[weights_row > 0].astype(float)
    return float(1.0 / (w ** 2).sum()) if len(w) else 0.0


# -- comparison grid / reporting -------------------------------------------
def run_comparison(start, end, prices, dfii10, constructions=("marquee", "frozen", "thematic")):
    spy_ret = prices["SPY"].pct_change()
    tan_ret = prices[config.FEARED_PROXY].pct_change() if config.FEARED_PROXY in prices.columns else None
    volt_ret = prices["VOLT"].pct_change() if "VOLT" in prices.columns else pd.Series(dtype=float)
    sleeve_r = hedge_overlay.sleeve_returns(prices)
    short_r = hedge_overlay.short_returns(prices)

    cand = sorted(set(config.MARQUEE_UNIVERSE) | set(universe.load_seed().index))
    cand_prices = prices[[c for c in cand if c in prices.columns]]

    cells, plain_books, eff_n = {}, {}, {}
    for con in constructions:
        br = simulate_basket(cand_prices, start, end, config.REBALANCE_LAG_DAYS,
                             config.MAX_SINGLE_NAME_WEIGHT, universe.target_fn(con, cand_prices))
        base_r = br.returns
        eff_n[con] = _effective_n(br.weights.iloc[-1]) if not br.weights.empty else 0.0

        plain_r = compose(base_r, spy_ret, use_valuation=False)
        plain_books[con] = plain_r
        plain_cagr = compute_metrics(plain_r, config.RF, config.ANN)["cagr"]
        mask = hedge_overlay.real_yield_rising_mask(dfii10, plain_r.index)

        for label, use_val, use_slv, use_sht in STACKS:
            core_r = compose(base_r, spy_ret, use_valuation=use_val)
            cell_r = (core_r if not (use_slv or use_sht)
                      else hedge_overlay.apply_hedge(core_r, sleeve_r, short_r, mask, use_slv, use_sht))
            m = compute_metrics(cell_r, config.RF, config.ANN)
            cells[(con, label)] = {
                "metrics": m,
                "beta": _beta(cell_r, spy_ret),
                "corr_volt": _corr(cell_r, volt_ret),
                "episode_dd": episode_drawdowns(cell_r, config.EPISODES),
                "feared_pnl": feared_pnl(cell_r, plain_r, tan_ret),
                "calm_drag": calm_drag(cell_r, plain_r, config.CALM_WINDOWS),
                "subwindow_sharpe": {
                    k: compute_metrics(cell_r.loc[a:b], config.RF, config.ANN)["sharpe"]
                    for k, (a, b) in config.SUBWINDOWS.items()
                },
                "drag_vs_plain": float(plain_cagr - m["cagr"]) if m["cagr"] == m["cagr"] else float("nan"),
                "returns": cell_r,
            }

    benchmarks = {
        b: compute_metrics(prices[b].pct_change().loc[start:end].dropna(), config.RF, config.ANN)
        for b in config.BENCHMARKS if b in prices.columns
    }
    plateau = []
    for sc in config.VAL_PLATEAU_SCALES:
        r = core_return(cand_prices, "frozen", start, end, spy_ret, use_valuation=True, val_scale=sc)
        pm = compute_metrics(r, config.RF, config.ANN)
        plateau.append({"scale": sc, "sharpe": pm["sharpe"], "max_dd": pm["max_dd"], "cagr": pm["cagr"]})

    return {"start": start, "end": end, "cells": cells, "eff_n": eff_n,
            "benchmarks": benchmarks, "plateau": plateau, "plain_books": plain_books}


def pick_winner(result):
    passing = []
    for key, c in result["cells"].items():
        m = c["metrics"]
        subs = [s for s in c["subwindow_sharpe"].values() if s == s]
        ok = (m["max_dd"] == m["max_dd"] and m["max_dd"] >= config.WINNER_MAXDD_MAX
              and subs and all(s >= config.WINNER_SUBWINDOW_SHARPE_MIN for s in subs)
              and c["feared_pnl"] == c["feared_pnl"] and c["feared_pnl"] >= config.WINNER_FEARED_PNL_MIN
              and c["drag_vs_plain"] == c["drag_vs_plain"] and c["drag_vs_plain"] <= config.WINNER_DRAG_MAX)
        if ok:
            passing.append((key, m["sharpe"]))
    passing.sort(key=lambda kv: (kv[1] if kv[1] == kv[1] else -1e9), reverse=True)
    return {"winner": passing[0][0] if passing else None,
            "runner_up": passing[1][0] if len(passing) > 1 else None,
            "n_passing": len(passing)}


def _fmt_pct(x):
    return "   n/a" if x != x else f"{x * 100:6.1f}%"


def comparison_table(result):
    lines = [f"ELECTRIFICATION STRATEGY -- {result['start']} -> {result['end']}", ""]
    lines.append(f"{'cell':<28}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'beta':>7}"
                 f"{'drag':>8}{'feared':>8}{'corrVOLT':>9}")
    for (con, label), c in result["cells"].items():
        m = c["metrics"]
        sharpe = m["sharpe"] if m["sharpe"] == m["sharpe"] else float("nan")
        lines.append(
            f"{con + ' ' + label:<28}{_fmt_pct(m['cagr'])}{_fmt_pct(m['ann_vol'])}"
            f"{sharpe:>8.2f}{_fmt_pct(m['max_dd'])}{c['beta']:>7.2f}{_fmt_pct(c['drag_vs_plain'])}"
            f"{_fmt_pct(c['feared_pnl'])}{c['corr_volt']:>9.2f}"
        )
    lines += ["", "effective #names: " + "  ".join(f"{k} {v:.1f}" for k, v in result["eff_n"].items()), ""]
    lines.append("benchmarks (CAGR / Sharpe / MaxDD):")
    for b, m in result["benchmarks"].items():
        lines.append(f"  {b:<6}{_fmt_pct(m['cagr'])}  {m['sharpe']:>5.2f}  {_fmt_pct(m['max_dd'])}")
    lines += ["", "valuation plateau (frozen +val, thresholds x scale):"]
    for row in result["plateau"]:
        lines.append(f"  x{row['scale']:<4} Sharpe {row['sharpe']:>5.2f}  "
                     f"MaxDD {_fmt_pct(row['max_dd'])}  CAGR {_fmt_pct(row['cagr'])}")
    return "\n".join(lines)


def write_outputs(result, winner, out_dir, make_plot=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (con, label), c in result["cells"].items():
        m = c["metrics"]
        rows.append({"name": f"{con} {label}", "cagr": m["cagr"], "ann_vol": m["ann_vol"],
                     "sharpe": m["sharpe"], "max_dd": m["max_dd"], "beta": c["beta"],
                     "drag_vs_plain": c["drag_vs_plain"], "feared_pnl": c["feared_pnl"],
                     "corr_volt": c["corr_volt"]})
    for b, m in result["benchmarks"].items():
        rows.append({"name": f"BENCH {b}", "cagr": m["cagr"], "ann_vol": m["ann_vol"],
                     "sharpe": m["sharpe"], "max_dd": m["max_dd"]})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    ep = pd.DataFrame({f"{con} {label}": c["episode_dd"]
                       for (con, label), c in result["cells"].items()}).T
    ep.to_csv(out / "episode_drawdowns.csv")

    (out / "plateau.json").write_text(json.dumps(
        {"winner": str(winner["winner"]), "runner_up": str(winner["runner_up"]),
         "n_passing": winner["n_passing"], "plateau": result["plateau"]}, indent=2))

    rets = pd.DataFrame({f"{con}|{label}": c["returns"]
                         for (con, label), c in result["cells"].items()
                         if c["returns"] is not None})
    rets.to_csv(out / "returns.csv")

    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})
        wkey = winner["winner"] or ("marquee", "plain")
        series = {
            "winner: " + " ".join(wkey): result["cells"][wkey]["returns"],
            "marquee plain": result["cells"][("marquee", "plain")]["returns"],
        }
        for name, r in series.items():
            if r is None:
                continue
            cc = (1 + r).cumprod()
            ax1.plot(cc.index, cc.values, lw=1.6, label=name)
        ax1.set_yscale("log")
        ax1.legend(fontsize=8)
        ax1.set_ylabel("growth of 1 (log)")
        wr = result["cells"][wkey]["returns"]
        if wr is not None:
            cc = (1 + wr).cumprod()
            dd = cc / cc.cummax() - 1.0
            ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="#8c2d04")
        ax2.set_ylabel("winner drawdown")
        fig.tight_layout()
        fig.savefig(out / "performance.png", dpi=120)
        plt.close(fig)
