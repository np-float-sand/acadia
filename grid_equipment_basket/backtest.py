from __future__ import annotations

"""Metrics, benchmark-relative stats, and full-run orchestration for the
grid-equipment basket. Simple-return convention (see README)."""

import warnings

import numpy as np
import pandas as pd

from grid_equipment_basket import config
from grid_equipment_basket.basket import simulate_basket


def compute_metrics(returns: pd.Series, rf_annual: float = 0.04, ann_factor: int = 252) -> dict:
    r = returns.dropna()
    out: dict = {"n_obs": int(len(r))}
    if len(r) < 2 or r.std() == 0:
        out.update(cagr=np.nan, ann_vol=np.nan, sharpe=np.nan,
                   sortino=np.nan, max_dd=np.nan, hit_rate=np.nan)
        return out
    growth = float((1.0 + r).prod())
    years = len(r) / ann_factor
    cagr = growth ** (1.0 / years) - 1.0
    ann_vol = float(r.std() * np.sqrt(ann_factor))
    excess = r - rf_annual / ann_factor
    sharpe = float(excess.mean() / excess.std() * np.sqrt(ann_factor))
    downside = float(r[r < 0].std() * np.sqrt(ann_factor))
    sortino = float((r.mean() * ann_factor - rf_annual) / downside) if downside > 0 else np.nan
    curve = (1.0 + r).cumprod()
    max_dd = float((curve / curve.cummax() - 1.0).min())
    out.update(
        cagr=round(cagr, 4), ann_vol=round(ann_vol, 4), sharpe=round(sharpe, 3),
        sortino=round(sortino, 3) if sortino == sortino else np.nan,
        max_dd=round(max_dd, 4), hit_rate=round(float((r > 0).mean()), 3),
    )
    return out


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    by_year = (1.0 + r).groupby(r.index.year).prod() - 1.0
    by_year.index = by_year.index.astype(int)
    return by_year.round(4)


def relative_metrics(basket: pd.Series, bench: pd.Series,
                     rf_annual: float = 0.04, ann_factor: int = 252) -> dict:
    df = pd.concat([basket.rename("b"), bench.rename("m")], axis=1).dropna()
    if len(df) < 2:
        return {"corr": np.nan, "tracking_error": np.nan,
                "information_ratio": np.nan, "excess_cagr": np.nan}
    active = df["b"] - df["m"]
    te = float(active.std() * np.sqrt(ann_factor))
    ir = float(active.mean() / active.std() * np.sqrt(ann_factor)) if active.std() > 0 else 0.0
    b_cagr = compute_metrics(df["b"], rf_annual, ann_factor)["cagr"]
    m_cagr = compute_metrics(df["m"], rf_annual, ann_factor)["cagr"]
    excess_cagr = (b_cagr - m_cagr) if (b_cagr == b_cagr and m_cagr == m_cagr) else np.nan
    return {
        "corr": round(float(df["b"].corr(df["m"])), 3),
        "tracking_error": round(te, 4),
        "information_ratio": round(ir, 3) if ir == ir else np.nan,
        "excess_cagr": round(float(excess_cagr), 4) if excess_cagr == excess_cagr else np.nan,
    }


def _default_price_fn(tickers, start, end):
    from grid_equipment_basket.data.prices import fetch_prices
    return fetch_prices(tickers, start, end)


def run(start: str, end: str, universe=None, benchmarks=None,
        target_fn=None, price_fn=None) -> dict:
    universe = list(universe or config.UNIVERSE)
    benchmarks = list(benchmarks or config.BENCHMARKS)
    price_fn = price_fn or _default_price_fn

    prices = price_fn(sorted(set(universe + benchmarks)), start, end)
    uni_cols = [t for t in universe if t in prices.columns]
    missing_uni = [t for t in universe if t not in prices.columns]
    if missing_uni:
        warnings.warn(
            f"universe names absent from price data, excluded from basket: {missing_uni}",
            RuntimeWarning, stacklevel=2,
        )
    br = simulate_basket(
        prices[uni_cols], start, end,
        config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, target_fn,
    )
    bench_cols = [t for t in benchmarks if t in prices.columns]
    bench_ret = prices[bench_cols].loc[start:end].pct_change().dropna(how="all")

    res: dict = {
        "start": start, "end": end, "universe": universe,
        "n_names_end": int(br.weights.iloc[-1].gt(0).sum()) if not br.weights.empty else 0,
        "basket": compute_metrics(br.returns, config.RISK_FREE_RATE, config.ANN_FACTOR),
        "basket_calendar": calendar_year_returns(br.returns),
        "basket_returns": br.returns,
        "benchmark_returns": bench_ret,
        "benchmarks": {},
    }
    for b in bench_ret.columns:
        col = bench_ret[b].dropna()
        res["benchmarks"][b] = {
            "metrics": compute_metrics(col, config.RISK_FREE_RATE, config.ANN_FACTOR),
            "calendar": calendar_year_returns(col),
            "relative": relative_metrics(br.returns, col, config.RISK_FREE_RATE, config.ANN_FACTOR),
        }
    return res


def results_table(results: dict) -> str:
    rows = [("BASKET", results["basket"])]
    rows += [(b, results["benchmarks"][b]["metrics"]) for b in results["benchmarks"]]
    lines = [
        f"Window {results['start']} -> {results['end']}   "
        f"(basket names at end: {results['n_names_end']})",
        f"{'':<10}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}",
    ]
    for name, m in rows:
        lines.append(
            f"{name:<10}{_p(m['cagr']):>9}{_p(m['ann_vol']):>9}"
            f"{_f(m['sharpe']):>9}{_f(m['sortino']):>9}{_p(m['max_dd']):>9}"
        )
    lines.append("")
    lines.append(f"{'vs':<10}{'exCAGR':>9}{'corr':>9}{'TE':>9}{'IR':>9}")
    for b in results["benchmarks"]:
        rel = results["benchmarks"][b]["relative"]
        lines.append(
            f"{b:<10}{_p(rel['excess_cagr']):>9}{_f(rel['corr']):>9}"
            f"{_p(rel['tracking_error']):>9}{_f(rel['information_ratio']):>9}"
        )
    return "\n".join(lines)


def _p(x) -> str:
    return "n/a" if x != x else f"{x * 100:.1f}%"


def _f(x) -> str:
    return "n/a" if x != x else f"{x:.2f}"


def plot(results: dict, save_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basket = results["basket_returns"]
    bench = results["benchmark_returns"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    bcurve = (1 + basket).cumprod()
    ax1.plot(bcurve.index, bcurve.values, lw=2.0, color="#1a3a5c", label="Basket")
    for c in bench.columns:
        cc = (1 + bench[c].dropna()).cumprod()
        ax1.plot(cc.index, cc.values, lw=1.0, alpha=0.8, label=c)
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of 1 (log)")
    ax1.legend(fontsize=8, ncol=3)
    ax1.set_title("Grid Equipment Suppliers Basket vs Benchmarks")
    dd = bcurve / bcurve.cummax() - 1.0
    ax2.fill_between(dd.index, dd.values, 0, color="#8c2d04", alpha=0.5)
    ax2.set_ylabel("Basket drawdown")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _load_vc_inputs(fund_df, backlog_df):
    if fund_df is None:
        from grid_equipment_basket.margin_data import combine_fundamentals, fetch_fundamentals
        fund_df = combine_fundamentals({t: fetch_fundamentals(t) for t in config.UNIVERSE})
    if backlog_df is None:
        from grid_equipment_basket.backlog_data import load_backlog_csv
        backlog_df = load_backlog_csv()
    return fund_df, backlog_df


def value_chain_report(start: str, end: str, price_fn=None, drop_winners: bool = False,
                       fund_df=None, backlog_df=None) -> dict:
    from grid_equipment_basket import hedges, value_chain   # lazy: hedges imports backtest

    price_fn = price_fn or _default_price_fn
    fund_df, backlog_df = _load_vc_inputs(fund_df, backlog_df)

    makers = [m for m in config.BUCKET_MAKERS if not (drop_winners and m in config.VC_DROP_WINNERS)]
    universe = makers + config.BUCKET_CONTRACTORS
    tickers = sorted(set(universe + config.BENCHMARKS + [config.COND_SHORT_TICKER]))
    prices = price_fn(tickers, start, end)
    uni_cols = [t for t in universe if t in prices.columns]
    missing = [t for t in universe if t not in prices.columns]
    if missing:
        warnings.warn(
            f"universe names absent from price data, excluded: {missing}",
            RuntimeWarning, stacklevel=2,
        )

    rf, af = config.RISK_FREE_RATE, config.ANN_FACTOR

    ew = simulate_basket(prices[uni_cols], start, end,
                         config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT)
    def _tilt_fn(available, asof):
        return value_chain.value_chain_tilt_targets(
            available, asof, fund_df, backlog_df,
            cap=config.MAX_SINGLE_NAME_WEIGHT,
            base_maker=config.VC_BASE_MAKER, base_contractor=config.VC_BASE_CONTRACTOR,
            within_top=config.VC_WITHIN_TOP, within_bottom=config.VC_WITHIN_BOTTOM,
        )
    tilt = simulate_basket(prices[uni_cols], start, end,
                           config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, _tilt_fn)

    pair_prices = prices[uni_cols]
    pair = hedges.simulate_pair(pair_prices, start, end, fund_df, backlog_df,
                                config.REBALANCE_LAG_DAYS)

    if config.COND_SHORT_TICKER in prices.columns:
        qqq = prices[config.COND_SHORT_TICKER].pct_change().dropna()
    else:
        warnings.warn(
            f"conditional-short ticker {config.COND_SHORT_TICKER!r} absent from price data; "
            "conditional-short overlay is inert",
            RuntimeWarning, stacklevel=2,
        )
        qqq = pd.Series(dtype=float)
    ew_curve_prices = (1.0 + ew.returns).cumprod()
    mask = hedges.conditional_short_mask(
        ew_curve_prices, config.COND_SHORT_MA_DAYS, config.COND_SHORT_VOL_DAYS, config.COND_SHORT_VOL_REF_DAYS)

    over_pair = hedges.pair_overlay(ew.returns, pair, config.PAIR_OVERLAY_WEIGHT)
    over_cond = hedges.conditional_short_overlay(ew.returns, qqq, mask, config.COND_SHORT_WEIGHT)
    k = hedges.risk_match_weight(ew.returns, pair, over_cond)
    over_pair_rm = hedges.pair_overlay(ew.returns, pair, k)

    # The Gate-2 drawdown episode is pinned to a hard-coded 2024-H2 window (spec §7.3);
    # a window that does not span it (e.g. --prior-regime 2020-2022) has no episode, so
    # Gate 2 is not applicable there. Gate 1 and the standalone rows still compute.
    try:
        peak, trough = hedges.find_drawdown_episode(
            ew.returns, config.DRAWDOWN_PEAK_WINDOW, config.DRAWDOWN_TROUGH_END)
    except ValueError:
        peak = trough = None

    def _epi_dd(r):
        return float("nan") if peak is None else hedges.episode_drawdown(r, peak, trough)

    bench_ret = prices[[b for b in config.BENCHMARKS if b in prices.columns]].loc[start:end].pct_change().dropna(how="all")

    def M(r):
        return compute_metrics(r, rf, af)

    cond_sleeve = (-config.COND_SHORT_WEIGHT
                   * qqq.where(mask.reindex(qqq.index).fillna(False), 0.0)).dropna()
    # A conditional short that never engages (all-zero / zero-variance sleeve, or no
    # QQQ at all) has zero carry, not undefined — compute_metrics returns NaN cagr on
    # a zero-variance series, which would auto-fail Gate 2's carry leg.
    cond_carry = (hedges.annualized_carry(cond_sleeve)
                  if len(cond_sleeve) >= 2 and cond_sleeve.std() > 0 else 0.0)

    report = {
        "start": start, "end": end, "drop_winners": drop_winners,
        "makers": makers,
        "equal_weight": M(ew.returns),
        "value_chain_tilt": M(tilt.returns),
        "benchmarks": {
            b: {"metrics": M(bench_ret[b].dropna()),
                "relative_to_tilt": relative_metrics(tilt.returns, bench_ret[b].dropna(), rf, af)}
            for b in bench_ret.columns
        },
        "pair_standalone": {**M(pair), "carry": hedges.annualized_carry(pair)},
        "overlays": {
            "pair_30": {**M(over_pair), "episode_dd": _epi_dd(over_pair)},
            "cond_short": {**M(over_cond), "episode_dd": _epi_dd(over_cond),
                           "carry": cond_carry},
            "pair_risk_matched": {**M(over_pair_rm), "episode_dd": _epi_dd(over_pair_rm),
                                  "match_weight": float(k)},
        },
        "episode": {"peak": peak, "trough": trough,
                    "equal_weight_dd": _epi_dd(ew.returns)},
    }
    g1 = {"tilt_sharpe": report["value_chain_tilt"]["sharpe"], "ew_sharpe": report["equal_weight"]["sharpe"],
          "tilt_cagr": report["value_chain_tilt"]["cagr"], "ew_cagr": report["equal_weight"]["cagr"]}
    g1["passed"] = bool(g1["tilt_sharpe"] > g1["ew_sharpe"] and g1["tilt_cagr"] > g1["ew_cagr"])
    g2 = {"pair_episode_dd": report["overlays"]["pair_30"]["episode_dd"],
          "cond_episode_dd": report["overlays"]["cond_short"]["episode_dd"],
          "pair_carry": report["pair_standalone"]["carry"],
          "cond_carry": report["overlays"]["cond_short"]["carry"],
          "applicable": peak is not None}
    g2["passed"] = bool(peak is not None
                        and g2["pair_episode_dd"] >= g2["cond_episode_dd"]
                        and g2["pair_carry"] >= g2["cond_carry"])
    report["gate1"], report["gate2"] = g1, g2
    return report


def coverage_report(start: str, end: str, price_fn=None, fund_df=None, backlog_df=None) -> dict:
    """Coverage-alone tilt (handoff 2026-09-01 §8.3(a)) -- equal-weight vs
    ``basket.coverage_tilt_targets``, same Gate 1 criterion as
    ``value_chain_report`` (tilt beats equal-weight on Sharpe AND CAGR)."""
    from grid_equipment_basket import basket as bk

    price_fn = price_fn or _default_price_fn
    fund_df, backlog_df = _load_vc_inputs(fund_df, backlog_df)
    rf, af = config.RISK_FREE_RATE, config.ANN_FACTOR

    prices = price_fn(config.UNIVERSE, start, end)
    uni_cols = [t for t in config.UNIVERSE if t in prices.columns]

    ew = simulate_basket(prices[uni_cols], start, end,
                         config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT)

    def _tilt_fn(available, asof):
        return bk.coverage_tilt_targets(available, asof, fund_df, backlog_df,
                                        cap=config.MAX_SINGLE_NAME_WEIGHT)
    tilt = simulate_basket(prices[uni_cols], start, end,
                           config.REBALANCE_LAG_DAYS, config.MAX_SINGLE_NAME_WEIGHT, _tilt_fn)

    equal_weight = compute_metrics(ew.returns, rf, af)
    coverage_tilt = compute_metrics(tilt.returns, rf, af)

    gate = {"tilt_sharpe": coverage_tilt["sharpe"], "ew_sharpe": equal_weight["sharpe"],
           "tilt_cagr": coverage_tilt["cagr"], "ew_cagr": equal_weight["cagr"]}
    gate["passed"] = bool(gate["tilt_sharpe"] > gate["ew_sharpe"] and gate["tilt_cagr"] > gate["ew_cagr"])

    return {"start": start, "end": end,
           "equal_weight": equal_weight, "coverage_tilt": coverage_tilt, "gate": gate}


def value_chain_table(report: dict) -> str:
    L = [f"Value-chain reframe  {report['start']} -> {report['end']}"
         + ("  [drop-winners]" if report["drop_winners"] else ""),
         f"  makers: {', '.join(report['makers'])}",
         f"{'':<22}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>9}"]
    def line(name, m, extra=""):
        return f"{name:<22}{_p(m['cagr']):>9}{_p(m['ann_vol']):>9}{_f(m['sharpe']):>9}{_p(m['max_dd']):>9}{extra}"
    L.append(line("equal-weight", report["equal_weight"]))
    L.append(line("value-chain tilt", report["value_chain_tilt"]))
    for b, blk in report["benchmarks"].items():
        L.append(line(b, blk["metrics"]))
    L.append("")
    L.append(line("pair (standalone)", report["pair_standalone"],
                  f"  carry {_p(report['pair_standalone']['carry'])}"))
    for k, m in report["overlays"].items():
        L.append(line(f"EW + {k}", m, f"  episodeDD {_p(m['episode_dd'])}"))
    g1, g2 = report["gate1"], report["gate2"]
    L.append("")
    L.append(f"GATE 1 (tilt vs equal-weight):  Sharpe {_f(g1['tilt_sharpe'])} vs {_f(g1['ew_sharpe'])} | "
             f"CAGR {_p(g1['tilt_cagr'])} vs {_p(g1['ew_cagr'])}  ->  {'PASS' if g1['passed'] else 'FAIL'}")
    if report["episode"]["peak"] is None:
        L.append("GATE 2: n/a (window has no 2024-H2 drawdown episode)")
    else:
        L.append(f"GATE 2 (pair vs conditional short):  episodeDD {_p(g2['pair_episode_dd'])} vs {_p(g2['cond_episode_dd'])} | "
                 f"carry {_p(g2['pair_carry'])} vs {_p(g2['cond_carry'])}  ->  {'PASS' if g2['passed'] else 'FAIL'}")
    return "\n".join(L)
