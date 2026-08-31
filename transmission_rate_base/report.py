from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transmission_rate_base import config, gate as gate_mod
from transmission_rate_base import additivity as ad
from transmission_rate_base import backtest as bt
from transmission_rate_base import portfolio as pf
from transmission_rate_base import signal as sig
from transmission_rate_base.data import ferc_form1, utility_map
from transmission_rate_base.data import prices as prices_mod


def _load_ferc(offline: bool, refresh: bool, injected) -> dict:
    if injected is not None:
        return injected
    return {k: ferc_form1.fetch_table(k, refresh=refresh, offline=offline)
            for k in config.FERC_TABLES}


def _quintile_means(neutral: pd.DataFrame, monthly_ret: pd.DataFrame, rebal_dates) -> list[float]:
    sig_by_year = neutral.set_index(["year", "ticker"])["neutral_signal"]
    buckets: dict[int, list[float]] = {q: [] for q in range(config.QUINTILES)}
    for d in rebal_dates:
        try:
            sy = sig_by_year.loc[d.year - 1].dropna().sort_values()
        except KeyError:
            continue
        if len(sy) < config.QUINTILES:
            continue
        span = monthly_ret.index[(monthly_ret.index > d) &
                                 (monthly_ret.index <= d + pd.DateOffset(years=1))]
        cols = [t for t in sy.index if t in monthly_ret.columns]
        if len(span) == 0 or len(cols) < config.QUINTILES:
            continue
        sy = sy[cols]
        labels = pd.qcut(sy.rank(method="first"), config.QUINTILES, labels=False)
        fwd = (1 + monthly_ret.loc[span, cols]).prod() - 1
        for q in range(config.QUINTILES):
            names = sy.index[labels == q]
            if len(names):
                buckets[q].append(float(fwd[names].mean()))
    return [float(np.nanmean(buckets[q])) if buckets[q] else np.nan
            for q in range(config.QUINTILES)]


def _pre_thesis_tstat(neutral: pd.DataFrame, prices, offline: bool) -> dict:
    dr = prices.daily_returns(sorted(neutral["ticker"].unique()),
                              config.PRE_THESIS_START, config.PRE_THESIS_END, offline=offline)
    if dr.empty:
        return {"t_stat": 0.0}
    ew = dr.mean(axis=1)
    w = pf.quintile_ls_weights(neutral, dr, ew)
    mr = (1 + dr).resample("ME").prod() - 1
    strat, _ = bt.run_backtest(w, mr)
    sd = strat.std(ddof=1)
    if strat.empty or not sd or sd == 0:
        return {"t_stat": 0.0}
    return {"t_stat": float(strat.mean() / sd * np.sqrt(len(strat)))}


def run_pipeline(*, start: str = config.PRIMARY_START, end: str = config.PRIMARY_END,
                 offline: bool = False, refresh_ferc: bool = False, pre_thesis: bool = False,
                 out_dir: str | Path = "output_transmission_rate_base",
                 _ferc=None, _prices=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    F = _load_ferc(offline, refresh_ferc, _ferc)
    prices = _prices if _prices is not None else prices_mod

    filer_map = utility_map.resolve_filers(F["utility_xwalk"])
    net_tx = ferc_form1.net_transmission_plant(F["plant_in_service"], F["dep_by_function"],
                                              F["plant_summary"])
    totals = ferc_form1.total_net_utility_plant(F["plant_summary"])
    panel = sig.build_parent_panel(net_tx, totals, filer_map)
    raw = sig.primary_signal(panel)
    neutral = sig.neutralize(raw, panel, sig.load_segment_mix())

    tickers = sorted(neutral["ticker"].unique())
    dr = prices.daily_returns(tickers, start, end, offline=offline)
    ew = dr.mean(axis=1)
    w_ls = pf.quintile_ls_weights(neutral, dr, ew)
    w_tilt = pf.long_tilt_weights(neutral, dr)
    mr = prices.monthly_returns(tickers, start, end, offline=offline)

    strat, metrics = bt.run_backtest(w_ls, mr)
    tilt_strat, tilt_metrics = bt.run_backtest(w_tilt, mr)
    yr = bt.annual_returns(strat)
    rd = pf.rebalance_dates(dr.index, start, end)
    ic = bt.signal_rank_ic(neutral, dr, ew, rd)
    qmeans = _quintile_means(neutral, mr, rd)

    try:
        xlu = prices.daily_returns(["XLU"], start, end, offline=offline)["XLU"]
    except Exception:
        xlu = ew.copy()
    controls = ad.build_controls(dr, ew, xlu, ad.load_style_inputs(), rd)
    add_res = ad.run_additivity(strat, controls)

    pt = _pre_thesis_tstat(neutral, prices, offline) if pre_thesis else {"t_stat": 0.0}

    verdict = gate_mod.evaluate_gate(
        ic={"mean_ic": ic["mean_ic"], "t_stat": ic["t_stat"]},
        spread={"sharpe": metrics["sharpe"], "quintile_means": qmeans,
                "positive_years_frac": metrics["positive_years_frac"]},
        additivity=add_res, pre_thesis=pt)

    neutral.to_csv(out / "signal_panel.csv", index=False)
    w_ls.to_csv(out / "weights.csv")
    strat.rename("strategy").to_csv(out / "pnl.csv")
    yr.rename("annual_return").to_csv(out / "annual_returns.csv")
    ic["ic"].rename("rank_ic").to_csv(out / "ic.csv")
    pd.Series(add_res).to_csv(out / "additivity.csv")
    (out / "verdict.txt").write_text("\n".join(verdict["reasons"]) + "\n")

    return {"verdict": verdict, "metrics": metrics, "tilt_metrics": tilt_metrics,
            "ic": ic, "additivity": add_res, "annual_returns": yr,
            "quintile_means": qmeans}
