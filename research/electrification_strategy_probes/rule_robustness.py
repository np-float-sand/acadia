"""Is the investor strategy's rule stack sensible, or data-mined? Three checks on `investor_book`.

    python research/electrification_strategy_probes/rule_robustness.py

  3. Parameter plateaus -- move one setting at a time around the chosen value.
  4. Ablation ladder    -- start from the plain equal-weight screen, add one rule at a time.
  5. Deflated Sharpe    -- the headline Sharpe discounted for the number of variants tried.

Every run charges 10bp on every trade and borrow on the short (investor_book.build). Writes
output_html/img/tophalf/rule_robustness.json and prints the tables.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_tophalf_proposal as t  # noqa: E402
import investor_book as ib  # noqa: E402

WINS = {"full": (t.START, t.END), "2019-22": ("2019-01-01", "2022-12-31"), "2023+": ("2023-01-01", t.END)}
EULER = 0.5772156649


def summarise(r: pd.Series) -> dict:
    out = {}
    for w, (a, e) in WINS.items():
        s = t.a1.stats(r.loc[a:e])
        out[w] = {"sharpe": round(s["sharpe"], 3), "calmar": round(s["calmar"], 3),
                  "maxdd": round(s["maxdd"], 4), "cagr": round(s["cagr"], 4)}
    return out


def row(name, s):
    return {"variant": name, "SR full": s["full"]["sharpe"], "SR 19-22": s["2019-22"]["sharpe"],
            "SR 23+": s["2023+"]["sharpe"], "Calmar full": s["full"]["calmar"],
            "maxDD full": f"{s['full']['maxdd']:.1%}", "return full": f"{s['full']['cagr']:.1%}"}


def psr(r: pd.Series, sr_star_daily: float) -> float:
    """Probabilistic Sharpe ratio: P(true daily Sharpe > sr_star) given skew and kurtosis."""
    ex = (r - t.config.RF / t.config.ANN).dropna()
    sr = ex.mean() / ex.std()
    g3, g4 = float(ex.skew()), float(ex.kurt()) + 3.0
    return float(norm.cdf((sr - sr_star_daily) * np.sqrt(len(ex) - 1)
                          / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def expected_max_sharpe(n_trials: int, var_sr_daily: float) -> float:
    """Expected maximum daily Sharpe among n_trials skill-less strategies (Bailey & Lopez de Prado)."""
    return float(np.sqrt(var_sr_daily) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                                          + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def main():
    pub, members, px, comp = t.load()
    spy = px["SPY"].loc[t.START:t.END].pct_change()
    dfii = t.fetch_series(t.config.HEDGE_SHORT_SIGNAL, "2003-01-01", t.END)
    ry = t.fetch_series(t.RATE_SIGNAL, "2003-01-01", t.END).dropna()
    cash = t.fetch_series(t.CASH_SERIES, "2003-01-01", t.END).dropna()
    g = t.supplier_growth(members)
    off = ry.clip(lower=10.0)                       # rule 11 never fires

    def run(**kw):
        rate = kw.pop("ry", ry)
        params = dict(growth=g)
        params.update(kw)
        return ib.build(members, comp, px, spy, dfii, rate, cash, **params)["returns"]

    results, series = {"plateaus": {}, "ladder": {}}, {}
    base = run()
    series["final"] = base
    results["final"] = summarise(base)

    # ---- 3. plateaus -------------------------------------------------------------------------
    grid = {"band": [0.10, 0.15, 0.20, 0.25, 0.30],
            "long_hold": [None, 0.55, 0.60, 0.65, 0.70, 0.75],
            "short_k": [0.15, 0.20, 0.25, 0.30, 0.35],
            "short_q": [0.15, 0.20, 0.25],
            "short_freq": ["M", "2M", "Q", "S"],
            "ry_thr": [-0.5, -0.25, 0.0, 0.25, 0.5],
            "g_thr": [-0.02, -0.01, 0.0, 0.01, 0.02]}
    for param, vals in grid.items():
        results["plateaus"][param] = {}
        for v in vals:
            r = run(**{param: v})
            series[f"{param}={v}"] = r
            results["plateaus"][param][str(v)] = summarise(r)

    # ---- 4. ablation ladder ------------------------------------------------------------------
    ladder = [
        ("L0 rules 1-5, equal weight, no overlay, no short", dict(rank=False, overlay=False, short=None, ry=off, growth=None)),
        ("L1 + risk overlay (rules 7-9, band, monthly sleeve)", dict(rank=False, short=None, ry=off, growth=None)),
        ("L2 + ranking, top half (rule 6)", dict(long_hold=None, short=None, ry=off, growth=None)),
        ("L3 + hold buffer 65%", dict(short=None, ry=off, growth=None)),
        ("L4 + short $0.25, quarterly (rule 10)", dict(ry=off, growth=None)),
        ("L5 + rule 11, real yield only", dict(growth=None)),
        ("L6 + rule 11, both conditions (final)", dict()),
    ]
    for name, kw in ladder:
        r = run(**kw)
        series[name] = r
        results["ladder"][name] = summarise(r)

    # ---- 5. deflated Sharpe ------------------------------------------------------------------
    srs = pd.Series({k: v.loc[t.START:t.END].pipe(lambda x: (x - t.config.RF / t.config.ANN).mean()
                                                   / (x - t.config.RF / t.config.ANN).std())
                     for k, v in series.items()})
    var_daily = float(srs.var())
    grid_sr_daily = 0.82 / np.sqrt(t.config.ANN)
    dsr = {}
    for n in (20, 50, 100, 300, 1000):
        star = expected_max_sharpe(n, var_daily)
        dsr[n] = {"sr_star_annual": round(star * np.sqrt(t.config.ANN), 3),
                  "dsr_full": round(psr(base.loc[t.START:t.END], star), 4),
                  "dsr_2023": round(psr(base.loc["2023-01-01":t.END], star), 4)}
    results["deflated"] = {"n_variants_measured": int(len(srs)),
                           "sd_annual_sharpe_across_variants": round(float(srs.std() * np.sqrt(t.config.ANN)), 3),
                           "psr_vs_zero_full": round(psr(base.loc[t.START:t.END], 0.0), 4),
                           "psr_vs_GRID_full": round(psr(base.loc[t.START:t.END], grid_sr_daily), 4),
                           "psr_vs_GRID_2023": round(psr(base.loc["2023-01-01":t.END],
                                                         0.93 / np.sqrt(t.config.ANN)), 4),
                           "by_trials": dsr}

    json.dump(results, open(os.path.join(t.OUT, "rule_robustness.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("FINAL", results["final"])
    for param, d in results["plateaus"].items():
        print(f"\n-- plateau: {param}")
        print(pd.DataFrame([row(f"{param}={k}", v) for k, v in d.items()]).to_string(index=False))
    print("\n-- ablation ladder")
    print(pd.DataFrame([row(k, v) for k, v in results["ladder"].items()]).to_string(index=False))
    print("\n-- deflated Sharpe")
    print(json.dumps(results["deflated"], indent=1))


if __name__ == "__main__":
    main()
