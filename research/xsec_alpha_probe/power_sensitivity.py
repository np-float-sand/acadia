"""Is sensitivity to power prices itself a reason to short? (2026-09-25)

    python -m research.xsec_alpha_probe.power_sensitivity

Hypothesis (user): the winners are resilient because they have buffered themselves against short-term
power-price moves, so HIGH |sensitivity| (either sign) marks names to avoid / short.

Pre-set before running:
  insulation = -mean(|beta_energy|, |beta_congestion|, |beta_scarcity|)   (raw rolling betas, market-
               controlled, from signals.power_signals; each |beta| winsorized+z-scored first)
  1. rank-IC of `insulation` at 1m / 3m, NW t, windows full / 2019-22 / 2023-26.
     Bar: |t| > 1.5 in 2023-26, positive sign (= insulated names outperform).
  2. descriptive: mean |beta| z of the TOP_HALF book vs the S_all14 short basket vs the universe.
  3. short sleeve: EW most-sensitive quintile, 0.25x on TOP_HALF (10bp + 1%/yr borrow), vs S_all14.
  4. ablation: the 14-signal short rebuilt WITHOUT the 6 power signals (S_8), 0.25x and matched.
Monthly, pre-overlay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.xsec_alpha_probe import composites as cp, data, evaluate as ev, run, signals as sg, tilt


def main() -> None:
    px_d = data.prices(); univ = data.universe(); px_m = data.monthly_prices(px_d)
    ret_m = px_m.pct_change(fill_method=None)
    fwd1, fwd3 = ret_m.shift(-1), px_m.shift(-3) / px_m - 1
    bm = data.monthly_prices(data.benchmarks())
    sigs = {}
    sigs.update(sg.fundamental_signals()); sigs.update(sg.price_signals(px_m))
    sigs.update(sg.power_signals(px_m, bm["SPY"]))
    absz = {b: ev.prep(sigs[b].abs().reindex(index=univ.index, columns=univ.columns).where(univ))
            for b in cp.POWER_BETAS}
    sens = pd.concat(list(absz.values()), keys=range(3)).groupby(level=1).mean()   # high = sensitive
    insul = -sens

    print("== 1. rank-IC of insulation (-|power beta|), + = insulated names outperform")
    for h, fwd, lags in (("1m", fwd1, 1), ("3m", fwd3, 2)):
        ic = ev.ic_series(insul, fwd.reindex(columns=insul.columns))
        ts = {w: ev.nw_tstat(run._slice(ic, w), lags=lags) for w in run.WINDOWS}
        print(f"  {h}: IC {run._slice(ic, 'full').mean():+.3f} | t full {ts['full']:+.2f} / "
              f"19-22 {ts['2019-22']:+.2f} / 23-26 {ts['2023-26']:+.2f} -> "
              f"{'PASS' if ts['2023-26'] > 1.5 else 'FAIL'}")

    print("\n== 2. who is power-sensitive? mean |beta| z (0 = universe average)")
    book = tilt.book_membership(univ.index)
    comp_long = tilt.composite(univ, book)
    comp14 = cp.composite(sigs, list(sigs), univ)
    rows = {"TOP_HALF (long book)": [], "rest of screen_auto book": [], "S_all14 short basket": [],
            "universe": []}
    for dt in univ.index[(univ.index >= run.START) & (univ.index <= run.END)]:
        s = sens.loc[dt].dropna()
        mem = [t for t in book.columns[book.loc[dt]] if t in s.index]
        if len(mem) < 3:
            continue
        z = comp_long.loc[dt].reindex(mem).fillna(0).sort_values(ascending=False)
        top = list(z.index[: max(1, len(z) // 2)])
        c = comp14.loc[dt].where(univ.loc[dt]).dropna()
        bot = [t for t in c[c <= c.quantile(0.2)].index if t in s.index]
        rows["TOP_HALF (long book)"].append(s.reindex(top).mean())
        rows["rest of screen_auto book"].append(s.reindex([t for t in mem if t not in top]).mean())
        rows["S_all14 short basket"].append(s.reindex(bot).mean())
        rows["universe"].append(s.mean())
    for k, v in rows.items():
        print(f"  {k:28s} {np.nanmean(v):+.2f}")

    print("\n== 3/4. short sleeves on TOP_HALF (monthly, 10bp + 1%/yr borrow)")
    br = tilt.book_returns(); top = br["TOP_HALF"]
    legs = {
        "S_all14 (14 signals)": run.book_returns(-comp14, fwd1, univ, None)["long"],
        "S_8 (14 minus power)": run.book_returns(
            -cp.composite(sigs, [k for k in sigs if k not in cp.POWER_BETAS + cp.POWER_SHOCKS], univ),
            fwd1, univ, None)["long"],
        "S_sens (most power-sensitive)": run.book_returns(sens, fwd1, univ, None)["long"],
    }
    out = []
    for name, s in legs.items():
        s = s.reindex(br.index)
        ok = s.notna() & (top < 0)
        row = {"short": name, "leg ann": s.mean() * 12, "corr": s.corr(top),
               "down beta": np.polyfit(top[ok], s[ok], 1)[0], "months missing": int(s.isna().sum())}
        s = s.fillna(0.0)
        for kk, lab in ((0.25, "0.25x"),):
            r = top - kk * s - kk * tilt.BORROW / 12
            for w, (a, e) in run.WINDOWS.items():
                p = ev.perf(r.loc[a:e])
                row[f"SR {w}"] = p["sharpe"]; row[f"Cal {w}"] = tilt.calmar(r.loc[a:e])
                row[f"DD {w}"] = p["maxdd"]
            row["2020 ret"] = float((1 + r.loc["2020"]).prod() - 1)
            row["2022 ret"] = float((1 + r.loc["2022"]).prod() - 1)
        out.append(row)
    ref = {"short": "none (TOP_HALF)"}
    for w, (a, e) in run.WINDOWS.items():
        ref[f"SR {w}"] = ev.perf(top.loc[a:e])["sharpe"]; ref[f"Cal {w}"] = tilt.calmar(top.loc[a:e])
        ref[f"DD {w}"] = ev.perf(top.loc[a:e])["maxdd"]
    ref["2020 ret"] = float((1 + top.loc["2020"]).prod() - 1)
    ref["2022 ret"] = float((1 + top.loc["2022"]).prod() - 1)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.DataFrame([ref] + out).set_index("short").round(2).to_string())


if __name__ == "__main__":
    main()
