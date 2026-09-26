"""Slack-grid holdout, 2016-01 -> 2018-12: does the investor rule stack survive a grid that is NOT tight?

    python research/electrification_strategy_probes/holdout_2016_18.py

No rule was tuned on these years. Everything is rebuilt point-in-time from filings as published:
  * membership (rules 1-5)  -- `auto_screen.build_membership`, same quantile/lookback, with 10-K wording
                               hits for 2012-16 (`holdout_fetch_hits.py`) added to the 2017+ cache;
  * long ranking            -- mean z of capex-light and revenue acceleration (as `tilt.composite`);
  * short score             -- equal-weight composite of the available signals (fundamental + price;
                               the power-market betas need ISO data from mid-2017, so most of the
                               window runs on 8 of the 14 signals);
  * rule 11                 -- real yield and real supplier revenue growth, as in the proposal.
The strategy code is `investor_book.build`, unchanged, with every trade charged 10bp.

Caveats: universe = today's listed companies (survivorship bias: flatters the long, removes failed
shorts); 3 years only; prices start 2015-01, so the first year's trailing windows are short.
Writes output_html/img/tophalf/holdout_2016_18.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import build_tophalf_proposal as t  # noqa: E402
import investor_book as ib  # noqa: E402
from capex_intensity_strategy import edgar, fundamentals, pipeline  # noqa: E402
from electrification_strategy import auto_screen, config  # noqa: E402
from research.xsec_alpha_probe import composites, data, evaluate as ev, signals as sg  # noqa: E402

H0, H1 = "2016-01-01", "2018-12-31"
HITS_FILES = ("vocabulary_hits.csv", "vocabulary_hits_short_universe.csv", "vocabulary_hits_2012_2016.csv")
WINS = {"2016-18": (H0, H1), "2016": ("2016", "2016"), "2017": ("2017", "2017"), "2018": ("2018", "2018")}


def load_hits() -> pd.DataFrame:
    frames = [pd.read_csv(os.path.join(config.DATA_DIR, f), comment="#") for f in HITS_FILES]
    h = pd.concat(frames, ignore_index=True)
    return h[h["phrase"] != "__done__"].dropna(subset=["file_date"])


def main():
    hits = load_hits()
    # ---- membership (rules 1-5), point in time ------------------------------------------------
    built = pipeline.build(start="2015-01-01", end=H1)
    score, mask, cands = built["score"], built["mask"], built["candidates"]
    power = set(cands.loc[cands["group"] == "POWER", "ticker"])
    panel = fundamentals.ttm_panel(edgar.fetch_facts({}), pd.date_range("2015-01-01", H1, freq="ME"))
    import build_a1a4_proposal as a1
    members = auto_screen.build_membership(score, mask, panel, hits, power_tickers=power,
                                           quantile=a1.QUANTILE, lookback=a1.LOOKBACK)
    members = members.loc[:H1]
    # ---- signals, rebuilt from 2015 ------------------------------------------------------------
    fpanel = sg.fundamental_panel(data.facts(), pd.date_range("2015-06-30", H1, freq="ME"))
    feats = [c for c in fpanel.columns if c not in ("date", "ticker")]
    fsig = {f: fpanel.pivot_table(index="date", columns="ticker", values=f) for f in feats}
    univ = data.universe()
    cols = univ.columns.union(members.columns)
    book = members.reindex(index=univ.index, method="ffill").reindex(columns=cols, fill_value=False).fillna(False)
    msk = univ.reindex(columns=cols, fill_value=False) | book.astype(bool)
    z = lambda x: ev.prep(x.reindex(index=univ.index, columns=cols).where(msk))  # noqa: E731
    comp = z(fsig["capex_light"]).add(z(fsig["rev_accel"]), fill_value=0) / 2
    px_m = data.monthly_prices(data.prices())
    bm = data.monthly_prices(data.benchmarks())
    sigs = dict(fsig)
    sigs.update(sg.price_signals(px_m))
    try:
        sigs.update(sg.power_signals(px_m, bm["SPY"]))
    except Exception as e:  # noqa: BLE001 -- power data optional before mid-2017
        print("power signals unavailable:", e)
    comp14 = composites.composite(sigs, list(sigs), univ)
    n_sig = pd.concat({k: v.reindex(index=univ.index).notna().any(axis=1) for k, v in sigs.items()}, axis=1).sum(axis=1)

    # ---- point the proposal's code at the holdout window --------------------------------------
    t.START, t.END = H0, H1
    sg.fundamental_signals = lambda *a, **k: fsig                         # supplier_growth reads this
    vocab = auto_screen.vocabulary_years(hits, config.AUTO_CUSTOMER_TERMS, list(range(2015, 2020)),
                                         lookback=a1.LOOKBACK)
    by_year: dict[int, set] = {}
    for tk, ys in vocab.items():
        for y in ys:
            by_year.setdefault(y, set()).add(tk)
    t.rule2_filter = lambda: (lambda me, idx: [x for x in idx if x not in by_year.get(me.year, set())])
    ib._CACHE.clear()
    ib._CACHE["comp14"] = comp14

    pub, _, px, _ = t.load()
    spy = px["SPY"].loc[H0:H1].pct_change()
    dfii = t.fetch_series(config.HEDGE_SHORT_SIGNAL, "2003-01-01", H1)
    ry = t.fetch_series(t.RATE_SIGNAL, "2003-01-01", H1).dropna()
    cash = t.fetch_series(t.CASH_SERIES, "2003-01-01", H1).dropna()
    g = t.supplier_growth(members)
    off = ry.clip(lower=10.0)

    def run(**kw):
        rate = kw.pop("ry", ry)
        params = dict(growth=g)
        params.update(kw)
        return ib.build(members, comp, px, spy, dfii, rate, cash, **params)

    def summ(r):
        out = {}
        for w, (a, e) in WINS.items():
            s = t.a1.stats(r.loc[a:e])
            out[w] = {"sharpe": round(s["sharpe"], 2), "calmar": round(s["calmar"], 2) if s["calmar"] else None,
                      "maxdd": round(s["maxdd"], 4), "cagr": round(s["cagr"], 4), "vol": round(s["vol"], 4)}
        return out

    final = run()
    ladder = [
        ("L0 rules 1-5, equal weight, no overlay, no short", dict(rank=False, overlay=False, short=None, ry=off, growth=None)),
        ("L1 + risk overlay", dict(rank=False, short=None, ry=off, growth=None)),
        ("L2 + ranking, top half", dict(long_hold=None, short=None, ry=off, growth=None)),
        ("L3 + hold buffer", dict(short=None, ry=off, growth=None)),
        ("L4 + short $0.25", dict(ry=off, growth=None)),
        ("L5 + rule 11, real yield only", dict(growth=None)),
        ("L6 + rule 11, both (final)", dict()),
    ]
    res = {"ladder": {}, "bench": {}}
    series = {}
    for name, kw in ladder:
        r = final["returns"] if name.startswith("L6") else run(**kw)["returns"]
        series[name] = r
        res["ladder"][name] = summ(r)
    for b in ["GRID", "PAVE", "SPY", "XLI", "XLU"]:
        if b in px.columns:
            res["bench"][b] = summ(px[b].pct_change().loc[H0:H1].dropna())
    # ranking effect split by rule-11 state: does it hurt only when rule 11 would be off?
    w = final["invested"]
    diff = (series["L2 + ranking, top half"] - series["L1 + risk overlay"]).loc[H0:H1]
    res["ranking_effect"] = {
        "ann_return_diff_when_rule11_on": round(float(diff[w.loc[H0:H1] > 0].mean() * config.ANN), 4),
        "ann_return_diff_when_rule11_off": (round(float(diff[w.loc[H0:H1] == 0].mean() * config.ANN), 4)
                                            if (w.loc[H0:H1] == 0).any() else None),
        "by_year": {y: round(float((1 + diff.loc[str(y)]).prod() - 1), 4) for y in (2016, 2017, 2018)}}
    on = w.loc[H0:H1]
    spells = [(str(s.index[0].date()), str(s.index[-1].date()), len(s))
              for _, s in on[on == 0].groupby((on != on.shift()).cumsum()[on == 0])]
    res["rule11"] = {"share_invested": round(float(on.mean()), 3), "off_spells": spells,
                     "real_yield_min": round(float(ry.loc[H0:H1].min()), 2),
                     "supplier_real_growth_by_quarter": {str(d.date()): round(float(v), 3)
                                                         for d, v in g.loc[H0:H1].resample("QE").last().items()}}
    res["breadth"] = {"long_mean": round(float(final["long_held"].loc[H0:H1].mean()), 1),
                      "long_min": int(final["long_held"].loc[H0:H1][final["long_held"].loc[H0:H1] > 0].min()),
                      "short_mean": round(float(final["short_held"].loc[H0:H1][final["short_held"].loc[H0:H1] > 0].mean()), 1),
                      "members_per_month": round(float(members.loc[H0:H1].sum(axis=1).mean()), 1),
                      "short_signals_available_mean": round(float(n_sig.loc[H0:H1].mean()), 1)}
    res["by_year"] = {y: {k: round(float((1 + s.loc[str(y)].dropna()).prod() - 1), 4)
                          for k, s in [("strategy", final["returns"]), ("L1", series["L1 + risk overlay"]),
                                       *[(b, px[b].pct_change()) for b in ["GRID", "SPY", "PAVE"] if b in px]]}
                      for y in (2016, 2017, 2018)}
    json.dump(res, open(os.path.join(t.OUT, "holdout_2016_18.json"), "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    rows = [{"book": k, **{f"{w} {m}": v[w][m] for w in ("2016-18",) for m in ("sharpe", "calmar", "maxdd", "cagr", "vol")},
             **{f"{y} ret": v[y]["cagr"] for y in ("2016", "2017", "2018")}}
            for k, v in {**res["ladder"], **res["bench"]}.items()]
    print(pd.DataFrame(rows).to_string(index=False))
    for k in ("ranking_effect", "rule11", "breadth", "by_year"):
        print(k, json.dumps(res[k]))


if __name__ == "__main__":
    main()
