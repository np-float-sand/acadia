"""Plan step 3b: co-headquartered (geographic) peer momentum.

    python -m research.xsec_alpha_probe.geo

Parsons-Sabbatucci-Titman (RFS 2020): stocks of firms headquartered in the same
city lead each other, the effect is unrelated to analyst coverage, and it is
measured excluding same-industry firms so it is not industry momentum.

Links: SEC business address (city + state) from EDGAR submissions, for the same
655-firm pool as step 3a (eligible universe ever + current S&P 500). Industry =
2-digit SIC. Caveat: EDGAR gives today's address only (HQ moves are rare but real).

Signal (fixed 2026-09-24, before results) = equal-weight mean of same-city,
different-SIC2 peers' month-t returns, predicting the follower's month t+1; NaN
unless >= MIN_PEERS peers.
  geo_day_out  PRIMARY  peers outside the month's eligible universe, daytime returns
  geo_day_all           all peers (inside or outside), daytime
  geo_cc_out            outside peers, close-to-close
Same bar as run.py, then the benchmark test (TOP_HALF by composite: GRID-hedged
Sharpe 0.81, Calmar 0.41, ALPHA).
"""
from __future__ import annotations

import json

import pandas as pd

from research.xsec_alpha_probe import data, evaluate as ev, peers, run, tenk, tilt

MIN_PEERS = 2


def addresses(refresh: bool = False) -> pd.DataFrame:
    path = data.CACHE / "hq_addresses.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for t, cik in sorted(tenk.firm_ciks().items()):
        try:
            sub = tenk._get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
        except Exception as e:           # noqa: BLE001 -- one missing firm is fine
            print(f"  {t}: {e}")
            continue
        biz = sub.get("addresses", {}).get("business", {}) or {}
        rows.append({"ticker": t, "city": (biz.get("city") or "").strip().upper(),
                     "state": (biz.get("stateOrCountry") or "").strip().upper(),
                     "sic": sub.get("sic") or ""})
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def geo_signal(addr: pd.DataFrame, ret_m: pd.DataFrame, univ: pd.DataFrame, outside: bool) -> pd.DataFrame:
    a = addr[(addr["city"] != "")].copy()
    a["loc"] = a["city"] + ", " + a["state"]
    a["sic2"] = a["sic"].astype(str).str[:2]
    by_loc = a.groupby("loc")["ticker"].apply(list).to_dict()
    info = a.set_index("ticker")
    rows = {}
    for dt in univ.index:
        if dt not in ret_m.index:
            continue
        inside = set(univ.columns[univ.loc[dt]])
        r = ret_m.loc[dt]
        row = {}
        for f in inside & set(info.index):
            peers_ = [p for p in by_loc[info.at[f, "loc"]]
                      if p != f and info.at[p, "sic2"] != info.at[f, "sic2"]
                      and pd.notna(r.get(p)) and (not outside or p not in inside)]
            if len(peers_) >= MIN_PEERS:
                row[f] = float(r[peers_].mean())
        rows[dt] = row
    return pd.DataFrame(rows).T.sort_index()


def main() -> None:
    pd.set_option("display.width", 250, "display.max_columns", 40)
    univ = data.universe()
    px_m = data.monthly_prices(data.prices())
    fwd1 = px_m.pct_change(fill_method=None).shift(-1)
    fwd3 = px_m.shift(-3) / px_m - 1
    grid_fwd = data.monthly_prices(data.benchmarks())["GRID"].pct_change().shift(-1)
    day_m, cc_m = peers.monthly_returns(*peers.ohlc())
    addr = addresses()
    sigs = {"geo_day_out": geo_signal(addr, day_m, univ, True),
            "geo_day_all": geo_signal(addr, day_m, univ, False),
            "geo_cc_out": geo_signal(addr, cc_m, univ, True)}
    res = pd.DataFrame([run.evaluate_signal(k, v, univ, fwd1, fwd3, {"eq": None}, grid_fwd)
                        for k, v in sigs.items()]).set_index("signal")
    res.to_csv(data.CACHE / "results_geo.csv")
    run._print(res)

    book = tilt.book_membership(univ.index)
    comp = tilt.composite(univ, book)
    cols = univ.columns.union(book.columns)
    mask = univ.reindex(columns=cols, fill_value=False) | book.reindex(columns=cols, fill_value=False)
    geo_z = ev.prep(sigs["geo_day_out"].reindex(index=univ.index, columns=cols).where(mask))
    rows = []
    for name, score in {"TOP_HALF composite (benchmark)": comp,
                        "TOP_HALF geo_day_out alone": geo_z,
                        "TOP_HALF composite + geo": comp.add(geo_z, fill_value=0) / 2}.items():
        br = tilt.book_returns(score)
        hedged, _ = tilt.grid_hedge(br["TOP_HALF"], br["GRID"])
        rows += [{"book": name, **row} for row in tilt.stats(br["TOP_HALF"], hedged)]
    bench = pd.DataFrame(rows).set_index(["book", "window"])
    bench.to_csv(data.CACHE / "results_geo_book.csv")
    print(bench[["SR", "ann", "maxDD", "Calmar", "hedged SR", "alpha ann", "hedged Calmar",
                 "alpha t (NW1)", "label"]].round(2).to_string())


def placebo(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Shuffle geo_day_out z-scores among the names that have one, each month:
    separates 'which names have coverage' from 'what the signal says'."""
    import numpy as np
    univ = data.universe()
    day_m, _ = peers.monthly_returns(*peers.ohlc())
    sig = geo_signal(addresses(), day_m, univ, True)
    book = tilt.book_membership(univ.index)
    comp = tilt.composite(univ, book)
    cols = univ.columns.union(book.columns)
    mask = univ.reindex(columns=cols, fill_value=False) | book.reindex(columns=cols, fill_value=False)
    gz = ev.prep(sig.reindex(index=univ.index, columns=cols).where(mask))

    def hedged_sr(score):
        br = tilt.book_returns(score)
        h, _ = tilt.grid_hedge(br["TOP_HALF"], br["GRID"])
        return tilt.stats(br["TOP_HALF"], h)[0]["hedged SR"]

    rng, rows = np.random.default_rng(seed), []
    for _ in range(n):
        sh = gz.copy()
        for dt in sh.index:
            ok = sh.loc[dt].notna()
            sh.loc[dt, ok] = rng.permutation(sh.loc[dt, ok].to_numpy(copy=True))
        rows.append({"geo": hedged_sr(sh), "combo": hedged_sr(comp.add(sh, fill_value=0) / 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    if "--placebo" in sys.argv:
        print(placebo().describe(percentiles=[.5, .9]).round(2).to_string())
    else:
        main()
