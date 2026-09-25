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
ENERGY_SIC2 = frozenset({"13", "29"})      # oil & gas extraction, refining
# Operating HQ ZIPs for firms whose EDGAR address is a tax-inversion / foreign domicile.
# Only firms with a well-known US operating HQ; genuinely foreign firms keep their address.
OPERATING_HQ_ZIP: dict[str, str] = {
    "ETN": "44122",   # Eaton -- Beachwood OH (Cleveland); domicile Dublin
    "JCI": "53202",   # Johnson Controls -- Milwaukee; domicile Cork
    "NVT": "55416",   # nVent -- St. Louis Park MN; domicile London
    "PNR": "55422",   # Pentair -- Golden Valley MN; domicile London
    "TT": "28036",    # Trane -- Davidson NC (Charlotte); domicile Dublin
    "ALLE": "46032",  # Allegion -- Carmel IN (Indianapolis); domicile Dublin
    "STE": "44060",   # Steris -- Mentor OH (Cleveland); domicile Dublin
    "MDT": "55432",   # Medtronic -- Fridley MN (Minneapolis); domicile Galway
    "AON": "60601",   # Aon -- Chicago; domicile Dublin
    "TEL": "19312",   # TE Connectivity -- Berwyn PA (Philadelphia); domicile Galway
    "GRMN": "66062",  # Garmin -- Olathe KS (Kansas City); domicile Schaffhausen
    "ST": "02703",    # Sensata -- Attleboro MA; domicile Swindon
    "BG": "63141",    # Bunge -- St. Louis; domicile Geneva
}


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
                     "zip": (biz.get("zipCode") or "").strip()[:5],
                     "sic": sub.get("sic") or ""})
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def metro_locations(addr: pd.DataFrame) -> pd.DataFrame:
    """addr with `loc` = Census CBSA (metro area) from the HQ ZIP, after moving
    tax-inverted / foreign-domiciled firms to their operating HQ ZIP. Firms whose
    ZIP maps to no CBSA keep city+state."""
    a = addr.copy()
    a["zip"] = a["ticker"].map(OPERATING_HQ_ZIP).fillna(a["zip"])
    z = pd.read_csv(data.CACHE / "zip_cbsa.csv", dtype=str).set_index("zip")
    a["loc"] = a["zip"].map(z["cbsa_title"])
    a["loc"] = a["loc"].fillna(a["city"] + ", " + a["state"])
    return a


def geo_signal(addr: pd.DataFrame, ret_m: pd.DataFrame, univ: pd.DataFrame, outside: bool,
               drop_peer_sic2: frozenset[str] = frozenset()) -> pd.DataFrame:
    a = addr[(addr["city"] != "")].copy()
    if "loc" not in a:
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
                      and info.at[p, "sic2"] not in drop_peer_sic2
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


def metro_main() -> pd.DataFrame:
    """3b retry (fixed 2026-09-25, before results): metro-area (CBSA) peers with the
    operating-HQ fix; primary `metro` = outside-universe, different-SIC2 peers, daytime
    returns. Diagnostics: `metro_ex_energy` drops oil & gas peers (SIC 13, 29) to test
    whether the Houston links are just an oil factor; `city` / `city_ex_energy` repeat
    the first 3b build for comparison. Judged by Sharpe/Calmar of TOP_HALF by
    composite + signal vs the composite+city benchmark (hedged SR 0.87)."""
    pd.set_option("display.width", 250, "display.max_columns", 40)
    univ = data.universe()
    day_m, _ = peers.monthly_returns(*peers.ohlc())
    addr = addresses()
    metro = metro_locations(addr)
    sigs = {"city": geo_signal(addr, day_m, univ, True),
            "city_ex_energy": geo_signal(addr, day_m, univ, True, ENERGY_SIC2),
            "metro": geo_signal(metro, day_m, univ, True),
            "metro_ex_energy": geo_signal(metro, day_m, univ, True, ENERGY_SIC2)}
    book = tilt.book_membership(univ.index)
    comp = tilt.composite(univ, book)
    cols = univ.columns.union(book.columns)
    mask = univ.reindex(columns=cols, fill_value=False) | book.reindex(columns=cols, fill_value=False)
    rows = [{"book": "composite only", **r} for r in _top_half(comp)]
    for k, v in sigs.items():
        z = ev.prep(v.reindex(index=univ.index, columns=cols).where(mask))
        cov = (z.notna() & book.reindex(columns=cols, fill_value=False)).sum(axis=1).loc[run.START:]
        for r in _top_half(comp.add(z, fill_value=0) / 2):
            rows.append({"book": f"composite + {k}", "book names w/ signal": float(cov.median()), **r})
    out = pd.DataFrame(rows).set_index(["book", "window"])
    out.to_csv(data.CACHE / "results_geo_metro.csv")
    print(out[["book names w/ signal", "SR", "ann", "maxDD", "Calmar", "hedged SR", "alpha ann",
               "hedged Calmar", "label"]].round(2).to_string())
    return out


def _top_half(score: pd.DataFrame) -> list[dict]:
    br = tilt.book_returns(score)
    hedged, _ = tilt.grid_hedge(br["TOP_HALF"], br["GRID"])
    return tilt.stats(br["TOP_HALF"], hedged)


if __name__ == "__main__":
    import sys
    if "--metro" in sys.argv:
        metro_main()
    elif "--placebo" in sys.argv:
        print(placebo().describe(percentiles=[.5, .9]).round(2).to_string())
    else:
        main()
